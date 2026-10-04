#!/usr/bin/env python3
"""evalroute — cost per verified success, by lane, for model routing.

  python evalroute.py init                                  # example models.json + tasks.jsonl
  python evalroute.py run    -m models.json -t tasks.jsonl -k 3 [--only a,b] [--judge NAME]
  python evalroute.py grade  -t tasks.jsonl [--audit N]     # blind human grading, timed
  python evalroute.py report [--usd-per-hour 100] [--tol 0.0] [--csv out.csv]

Metrics (per lane x model):
  pass      mean over tasks of per-task pass rate; 95% CI = cluster bootstrap over tasks
  cov       tasks passed on >=1 of k samples. The long tail: retries can't buy these.
  pass^k    tasks passed on all k samples (reliability)
  all-in    (API $ + judge $ + human verify hours * usd_per_hour) / passes
            = cost of retry-until-pass when every attempt must be verified.
Route: per lane, among models with cov >= best_cov - tol, minimize all-in $/success.

Checks (task.check.type):
  exact  {"answer"}                    last "ANSWER: x" line, else last line
  regex  {"pattern"}
  python {"file","cmd","setup":{name:src},"timeout"}   last code block -> file; pass iff exit 0
  judge  {"rubric"}                    --judge model, "VERDICT: PASS|FAIL"; self-judging -> human
  human  {"rubric"}                    blind, via `grade` (model name hidden, order shuffled)
`grade --audit N` blind-regrades N auto-checked outputs to measure checker agreement.

Effort (REQUIRED on every model entry; run refuses to start without it):
  "effort": none|minimal|low|medium|high|xhigh|max, or "default" to deliberately send nothing.
  Provider defaults differ and drift (Kimi K3 native = max; Hermes sends medium; Opus 5.5 = medium,
  Opus 5 was high), so an unset effort means you don't know what you measured.
  openai -> body.reasoning_effort   anthropic -> output_config.effort (low..max)   cmd -> {effort}
  One entry per (model, effort): the report and router treat each as its own arm.
  New records carry a v2 run signature over the complete task/checker, effective
  model configuration including token cap/prices, and judge configuration.
  Legacy rows without a signature remain reportable, but cannot be resumed as v2.

Model api: openai (any OpenAI-compatible: Nous, OpenAI) | anthropic | cmd.
  cmd = agent-harness shim (e.g. Hermes): template gets {prompt_file} {system_file} {model} {effort};
  last stdout line may be JSON {"text":..., "usage":{"inp","out","cache_read"}},
  else stdout is the text and cost is unknown (flagged '?').
  Optional per model: "extra" (other API kwargs; not effort), "max_param" ("max_completion_tokens"
  for OpenAI reasoning models), "max_tokens" (per-entry cap; raise it for xhigh/max, since it bounds
  thinking + answer), "cached", "cache_write", "conc".
Records: append-only JSONL. `run` resumes only matching signatures; errored
cells retry. Mixed legacy/v2 or changed task/checker versions cannot be pooled
in one report; use separate output files for different taskset versions.
"""
import argparse, asyncio, hashlib, json, math, os, random, re, statistics, subprocess, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

INF, NAN = float("inf"), float("nan")
JUDGE = ("Grade the RESPONSE against the RUBRIC. Be strict: fluent but wrong is FAIL.\n\nTASK:\n{p}\n\n"
         "RUBRIC:\n{r}\n\nRESPONSE:\n{t}\n\nEnd with exactly one line: VERDICT: PASS or VERDICT: FAIL")


def jread(p): return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []
def jadd(p, r):
    with open(p, "a") as f: f.write(json.dumps(r, ensure_ascii=False) + "\n")
def key(r):
    legacy = f'{r["task"]}|{r["model"]}|{r["sample"]}|{r.get("cfg", "")}'
    return legacy + (f'|{r["sig"]}' if r.get("sig") else "")
def cfghash(m):
    return hashlib.sha256(json.dumps({k: m.get(k) for k in ("api", "model", "base_url", "effort", "extra", "cmd")},
                                     sort_keys=True).encode()).hexdigest()[:8]
def phash(t): return hashlib.sha256((t.get("system", "") + "\0" + t["prompt"]).encode()).hexdigest()[:10]

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()[:16]

def task_contract(t):
    """Prompt, system, lane, and full checker spec (including setup/rubric)."""
    return digest(t)

def model_contract(m, max_tokens):
    """Full request and price schedule; concurrency does not affect a response."""
    return digest({"model": {k: v for k, v in m.items() if k != "conc"},
                   "effective_max_tokens": max_tokens})

def run_signature(m, t, max_tokens, judge=None):
    active_judge = judge if t.get("check", {}).get("type") == "judge" and judge and judge["name"] != m["name"] else None
    return digest({"version": 2, "task": task_contract(t),
                   "model": model_contract(m, max_tokens),
                   "judge": (model_contract(active_judge, active_judge.get("max_tokens", max_tokens))
                             if active_judge else None),
                   "judge_prompt": JUDGE if active_judge else None})

def arm_contract(m, t, max_tokens, judge=None):
    """One arm across task checker types; keep the judge schedule in provenance.

    Whether a *particular task* needs a judge belongs in run_signature, not
    the arm key. Otherwise a mixed auto/judge taskset creates disjoint arms.
    """
    judge_for_arm = judge if judge and judge["name"] != m["name"] else None
    return digest({"arm": model_contract(m, max_tokens),
                   "judge": model_contract(judge_for_arm, judge_for_arm.get("max_tokens", max_tokens))
                   if judge_for_arm else None, "judge_prompt": JUDGE if judge_for_arm else None})


# ---------------------------------------------------------------- models
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
ANTHROPIC_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def validate(models):
    bad = []
    for m in models:
        n, e = m["name"], m.get("effort")
        if e is None:
            bad.append(f'{n}: missing "effort" (a level, or "default" to deliberately send none)')
        elif e != "default" and e not in EFFORTS:
            bad.append(f"{n}: unknown effort {e!r}; expected one of {sorted(EFFORTS)} or 'default'")
        elif m["api"] == "anthropic" and e not in ANTHROPIC_EFFORTS | {"default"}:
            bad.append(f"{n}: anthropic effort must be one of {sorted(ANTHROPIC_EFFORTS)}")
        elif m["api"] == "cmd" and e != "default" and "{effort}" not in m["cmd"]:
            bad.append(f"{n}: cmd has no {{effort}} placeholder, so effort={e} would be silently ignored")
        x = m.get("extra", {})
        if {"reasoning_effort", "output_config", "reasoning"} & (set(x) | set(x.get("extra_body", {}))):
            bad.append(f'{n}: set effort with the "effort" field, not inside "extra"')
    if bad: sys.exit("model config errors:\n  " + "\n  ".join(bad))
    d = [m["name"] for m in models if m.get("effort") == "default"]
    if d: print(f"note: provider-default effort (unpinned) for: {', '.join(d)}", file=sys.stderr)


_clients = {}
def client(m):
    if m["api"] == "cmd": return None
    if m["name"] not in _clients:
        k = os.environ.get(m["key_env"], "")
        if m["api"] == "anthropic":
            import anthropic; _clients[m["name"]] = anthropic.AsyncAnthropic(api_key=k)
        else:
            import openai
            base = os.path.expandvars(m["base_url"]) if m.get("base_url") else None
            _clients[m["name"]] = openai.AsyncOpenAI(api_key=k, base_url=base)
    return _clients[m["name"]]


async def call(m, system, prompt, max_tok):
    t0, x, c, e = time.perf_counter(), dict(m.get("extra", {})), client(m), m["effort"]
    if m["api"] == "anthropic":
        kw = dict(model=m["model"], max_tokens=max_tok, messages=[{"role": "user", "content": prompt}], **x)
        if e != "default": kw["output_config"] = {**kw.get("output_config", {}), "effort": e}
        if system: kw["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        r = await c.messages.create(**kw); u = r.usage
        text = "".join(b.text for b in r.content if b.type == "text")
        use = dict(inp=u.input_tokens, out=u.output_tokens,
                   cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                   cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0)
        trunc = r.stop_reason == "max_tokens"
    elif m["api"] == "openai":
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        eb = dict(x.pop("extra_body", {}))
        if e != "default": eb["reasoning_effort"] = e  # via extra_body: works on any SDK version / compatible endpoint
        r = await c.chat.completions.create(model=m["model"], messages=msgs, **({"extra_body": eb} if eb else {}),
                                            **{m.get("max_param", "max_tokens"): max_tok}, **x)
        u = r.usage
        cr = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        text = r.choices[0].message.content or ""
        use = dict(inp=u.prompt_tokens - cr, out=u.completion_tokens, cache_read=cr, cache_write=0)
        trunc = r.choices[0].finish_reason == "length"
    else:
        with tempfile.TemporaryDirectory() as d:
            pf, sf = Path(d, "prompt.txt"), Path(d, "system.txt")
            pf.write_text(prompt); sf.write_text(system or "")
            cmd = m["cmd"].format(prompt_file=pf, system_file=sf, model=m.get("model", ""), effort=e)
            p = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE,
                                                      stderr=asyncio.subprocess.PIPE)
            so, se = await asyncio.wait_for(p.communicate(), m.get("timeout", 3600))
            if p.returncode: raise RuntimeError(se.decode()[-500:])
        out = so.decode(); last = (out.strip().splitlines() or [""])[-1]
        try:
            j = json.loads(last)
            text = j["text"]; use = {**dict(inp=0, out=0, cache_read=0, cache_write=0), **j.get("usage", {})}
        except Exception:
            text, use = out, None
        trunc = False
    return text, use, time.perf_counter() - t0, trunc


def cost(m, u):
    if u is None: return None
    return (u["inp"] * m["in"] + u["out"] * m["out"] + u["cache_read"] * (m.get("cached") or m["in"])
            + u.get("cache_write", 0) * (m.get("cache_write") or m["in"])) / 1e6


# ---------------------------------------------------------------- checks
def last_code(t):
    b = re.findall(r"```[\w+-]*\n(.*?)```", t, re.S)
    return b[-1] if b else t


def auto_check(task, text):
    c = task.get("check", {"type": "human"}); ty = c["type"]
    if ty == "exact":
        a = re.findall(r"(?im)^\s*\**ANSWER:?\**:?\s*(.+?)\s*$", text)
        got = a[-1] if a else (text.strip().splitlines() or [""])[-1]
        return got.strip().strip("$*`. ") == str(c["answer"]).strip()
    if ty == "regex":
        return re.search(c["pattern"], text) is not None
    if ty == "python":
        with tempfile.TemporaryDirectory() as d:
            for n, s in c.get("setup", {}).items(): Path(d, n).write_text(s)
            Path(d, c["file"]).write_text(last_code(text))
            try:
                return subprocess.run(c["cmd"], shell=True, cwd=d, capture_output=True,
                                      timeout=c.get("timeout", 120)).returncode == 0
            except subprocess.TimeoutExpired:
                return False
    return None  # judge / human


# ---------------------------------------------------------------- run
async def run(a):
    tasks, model_list = jread(a.tasks), json.load(open(a.models))
    for kind, ids in (("task", [t["id"] for t in tasks]), ("model", [m["name"] for m in model_list])):
        if len(ids) != len(set(ids)):
            sys.exit(f"duplicate {kind} id in input; refusing to pool distinct cells")
    if not tasks or not model_list or a.k < 1 or a.max_tokens < 1:
        sys.exit("tasks, models, k and max_tokens must be nonempty/positive")
    allm = {m["name"]: m for m in model_list}
    models = [allm[n] for n in a.only.split(",")] if a.only else list(allm.values())
    J = allm[a.judge] if a.judge else None
    validate(models + ([J] if J else []))
    old_rows = jread(a.out)
    expected_tasks = {t["id"]: task_contract(t) for t in tasks}
    for r in old_rows:
        if "text" in r and r.get("task") in expected_tasks and (r.get("contract_version") != 2
                                                                  or r.get("ph") != expected_tasks[r["task"]]):
            sys.exit(f"mixed task contract for {r['task']} in {a.out}; use a separate --out "
                     "for changed prompts/checkers or legacy results before any provider calls")
    done = {key(r) for r in old_rows if "text" in r}
    sem = {m["name"]: asyncio.Semaphore(m.get("conc", a.conc)) for m in models}

    async def one(t, m, s):
        chk = t.get("check", {"type": "human"})
        maximum = m.get("max_tokens", a.max_tokens)
        active_judge = J if chk["type"] == "judge" and J and J["name"] != m["name"] else None
        r = dict(task=t["id"], lane=t["lane"], model=m["name"], model_id=m["model"], effort=m["effort"],
                 cfg=arm_contract(m, t, maximum, J), sig=run_signature(m, t, maximum, J),
                 contract_version=2, sample=s, ph=task_contract(t), check=chk["type"], ts=time.time())
        try:
            async with sem[m["name"]]:
                text, u, dt, tr = await call(m, t.get("system"), t["prompt"], maximum)
        except Exception as e:
            jadd(a.out, {**r, "error": repr(e)[:500]}); print(f"ERR {key(r)} {e!r}"[:240], file=sys.stderr); return
        r.update(text=text, usage=u, cost=cost(m, u), latency=round(dt, 2), truncated=tr, jcost=0.0)
        r["passed"] = await asyncio.to_thread(auto_check, t, text)
        if chk["type"] == "judge":
            r["judge_status"] = "pending"  # no usable grade until a real verdict or blind human grade
        if active_judge:
            try:
                jt, ju, _, _ = await call(J, None, JUDGE.format(p=t["prompt"], r=chk.get("rubric", ""), t=text), J.get("max_tokens", a.max_tokens))
                v = re.findall(r"VERDICT:\s*(PASS|FAIL)", jt)
                r["passed"], r["jcost"] = (v[-1] == "PASS") if v else None, cost(J, ju) or 0.0
                r["judge_status"] = "graded" if v else "pending_invalid_verdict"
            except Exception as e:
                r["judge_status"] = "pending_error"
                print(f"JUDGE ERR {key(r)} {e!r}"[:240], file=sys.stderr)
        jadd(a.out, r)
        c = "?" if r["cost"] is None else f'{r["cost"]:.4f}'
        print(f'{key(r):<48} pass={r["passed"]!s:<5} ${c}{" TRUNC" if tr else ""}')

    jobs = [one(t, m, s) for t in tasks for m in models for s in range(a.k)
            if key(dict(task=t["id"], model=m["name"], sample=s,
                        cfg=arm_contract(m, t, m.get("max_tokens", a.max_tokens), J),
                        sig=run_signature(m, t, m.get("max_tokens", a.max_tokens), J))) not in done]
    random.shuffle(jobs)
    print(f"{len(jobs)} cells to run ({len(done)} already done)")
    await asyncio.gather(*jobs)


# ---------------------------------------------------------------- grade
def grade(a):
    tasks = {t["id"]: t for t in jread(a.tasks)}
    rows = jread(a.out)
    runs = {key(r): r for r in rows if "text" in r}
    for r in runs.values():
        t = tasks.get(r["task"])
        expected = task_contract(t) if r.get("contract_version") == 2 else phash(t) if t else None
        if t is None or r.get("ph") != expected:
            sys.exit(f"mixed or stale task contract for {r['task']}: grade with the matching tasks.jsonl")
    graded = {r["grade_of"] for r in rows if "grade_of" in r}
    todo = [r for k, r in runs.items() if r["passed"] is None and k not in graded]
    aud = [r for k, r in runs.items() if r["passed"] is not None and k not in graded]
    random.shuffle(aud); todo += aud[:a.audit]; random.shuffle(todo)
    for i, r in enumerate(todo):
        t, audit = tasks[r["task"]], r["passed"] is not None
        print(f"\n{'=' * 78}\n[{i + 1}/{len(todo)}] lane={r['lane']} task={r['task']}{'  (audit)' if audit else ''}"
              f"\n--- PROMPT\n{t['prompt']}\n--- RUBRIC\n{t.get('check', {}).get('rubric', '(none)')}"
              f"\n--- RESPONSE\n{r['text']}\n")
        t0 = time.time()
        while (v := input("[p]ass [f]ail [s]kip [q]uit > ").strip().lower()) not in ("p", "f", "s", "q"): pass
        if v == "q": break
        if v == "s": continue
        jadd(a.out, dict(grade_of=key(r), passed=v == "p", verify_s=round(time.time() - t0, 1), audit=audit))


# ---------------------------------------------------------------- report
def pass_rate(tasks):  # tasks: list of per-task lists of bools
    return sum(sum(v) / len(v) for v in tasks) / len(tasks) if tasks else NAN


def boot_ci(per, B=2000, rng=random.Random(0)):
    v = list(per.values())
    if len(v) < 2: return NAN, NAN
    xs = sorted(pass_rate([rng.choice(v) for _ in v]) for _ in range(B))
    return xs[int(0.025 * B)], xs[int(0.975 * B) - 1]


def report(a):
    rows = jread(a.out)
    runs = {key(r): r for r in rows if "text" in r}
    versions = {r.get("contract_version", 1) for r in runs.values()}
    if len(versions) > 1 or versions - {1, 2}:
        sys.exit("mixed contract versions in results; report legacy and v2 runs separately")
    task_versions = defaultdict(set)
    for r in runs.values():
        task_versions[r["task"]].add(r.get("ph"))
    mixed = [t for t, versions_for_task in task_versions.items() if len(versions_for_task) > 1]
    if mixed:
        sys.exit(f"mixed task prompt/checker versions in results: {', '.join(sorted(mixed))}; report separately")
    taskset_path = Path(getattr(a, "tasks", ""))
    declared = {}
    if taskset_path.is_file():
        for t in jread(taskset_path):
            if not isinstance(t, dict) or not t.get("id") or not t.get("lane") or t["id"] in declared:
                sys.exit(f"invalid/duplicate task id or lane in {taskset_path}")
            declared[t["id"]] = t
        for r in runs.values():
            t = declared.get(r["task"])
            contract = (task_contract(t) if r.get("contract_version") == 2 else phash(t)) if t else None
            if t is None or t["lane"] != r["lane"] or r.get("ph") != contract:
                sys.exit(f"task {r['task']}: results do not match declared --tasks file; report separately")
    taskset_known = bool(declared)
    if not taskset_known:
        print("WARN no declared --tasks file: showing diagnostics only; no arm is routeable", file=sys.stderr)
    grades = {r["grade_of"]: r for r in rows if "grade_of" in r}
    cells, agree = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: [0, 0])
    cfgs, eff = defaultdict(set), {}
    for r in runs.values(): cfgs[r["model"]].add(r.get("cfg", ""))
    for m, c in cfgs.items():
        if len(c) > 1: print(f"WARN model {m}: {len(c)} configs in results; reported separately as {m}#<cfg>", file=sys.stderr)
    arm = lambda r: r["model"] if len(cfgs[r["model"]]) == 1 else f'{r["model"]}#{r.get("cfg", "")[:6]}'
    errs = defaultdict(int)
    for r in rows:
        if ("error" in r and key(r) not in runs and
                r.get("cfg", "") in cfgs[r["model"]]):
            errs[(r.get("lane"), arm(r))] += 1
    for k, r in runs.items():
        p, vs, g = r["passed"], 0.0, grades.get(k)
        if g and g["audit"]:
            agree[r["check"]][0] += g["passed"] == p; agree[r["check"]][1] += 1
        elif g:
            p, vs = g["passed"], g["verify_s"]
        eff[arm(r)] = r.get("effort", "?")
        cells[(r["lane"], arm(r))][r["task"]].append(
            dict(p=p, c=r["cost"], j=r["jcost"], v=vs, out=(r["usage"] or {}).get("out"), lat=r["latency"], tr=r.get("truncated")))

    # The independent taskset comes from the declared task file, not just the
    # cells reached by at least one arm. A missing task, sample, or grade makes
    # an arm incomplete; without that file all rows are diagnostics only.
    lane_tasks = defaultdict(set)
    if taskset_known:
        for t in declared.values():
            lane_tasks[t["lane"]].add(t["id"])
    else:
        for (lane, _), task_rows in cells.items():
            lane_tasks[lane].update(task_rows)
    expected_k = getattr(a, "k", None) or max(
        (len(v) for task_rows in cells.values() for v in task_rows.values()), default=1)
    stats = []
    for (lane, mod), T in cells.items():
        xs = [x for v in T.values() for x in v]
        g = [x for x in xs if x["p"] is not None]
        per = {t: [x["p"] for x in v if x["p"] is not None] for t, v in T.items()}
        per = {t: v for t, v in per.items() if v}
        tasks_in_lane = lane_tasks[lane]
        complete = (taskset_known and set(T) == tasks_in_lane and errs[(lane, mod)] == 0
                    and all(len(T[t]) == expected_k and all(x["p"] is not None for x in T[t])
                            for t in tasks_in_lane))
        P = sum(sum(v) for v in per.values())
        api = sum((x["c"] or 0) + (x["j"] or 0) for x in xs)
        hrs = sum(x["v"] for x in g) / 3600
        lo, hi = boot_ci(per)
        outs = [x["out"] for x in xs if x["out"] is not None]
        stats.append(dict(
            lane=lane, model=mod, effort=eff.get(mod, "?"), n=len(g), pend=len(xs) - len(g), trunc=sum(bool(x["tr"]) for x in xs),
            err=errs[(lane, mod)], complete=complete, expected=len(tasks_in_lane) * expected_k,
            pass_=pass_rate(list(per.values())), lo=lo, hi=hi,
            cov=sum(bool(per.get(t)) and any(per[t]) for t in tasks_in_lane) / len(tasks_in_lane) if tasks_in_lane else NAN,
            passk=sum(len(per.get(t, [])) == expected_k and all(per[t]) for t in tasks_in_lane) / len(tasks_in_lane) if tasks_in_lane else NAN,
            try_usd=api / len(xs) if xs else NAN, succ_usd=api / P if P else INF,
            vmin=hrs * 60 / len(g) if g else NAN,
            allin=(api + hrs * a.usd_per_hour) / P if P else INF,
            known=all(x["c"] is not None for x in xs),
            med_out=statistics.median(outs) if outs else NAN,
            p50s=statistics.median([x["lat"] for x in xs])))

    hdr = (f'  {"model":<22}{"effort":<8}{"n":>4}{"pend":>5}  {"pass [95% CI]":<19}{"cov":>5}{"pass^k":>7}'
           f'{"$/try":>9}{"$/succ":>9}{"vmin":>6}{"all-in$/s":>10}{"med_out":>8}{"p50s":>6}')
    for lane in sorted({s["lane"] for s in stats}):
        L = [s for s in stats if s["lane"] == lane]
        comparable = [s for s in L if s["complete"] and s["known"] and s["n"] > 0]
        bc = max((s["cov"] for s in comparable if not math.isnan(s["cov"])), default=NAN)
        elig = [s for s in comparable if s["cov"] >= bc - a.tol and s["allin"] < INF]
        route = min(elig, key=lambda s: s["allin"])["model"] if elig else None
        front = {s["model"] for s in comparable if not any(
            o["allin"] <= s["allin"] and o["pass_"] >= s["pass_"] and (o["allin"] < s["allin"] or o["pass_"] > s["pass_"])
            for o in comparable)}
        print(f"\nlane: {lane}   (route -> {route}; * = pareto on all-in$/succ vs pass)\n{hdr}")
        for s in sorted(L, key=lambda s: s["allin"]):
            q = "" if s["known"] else "?"
            flags = "".join([f" trunc={s['trunc']}" if s["trunc"] else "", f" err={s['err']}" if s["err"] else "",
                             f" incomplete ({s['n']}/{s['expected']} graded)" if not s["complete"] else "",
                             "  <- route" if s["model"] == route else ""])
            print(f'{"*" if s["model"] in front else " "} {s["model"]:<22}{s["effort"]:<8}{s["n"]:>4}{s["pend"]:>5}  '
                  f'{s["pass_"]:.2f} [{s["lo"]:.2f},{s["hi"]:.2f}]  {s["cov"]:>5.2f}{s["passk"]:>7.2f}'
                  f'{s["try_usd"]:>9.4f}{q}{s["succ_usd"]:>9.4f}{q}{s["vmin"]:>6.2f}{s["allin"]:>10.4f}{q}'
                  f'{s["med_out"]:>8.0f}{s["p50s"]:>6.1f}{flags}')

    for ck, (ok, n) in agree.items():
        print(f"checker audit [{ck}]: human agrees {ok}/{n} ({ok / n:.0%})")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)


# ---------------------------------------------------------------- init
NOUS = dict(api="openai", base_url="$NOUS_BASE_URL", key_env="NOUS_API_KEY")
OAI = dict(api="openai", key_env="OPENAI_API_KEY", max_param="max_completion_tokens")
ANT = dict(api="anthropic", key_env="ANTHROPIC_API_KEY")
_BASE = [  # (name, provider, model id, prices). medium = what Hermes sends by default; max = what most benchmarks used.
    ("ds-v4.1-flash", NOUS, "deepseek/deepseek-v4.1-flash", dict(inp=0.07, out=0.30)),
    ("glm-5.3-flash", NOUS, "z-ai/glm-5.3-flash", dict(inp=0.12, out=0.40)),
    ("glm-5.3", NOUS, "z-ai/glm-5.3", dict(inp=0.91, out=2.86)),
    ("kimi-k3", NOUS, "moonshotai/kimi-k3", dict(inp=0.88, out=10.53)),
    ("qwen3.8-max", NOUS, "qwen/qwen3.8-max", dict(inp=2.00, out=6.00)),
    ("gpt-6-luna", OAI, "gpt-6-luna", dict(inp=0.10, out=0.50, cached=0.01)),
    ("gpt-6-sol", OAI, "gpt-6-sol", dict(inp=2.00, out=10.00, cached=0.20)),
    ("opus-5.5", ANT, "claude-opus-5-5", dict(inp=4.00, out=20.00, cached=0.20, cache_write=5.00)),
]
EX_MODELS = [{"name": f"{n}@{e}", **p, "model": mid, "effort": e, "in": c["inp"], "out": c["out"],
              **{k: v for k, v in c.items() if k not in ("inp", "out")},
              **({"max_tokens": 64000} if e == "max" else {})}
             for n, p, mid, c in _BASE for e in ("medium", "max")]
EX_TASKS = [
    {"id": "code-lse", "lane": "coding",
     "prompt": "Write a numerically stable logsumexp(x, axis=-1) in NumPy. It must handle -inf entries and "
               "rows that are entirely -inf (return -inf, no NaN). Reply with one python code block.",
     "check": {"type": "python", "file": "sol.py", "cmd": "python test.py", "setup": {"test.py":
        "import numpy as np, warnings\nwarnings.simplefilter('ignore')\nfrom sol import logsumexp\n"
        "x=np.array([[1000.,1000.],[-np.inf,0.],[-np.inf,-np.inf]])\nr=logsumexp(x)\n"
        "assert np.allclose(r[:2],[1000+np.log(2),0.]) and np.isneginf(r[2]), r\n"}}},
    {"id": "math-cayley7", "lane": "math",
     "prompt": "How many labeled trees are there on 7 vertices? Give a one-paragraph derivation, then a final line 'ANSWER: <integer>'.",
     "check": {"type": "exact", "answer": "16807"}},
    {"id": "abs-monoid", "lane": "math-abstraction",
     "prompt": "Give the minimal algebraic structure under which prefix-sum (scan) admits an O(log n)-depth "
               "parallel algorithm, state it precisely, and show why associativity is necessary but commutativity is not.",
     "check": {"type": "human", "rubric": "Identifies semigroup/monoid (associativity; identity optional). "
               "Counterexample showing non-associative op breaks tree reduction. Notes commutativity unused. No false claims."}},
    {"id": "align-rm-optimum", "lane": "alignment",
     "prompt": "Critique: 'A reward model trained to convergence on human preference comparisons has an optimum "
               "aligned with human values.' Give the two strongest failure modes, with a concrete mechanism for each.",
     "check": {"type": "judge", "rubric": "Must give mechanistic failure modes (e.g. overoptimization/Goodhart "
               "off-distribution, labeler error or preferences != values, Bradley-Terry misspecification). "
               "Vague 'bias' claims without mechanism fail."}},
    {"id": "prose-lab", "lane": "prose",
     "prompt": "Write 200 words of literary prose: a researcher at 3am realizes the loss curve she's been proud of is a bug.",
     "check": {"type": "human", "rubric": "Specific, unclichéd, earns its emotional turn, no purple filler."}},
]


def init(a):
    for p, obj in ((a.models, EX_MODELS), (a.tasks, EX_TASKS)):
        if Path(p).exists(): print(f"exists, skipping: {p}"); continue
        with open(p, "w") as f:
            if p.endswith(".jsonl"): f.writelines(json.dumps(t) + "\n" for t in obj)
            else: json.dump(obj, f, indent=1)
        print(f"wrote {p}")


# ---------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["init", "run", "grade", "report"])
    ap.add_argument("-m", "--models", default="models.json")
    ap.add_argument("-t", "--tasks", default="tasks.jsonl")
    ap.add_argument("-o", "--out", default="runs.jsonl")
    ap.add_argument("-k", type=int, default=3, help="samples per task x model")
    ap.add_argument("--only", help="comma-separated model names")
    ap.add_argument("--judge", help="model name used for judge checks")
    ap.add_argument("--conc", type=int, default=4, help="default concurrency per model")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--audit", type=int, default=0, help="blind-regrade N auto-checked outputs")
    ap.add_argument("--usd-per-hour", type=float, default=100.0, help="value of your verification time")
    ap.add_argument("--tol", type=float, default=0.0, help="coverage slack when routing")
    ap.add_argument("--csv")
    a = ap.parse_args()
    {"init": init, "grade": grade, "report": report}.get(a.cmd, lambda a: asyncio.run(run(a)))(a)
