"""Tool schemas for the evalroute plugin — what the model reads."""

EVALROUTE_ROUTE = {
    "name": "evalroute_route",
    "description": (
        "Route a task to the right (model, reasoning effort) arm before starting work. "
        "Use when the user asks which model to use for a task, before a long or expensive "
        "session, when switching models mid-project, or when planning subagent delegation. "
        "Returns a route card with the lane, the model, the effort, the escalation option, "
        "and the provenance of the recommendation. Strong keyword rules run locally; "
        "weak-signal tasks may call the host LLM and consume tokens. Measured lanes and "
        "prior-only lanes are labeled in the card."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task description to classify. A short phrase or the "
                               "first sentence is enough; the classifier matches keywords.",
            },
            "lane": {
                "type": "string",
                "description": "Optional: pin the lane explicitly (one of the route table's "
                               "lane ids, e.g. 'math-first-principles') to skip classification.",
            },
            "replace_route_id": {
                "type": "string",
                "description": "Optional with lane: the prior route card's explicit ID. Only that "
                               "pending route is consumed and corrected; task text alone never "
                               "identifies a session.",
            },
        },
        "required": ["task"],
    },
}
