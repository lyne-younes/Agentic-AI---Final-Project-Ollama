"""Conditional edges of the graph. They only read state; LangGraph ignores changes made here."""

from app.agents.graph.state import AgentState

ORCHESTRATOR = "orchestrator"
RETRIEVE_CONTEXT = "retrieve_context"
QA_AGENT = "qa_agent"
STUDY_TOOLS_AGENT = "study_tools_agent"
SUMMARY_AGENT = "summary_agent"
STUDY_PLANNER_AGENT = "study_planner_agent"
RECORD_RESULT = "record_result"
ASK_CLARIFICATION = "ask_clarification"
COMPOSE_RESPONSE = "compose_response"

AGENT_NODE_FOR_INTENT = {
    "qa_agent": QA_AGENT,
    "study_tools_agent": STUDY_TOOLS_AGENT,
    "summary_agent": SUMMARY_AGENT,
    "study_planner_agent": STUDY_PLANNER_AGENT,
}

# Agents that work from retrieved course material (the planner uses the calendar instead).
RETRIEVAL_AGENTS = (QA_AGENT, STUDY_TOOLS_AGENT, SUMMARY_AGENT)


def route_after_orchestrator(state: AgentState) -> str:
    """Read the orchestrator's last decision and reach the node that carries it out."""
    if state.get("needs_clarification"):
        return ASK_CLARIFICATION
    if state.get("done"):
        return COMPOSE_RESPONSE
    if state["pending_delegation"] == STUDY_PLANNER_AGENT:
        return STUDY_PLANNER_AGENT
    return RETRIEVE_CONTEXT


def route_after_retrieval(state: AgentState) -> str:
    return AGENT_NODE_FOR_INTENT[state["pending_delegation"]]
