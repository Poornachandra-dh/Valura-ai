from typing import Dict, Any
from src.classifier import ClassificationResult

def route(classification: ClassificationResult, user: Dict[str, Any], llm=None) -> Dict[str, Any]:
    """
    Routes the classified query to the target agent.
    If the agent is portfolio_health, routes to the actual implementation.
    Otherwise, routes to a Stub Agent.
    """
    if classification.agent == "portfolio_health":
        from src.agents import portfolio_health
        # Pass the classification intent so the agent knows exactly what the user asked
        return portfolio_health.run(user=user, llm=llm, classification=classification)
    else:
        # Stub Agent response
        return {
            "intent": classification.intent,
            "extracted_entities": classification.entities,
            "target_agent": classification.agent,
            "message": f"The {classification.agent} is not implemented in this build."
        }
