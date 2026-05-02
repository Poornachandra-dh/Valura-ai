import re
from pydantic import BaseModel
from typing import Optional

class SafetyVerdict(BaseModel):
    blocked: bool
    category: Optional[str] = None
    message: Optional[str] = None

# We use regexes that target *intent to act* or *requests for help* rather than just the topic.
# Educational queries typically ask "what is", "how does", "explain", "are", etc.
SAFETY_RULES = [
    {
        "category": "insider_trading",
        "pattern": re.compile(r"(unannounced|confidential|non-public|before tomorrow's announcement|tip about earnings).*(buy|sell|trade|load up)|(buy|sell|trade|load up).*(unannounced|confidential|non-public|before tomorrow's announcement|tip about earnings)", re.IGNORECASE),
        "message": "I cannot assist with trading based on material non-public information. This constitutes insider trading, which is strictly prohibited."
    },
    {
        "category": "market_manipulation",
        "pattern": re.compile(r"(pump up|coordinated buying|wash trade|create volume).* (before selling|scheme|between two accounts|stock)|(help me|design a|how can i) (pump|wash trade|coordinated buying)", re.IGNORECASE),
        "message": "I cannot provide strategies or assistance intended to artificially manipulate market prices or volumes."
    },
    {
        "category": "money_laundering",
        "pattern": re.compile(r"(without reporting it|avoid the .* reporting threshold|obscure the source|hide .* from the tax)", re.IGNORECASE),
        "message": "I cannot assist with structuring transactions to evade reporting requirements or obscure the source of funds."
    },
    {
        "category": "guaranteed_returns",
        "pattern": re.compile(r"(guarantee me|promise me|100% certain|foolproof way)", re.IGNORECASE),
        "message": "I cannot guarantee returns or promise risk-free profits. All investments carry risk."
    },
    {
        "category": "reckless_advice",
        "pattern": re.compile(r"(all my retirement .* in crypto|margin loan to buy|entire emergency fund into|mortgage my house for)", re.IGNORECASE),
        "message": "I cannot recommend or endorse reckless financial strategies that jeopardise your essential capital or involve excessive leverage."
    },
    {
        "category": "sanctions_evasion",
        "pattern": re.compile(r"(bypass OFAC|invest in a sanctioned .* without it being traced)", re.IGNORECASE),
        "message": "I cannot assist with transactions intended to bypass international sanctions or regulations."
    },
    {
        "category": "fraud",
        "pattern": re.compile(r"(draft a fake)", re.IGNORECASE),
        "message": "I cannot assist in creating fraudulent documents or claims."
    }
]

def check(query: str) -> SafetyVerdict:
    """
    Synchronous local safety guard.
    Runs fast regex checks against the query to block harmful intent.
    """
    for rule in SAFETY_RULES:
        if rule["pattern"].search(query):
            return SafetyVerdict(
                blocked=True,
                category=rule["category"],
                message=rule["message"]
            )
            
    return SafetyVerdict(blocked=False)
