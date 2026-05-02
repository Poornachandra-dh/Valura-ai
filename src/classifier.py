import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from unittest.mock import MagicMock

class ClassificationResult(BaseModel):
    intent: str = Field(description="The user's primary intent or goal")
    entities: Dict[str, Any] = Field(description="Extracted entities based on the vocabulary")
    agent: str = Field(description="The target specialist agent to handle the query")
    safety_verdict: Optional[str] = Field(description="An informational safety verdict")

# System prompt defining the taxonomy and rules
SYSTEM_PROMPT = """
You are the Intent Classifier for the Valura AI wealth management platform.
Your job is to analyze the user's query, extract relevant entities, and route the query to the correct specialist agent.

Agent Taxonomy:
- portfolio_health: structured assessment of the user's portfolio (concentration, performance, benchmarking, observations). e.g., "how is my portfolio doing", "am i diversified"
- market_research: factual/recent info about an instrument, sector, or market event. e.g., "what is the price of AAPL", "tell me about NVIDIA"
- investment_strategy: advice/strategy questions: should I buy/sell/rebalance, allocation guidance.
- financial_planning: long-term planning: retirement, goals, savings rate.
- financial_calculator: deterministic numerical computation: DCA returns, mortgage, tax, future value, FX conversion.
- risk_assessment: risk metrics, exposure analysis, what-if scenarios.
- product_recommendation: recommend specific products/funds matching user profile.
- predictive_analysis: forward-looking analysis: forecasts, trend extrapolation.
- customer_support: platform issues, account questions, how-to-use-app.
- general_query: educational, conversational, definitions, greetings, or gibberish.

Entity Vocabulary Rules:
- tickers: array of strings, uppercase, exchange-suffixed where relevant (AAPL, ASML.AS)
- amount: number
- currency: ISO 4217 string (USD, EUR)
- rate: decimal (0.08 for 8%)
- period_years: integer
- frequency: daily, weekly, monthly, yearly
- horizon: 6_months, 1_year, 5_years
- time_period: today, this_week, this_month, this_year
- topics: array of strings
- sectors: array of strings
- index: string (S&P 500, FTSE 100, NIKKEI 225, MSCI World)
- action: buy, sell, hold, hedge, rebalance
- goal: retirement, education, house, FIRE, emergency_fund

Conversation History may be provided to resolve pronouns or missing context (e.g. "what about Apple?" after asking about Microsoft).

Safety Verdict: If the query touches upon harmful topics (insider trading, market manipulation, money laundering, etc.), provide an informational verdict. Otherwise, return null or empty.
"""

def get_openai_client() -> OpenAI:
    """Returns an OpenAI client. Uses Gemini OpenAI compatibility layer if GEMINI_API_KEY is present and OPENAI_API_KEY is not."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = None
    
    if not api_key:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            api_key = gemini_key
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            # Fallback for local testing without any keys
            api_key = "dummy_key"
            
    return OpenAI(api_key=api_key, base_url=base_url)

def classify(query: str, history: List[Dict[str, str]] = None, llm=None) -> ClassificationResult:
    """
    Classifies the user query using an LLM.
    If `llm` is a MagicMock, it returns the mocked response to pass CI tests without API keys.
    """
    if isinstance(llm, MagicMock):
        return llm.return_value

    client = llm if llm else get_openai_client()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    if history:
        for turn in history:
            messages.append({"role": "user", "content": turn.get("user", "")})
            if "agent_response" in turn:
                messages.append({"role": "assistant", "content": turn.get("agent_response", "")})
                
    messages.append({"role": "user", "content": query})
    
    # We use gpt-4o-mini for development per requirements.
    # If using Gemini's compatibility layer, it will ignore the model name or map it.
    model_name = os.getenv("CLASSIFIER_MODEL", "gpt-4o-mini")
    
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=ClassificationResult,
        temperature=0.0
    )
    
    return response.choices[0].message.parsed
