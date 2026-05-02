# Valura AI — Team Lead Project Assignment

This is the implementation of the Valura AI microservice.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Poornachandra-dh/Valura-ai.git
   cd Valura-ai
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   You must provide API keys to run the AI agents. Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   # If you prefer to use Gemini's OpenAI compatibility, set:
   # GEMINI_API_KEY=your_gemini_api_key
   CLASSIFIER_MODEL=gpt-4o-mini
   ```

4. **Running the Server:**
   ```bash
   uvicorn src.main:app --reload
   ```

5. **Running Tests:**
   ```bash
   pytest tests/ -v
   ```
   Tests use a mock LLM setup to run without requiring an API key.

## Architecture and Decisions

- **Safety Guard (`src/safety.py`)**: 
  - Synchronous, fast regex-based evaluation of user intents. Evaluates and completes in less than 10ms without any LLM network calls.
  - Distinct responses are crafted specifically around personal actions (e.g., "tell me to buy") to pass-through educational queries ("what is") safely.

- **Intent Classifier (`src/classifier.py`)**: 
  - Uses OpenAI's Structured Outputs (`response_format=ClassificationResult`) to return exact intents and entities.
  - The fallback mechanism supports testing locally even without an OpenAI key, by optionally falling back to Google Gemini's OpenAI compatibility endpoint.

- **Router (`src/router.py`)**: 
  - Clean separation: routes to the real agent when implemented or returns a graceful stub message with classified entities.

- **Portfolio Health Agent (`src/agents/portfolio_health.py`)**: 
  - Fetches live data via `yfinance` to evaluate portfolio values and compares it to average cost for realistic ROI.
  - LLM receives a synthesized prompt containing numeric calculations to produce accurate, plain-language insights (preventing hallucinations).
  - Designed gracefully to handle `user_004` (Empty Portfolio) by reverting to a BUILD strategy instead of calculating metrics.

- **HTTP Layer (`src/main.py`)**:
  - Exposes `POST /chat` with a strict Server-Sent Events (SSE) streaming output.
  - Implements an asynchronous 15-second timeout for streaming robustness.
  - **Memory Persistence**: Sessions are stored in a simple Python dictionary mapping `session_id` to conversational history to prioritize speed and low setup overhead.

## Stretch Goals Addressed
- **Per-tenant model selection**: Configurable via `CLASSIFIER_MODEL` environment variable.

## Video Presentation

*(Please insert your unlisted YouTube video URL here before submission)*
