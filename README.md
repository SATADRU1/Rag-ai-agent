# Rag-ai-agent
a simple production ready retrieval augment generation ai project based on ai/ml topic

## Setup

1. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

2. Run the development server:
   ```bash
   uv run uvicorn main:app
   ```

3. Start the Inngest CLI:
   ```bash
   npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
   ```
