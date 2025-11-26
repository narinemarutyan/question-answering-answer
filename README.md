## Q&A agent API 

### Objective
Agent that answers questions using OpenAI and a local knowledge base.

### What this project does

Q&A agent using:
- **LangChain** + **`langchain-openai`**
- **OpenAI GPT-4.1** (or compatible model via API key)
- Optional local knowledge in the `knowledge` directory

### Setup

- **Python**: 3.11+  
- **Poetry** installed  
- Install deps:
  ```bash
  poetry install
  ```
- Set `OPENAI_API_KEY` (via `.env` or environment variable)

### How to run

```bash
poetry run python -m src.main
```

This starts a CLI chat loop; type questions at the `You:` prompt and press Enter.

