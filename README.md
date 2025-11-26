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

### How to run (FastAPI)

Start the API server:

```bash
poetry run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Then open the interactive Swagger UI:

- API docs: `http://localhost:8000/docs`

From there you can:

- **List messages for a chat**:  
  - Endpoint: `POST /chat/get_messages`  
  - Example body:
    ```json
    {
      "chat_id": 1
    }
    ```

- **Ask a question in a chat (with history + routing + KB)**:  
  - Endpoint: `POST /chat/answer`  
  - Example body:
    ```json
    {
      "chat_id": 1,
      "question": "What do koalas eat?"
    }
    ```

- **Delete a chat**:  
  - Endpoint: `DELETE /chat/delete?chat_id=1` (e.g. `DELETE /chat/delete?chat_id=1`)

