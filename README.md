## Senior Enterprise Python LLM Developer Assessment

### Objective

Build a QnA agent API that answers questions using OpenAI and a local knowledge base.

**Focus**: This assessment emphasizes integration and delivery practices. Demonstrate your ability to build production-ready services with proper containerization and deployment patterns.

### Core Requirements

#### API Functionality

- **Chat-based QnA system** with the following capabilities:
  - **Create, list, and delete chat sessions**
  - **Retrieve message history for a chat**
  - **Post messages and receive AI-generated responses**
  - **Notify clients about chat updates** (choose your approach: polling, webhooks, websockets, SSE, etc.)

#### Agent Behavior

- **Use OpenAI-compatible API** for response generation
- **Agent must use tool/function calling** to query the knowledge base
- **Maintain conversation context** within chat sessions
- **Persist all interactions to storage**

#### Knowledge Base

- **Plain text files** in a designated directory (e.g., `./knowledge`)
- **Each file represents a discrete piece of information**
- **Agent determines when and how to retrieve relevant content**

### Technical Constraints

- **Framework**: FastAPI
- **Storage**: SQLite (embedded)
- **LLM Client**: OpenAI Python SDK (or compatible)
- **Dependency Management**: Poetry or uv
- **Testing**: pytest with async support

### Test Setup

For LLM access, use one of:

- **Free models via OpenRouter** (`https://openrouter.ai`)
- **Local Ollama installation** (`https://ollama.ai`)

Configure via environment variables. **No mock responses.**

### Deliverables

1. **Application Code**
   - Organized project structure
   - Database schema and initialization
   - API implementation with error handling
   - Tool/function definitions for KB access

2. **Tests**
   - Basic test coverage (at least happy path scenarios)
   - Integration tests with real LLM calls acceptable

3. **Docker Setup**
   - Dockerfile with multi-stage build
   - Non-root user execution
   - Optimized image size
   - Example environment configuration

4. **Kubernetes Manifest (Optional)**
   - Basic deployment configuration
   - Service definition
   - Consider production-ready patterns

5. **Documentation**
   - Setup and run instructions
   - API usage examples
   - Design decisions

### Production Considerations

While not required for implementation, consider and document (TODO items are fine):

- **Sensitive configuration in K8S environments**: how you would handle it
- **Service lifecycle management**: what endpoints would help orchestration systems manage service lifecycle?
- **Observability**: how operations might gain visibility into service health and performance
- **Persistent data**: how to handle persistent data in containerized deployments
- **TLS termination and ports**: how to handle TLS termination and port configuration

**Bonus**: If time permits, note potential performance improvements.

### Evaluation Criteria

- **API design and implementation quality**
- **Proper tool/function calling usage**
- **Docker and deployment best practices**
- **Code organization and async patterns**
- **Production awareness**

**Submission**: Provide a git repository or archive with all deliverables and this README.


