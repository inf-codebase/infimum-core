# infimum-core

Infimum core library: base entities, engine (DI/context/startup), database layer (PostgreSQL, Milvus, Qdrant, MongoDB), AI (LLM, VLM, speech, embeddings), and utilities.

**Documentation:** From repo root run `pip install -e ./core && pip install -r docs/requirements.txt && mkdocs serve`, then open http://127.0.0.1:8000. Or run `mkdocs build` to output the static site to `site/`.

## Install

**From PyPI:**

```bash
pip install infimum-core
```

**From GitHub (default branch):**

```bash
pip install "git+https://github.com/inf-codebase/infimum.git#subdirectory=infimum"
```

**From a GitHub release:** use the release tag (e.g. `v1.2.20`) with `@tag` before `#subdirectory`:

```bash
pip install "git+https://github.com/inf-codebase/infimum.git@v1.2.20#subdirectory=infimum"
```

**With optional extras:**

```bash
# All database backends
pip install "infimum-core[all-db]"

# LLM/Agent stack
pip install "infimum-core[llm]"

# Security features
pip install "infimum-core[security]"

# All extras
pip install "infimum-core[all-db,security,llm]"
```

Local development: `pip install -e .` from the infimum directory. With [uv](https://docs.astral.sh/uv/): `uv sync` from the infimum directory.



## Optional extras

| Extra    | Description                                      |
|----------|--------------------------------------------------|
| `mongo`  | MongoDB support (pymongo, motor)                 |
| `milvus` | Milvus vector database                           |
| `qdrant` | Qdrant vector database                           |
| `all-db` | All database backends (mongo, milvus, qdrant)    |
| `security` | JWT, password hashing (bcrypt, python-jose)    |
| `llm`    | LLM/agent stack (LangChain, LangGraph)           |

## Usage

### Basic Imports

```python
from infimum import Engine
from infimum.base.entity import BaseEntity, Document
from infimum.database import DatabaseManager, VectorIndexConfig
from infimum.engine import context, startup
from infimum.utils import string_utils, auto_config
```

### Example 1: Define Entities

```python
from infimum.base.entity import BaseEntity

class User(BaseEntity):
    """User entity with base fields."""
    name: str
    email: str
    age: int = None

# Create instance
user = User(name="John Doe", email="john@example.com", age=30)
print(user.model_dump())
```

### Example 2: Initialize Engine with Dependency Injection

```python
from infimum.engine import Engine

# Create engine instance
engine = Engine()

# Register services
engine.register("database", DatabaseManager(...))
engine.register("embedder", EmbeddingProvider(...))

# Retrieve services
db = engine.get("database")
```

### Example 3: Database Operations

```python
from infimum.database import DatabaseManager, VectorIndexConfig

# Create database manager
db = DatabaseManager(
    db_type="postgres",
    connection_string="postgresql://user:password@localhost/dbname"
)

# Create a table/collection
db.create_collection("users", User)

# Insert document
user = User(name="Alice", email="alice@example.com", age=28)
db.insert("users", user)

# Query
results = db.query("users", {"name": "Alice"})
```

### Example 4: Vector Database (Embeddings)

```python
from infimum.database import VectorIndexConfig
from infimum.ai.embeddings import OpenAIEmbeddingProvider

# Create embeddings
embedder = OpenAIEmbeddingProvider(api_key="sk-...")

# Create vector index
vector_config = VectorIndexConfig(
    collection_name="documents",
    dimension=1536,
    metric_type="cosine"
)

# Embed and search
text = "What is artificial intelligence?"
embedding = embedder.embed(text)

# Search similar vectors
results = db.search_vectors("documents", embedding, top_k=5)
```

### Example 5: LLM Integration

```python
from infimum.ai.llm import LLMProvider
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", api_key="sk-...")

# Simple generation
response = llm.invoke("What is the capital of France?")
print(response.content)

# With context from database
context = db.query("documents", {"topic": "France"})
prompt = f"Based on {context}, answer: What is the capital of France?"
response = llm.invoke(prompt)
```

### Example 6: Speech Processing

```python
from infimum.ai.speech.providers import MedASRProvider

# Initialize speech provider
speech_provider = MedASRProvider(api_key="...")

# Transcribe audio
audio_file = "speech.wav"
transcript = speech_provider.transcribe(audio_file)
print(f"Transcription: {transcript}")
```

### Example 7: Auto Configuration

```python
from infimum.utils.auto_config import AutoConfig

# Auto-load configuration from environment or config file
config = AutoConfig.from_env()

# Access config values
db_url = config.get("DATABASE_URL")
api_key = config.get("OPENAI_API_KEY")
```

### Example 8: Context Management

```python
from infimum.engine import context

# Set context values (useful for multi-tenant apps)
context.set("user_id", "user_123")
context.set("tenant_id", "tenant_456")

# Retrieve context values
user_id = context.get("user_id")
```

### Example 9: Complete Application

```python
from infimum import Engine
from infimum.base.entity import BaseEntity
from infimum.database import DatabaseManager
from infimum.ai.llm import LLMProvider
from langchain_openai import ChatOpenAI

# Define entity
class Article(BaseEntity):
    title: str
    content: str
    author: str

# Initialize components
engine = Engine()
db = DatabaseManager(db_type="postgres", connection_string="...")
llm = ChatOpenAI(model="gpt-4", api_key="...")

# Register with engine
engine.register("database", db)
engine.register("llm", llm)

# Use throughout app
db_service = engine.get("database")
db_service.insert("articles", Article(title="AI Guide", content="...", author="John"))

llm_service = engine.get("llm")
summary = llm_service.invoke("Summarize this article: ...")
```

## Subpackages

- **infimum.base** — Base entities, registry, repository interfaces
- **infimum.engine** — Dependency injection, context management, startup hooks, security (optional)
- **infimum.database** — Database managers and interfaces (PostgreSQL, Milvus, Qdrant, MongoDB)
- **infimum.ai** — LLM, VLM, speech, embeddings, data loaders, preprocessing
- **infimum.utils** — Configuration, validation, helpers, Redis client, etc.

## Configuration

Set environment variables for database connections:

```bash
export DATABASE_URL="postgresql://user:password@localhost/dbname"
export OPENAI_API_KEY="sk-..."
export MILVUS_URL="http://localhost:19530"
```

Then load with auto_config:

```python
from infimum.utils.auto_config import AutoConfig

config = AutoConfig.from_env()
db_url = config.get("DATABASE_URL")
```
