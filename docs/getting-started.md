# Getting started

## Install

**From PyPI:**

```bash
pip install infimum-core
```

**From GitHub:**

```bash
pip install "git+https://github.com/inf-codebase/infimum.git#subdirectory=core"
```

For a specific branch: add `@branch` before `#subdirectory`, e.g. `...infimum.git@main#subdirectory=core`.

Then `import core` in Python. Local editable: `pip install -e ./core` from repo root.

**Optional extras:** `pip install infimum-core[all-db,security,llm]` (see table below).

## Optional extras

| Extra      | Description                                      |
| ---------- | ------------------------------------------------ |
| `mongo`    | MongoDB support (pymongo, motor)                 |
| `milvus`   | Milvus vector database                          |
| `qdrant`   | Qdrant vector database                          |
| `all-db`   | All database backends (mongo, milvus, qdrant)   |
| `security` | JWT, password hashing (bcrypt, python-jose)      |
| `llm`      | LLM/agent stack (LangChain, LangGraph)          |

## Build a wheel

From the repository root:

```bash
pip install build
python -m build core/
```

Wheels and sdist are written to `core/dist/`. Install the wheel with:

```bash
pip install core/dist/infimum_core-*.whl
```

## Usage

```python
import core
from core.base.entity import BaseEntity, Document
from core.database import DatabaseManager, VectorIndexConfig
from core.engine import context, startup
from core.utils import string_utils, auto_config
```

## Subpackages

- **core.base** — Entities, registry, repository base
- **core.engine** — Context, decorators, startup, security (optional)
- **core.database** — Database managers and interfaces (Postgres, Milvus, Qdrant, Mongo)
- **core.ai** — LLM, VLM, speech, embeddings, data loaders, preprocessing
- **core.utils** — Config, embedding helpers, crawling, Redis, validation, etc.

See the [API Reference](api/base.md) for detailed API docs.
