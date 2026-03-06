"""
OpenAI Embedding Provider — integrated with ProviderFactory/ProviderRegistry.

Replaces the old standalone implementation while keeping the same
embed() / get_embedding_dimension() public API for backward compatibility.
"""

from typing import List, Optional, Union

from loguru import logger
from openai import OpenAI

# Direct submodule imports — avoid triggering core.ai.__init__ (which has a
# pre-existing broken `from .llm import Agent...` that requires optional deps)
from core.ai.embeddings.base import EmbeddingProvider
from core.ai.base.providers.base import ModelHandle
from core.ai.base.providers.config import ModelConfig
from core.ai.base.providers.registry import ProviderRegistry, ProviderMetadata
from core.ai.base.providers.factory import ProviderFactory


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider — integrated with the shared provider chain.

    Supports the same embedding models as before (text-embedding-3-small,
    text-embedding-3-large, text-embedding-ada-002) and gains:

    * Lifecycle events via the Observer pattern (INFERENCE_STARTED, etc.)
    * Registration in ProviderFactory
    * Creation via ``ProviderFactory.create("embedding", "openai", config)``

    Backward-compatible direct-use API
    ------------------------------------
    The provider can still be instantiated directly::

        provider = OpenAIEmbeddingProvider(api_key="sk-...")
        vectors = provider.embed("Hello world")
        dim     = provider.get_embedding_dimension()

    Factory path (preferred for new code)::

        from core.ai.base.providers.factory import ProviderFactory
        from core.ai.base.providers.config import ModelConfig

        config   = ModelConfig(model_type="embedding", provider="openai",
                               model_name="text-embedding-3-small", model_path="",
                               extra_params={"api_key": "sk-..."})
        provider = ProviderFactory.create("embedding", "openai", config)
        handle   = provider.get_model(config)   # fires observer events
        vectors  = provider.embed("Hello world")
    """

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "text-embedding-002": 1536,
    }

    # ------------------------------------------------------------------ #
    #  Constructor                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "text-embedding-3-small",
        config: Optional[ModelConfig] = None,
    ):
        """Initialise provider.

        Args:
            api_key: OpenAI API key. Required unless *config* supplies it
                via ``extra_params["api_key"]``.
            default_model: Model name used when embed() is called without
                an explicit *model* argument.
            config: Optional ModelConfig. When supplied the ``api_key`` and
                model name are derived from the config if not explicitly passed.
        """
        # Support config-driven construction (ProviderFactory path)
        if config is not None and not api_key:
            api_key = config.extra_params.get("api_key", "")
        if config is not None and config.model_name:
            default_model = config.model_name

        if not api_key:
            raise ValueError("OpenAI API key is required")

        # Call BaseProvider.__init__ which initialises Observable
        super().__init__(config)

        self._api_key = api_key
        self.default_model = default_model
        # Initialise client eagerly (cheap — no network call yet)
        self.client = OpenAI(api_key=api_key)
        logger.debug(f"Initialised OpenAIEmbeddingProvider with model: {default_model}")

    # ------------------------------------------------------------------ #
    #  BaseProvider lifecycle implementation                               #
    # ------------------------------------------------------------------ #

    def load_model(self, config: ModelConfig) -> ModelHandle:
        """Initialise the OpenAI client and wrap it in a ModelHandle.

        The base-class ``get_model()`` template fires
        MODEL_LOADING_STARTED / MODEL_LOADING_COMPLETED observer events
        around this call automatically.
        """
        api_key = config.extra_params.get("api_key", self._api_key)
        model_name = config.model_name or self.default_model

        client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI embedding client loaded (model={model_name})")
        return ModelHandle(
            model=client,
            config=config,
            metadata={
                "model_name": model_name,
                "provider": "openai",
                "type": "embedding",
                "dimensions": self.MODEL_DIMENSIONS.get(model_name, 1536),
            },
        )

    def unload_model(self, handle: ModelHandle) -> None:
        """Release resources (no-op for stateless HTTP clients)."""
        logger.debug("OpenAIEmbeddingProvider.unload_model called (no-op)")

    # ------------------------------------------------------------------ #
    #  Embedding API                                                       #
    # ------------------------------------------------------------------ #

    def embed(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI Embeddings API.

        Fires observer events (INFERENCE_STARTED / COMPLETED / FAILED).
        """
        if isinstance(texts, str):
            texts = [texts]

        model = model or self.default_model

        self._notify_embed_started(len(texts), model)
        try:
            response = self.client.embeddings.create(model=model, input=texts)
            embeddings = [item.embedding for item in response.data]
            logger.info(
                f"Generated {len(embeddings)} embedding(s) via {model}"
            )
            self._notify_embed_completed(len(texts), model)
            return embeddings
        except Exception as exc:
            logger.error(f"Embedding error: {exc}")
            self._notify_embed_failed(str(exc), model)
            raise ValueError(f"Failed to generate embeddings: {exc}") from exc

    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Return vector dimension for *model* (default: ``default_model``)."""
        model = model or self.default_model
        dim = self.MODEL_DIMENSIONS.get(model, 1536)
        logger.debug(f"Embedding dimension for {model}: {dim}")
        return dim


# --------------------------------------------------------------------------- #
#  Auto-registration in ProviderFactory and ProviderRegistry                  #
# --------------------------------------------------------------------------- #

ProviderFactory.register(
    model_type="embedding",
    provider_name="openai",
    provider_class=OpenAIEmbeddingProvider,
)

ProviderRegistry.register(
    "embedding-openai",
    ProviderMetadata(
        model_type="embedding",
        provider_name="openai",
        capabilities={"embed", "similarity"},
        description="OpenAI text embedding provider (text-embedding-3-small, etc.)",
        version="1.0.0",
        requirements=["openai>=2.15.0"],
    ),
)
