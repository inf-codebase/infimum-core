"""
EmbeddingProvider — participates in the shared BaseProvider abstraction.

This class bridges the embedding-specific embed()/get_embedding_dimension()
surface with the lifecycle (load_model/unload_model), observer events and
ProviderFactory/ProviderRegistry mechanism used by LLM, VLM and speech.

Usage
-----
Concrete providers should subclass EmbeddingProvider and implement:
  - load_model()   : initialise the API client / local model
  - unload_model() : release resources
  - embed()        : generate embeddings
  - get_embedding_dimension() : return vector width for a model name
"""

from abc import abstractmethod
from typing import List, Optional, Union

# Use direct submodule imports (not via core.ai which has a broken __init__)
from core.ai.base.providers.base import BaseProvider, ModelHandle
from core.ai.base.providers.config import ModelConfig
from core.ai.base.observers.events import Event, EventType


class EmbeddingProvider(BaseProvider):
    """Abstract base for embedding providers.

    Extends :class:`~core.ai.base.providers.base.BaseProvider` so that
    embedding providers participate in the same
    provider/registry/factory/observer chain as LLM, VLM and speech.

    Concrete implementations must override:

    * :meth:`load_model` — initialise the client / model and return a
      :class:`~core.ai.base.providers.base.ModelHandle`.
    * :meth:`unload_model` — release resources held by a handle.
    * :meth:`embed` — the core embedding routine.
    * :meth:`get_embedding_dimension` — return vector width.

    Example
    -------
    See :class:`~core.ai.embeddings.providers.openai.OpenAIEmbeddingProvider`
    for a complete reference implementation.
    """

    # ------------------------------------------------------------------ #
    #  BaseProvider abstract hooks                                         #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_model(self, config: ModelConfig) -> ModelHandle:
        """Load / initialise the underlying embedding client or model.

        Args:
            config: ModelConfig with ``model_type="embedding"``.

        Returns:
            ModelHandle wrapping the initialised client, metadata, and config.
        """

    @abstractmethod
    def unload_model(self, handle: ModelHandle) -> None:
        """Release resources held by *handle*.

        Args:
            handle: The handle returned by :meth:`load_model`.
        """

    # ------------------------------------------------------------------ #
    #  Embedding-specific abstract methods                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def embed(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embedding vectors for one or more texts.

        Args:
            texts: A single string **or** a list of strings to embed.
            model: Optional model name; falls back to the provider's
                default if not supplied.

        Returns:
            A list of float vectors — one per input text.

        Raises:
            ValueError: On API / inference errors.
        """

    @abstractmethod
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Return the dimensionality of the embedding vectors.

        Args:
            model: Optional model name; falls back to the provider's
                default if not supplied.

        Returns:
            Integer dimension (e.g. 1536 for *text-embedding-3-small*).
        """

    # ------------------------------------------------------------------ #
    #  Observer helpers                                                    #
    # ------------------------------------------------------------------ #

    def _notify_embed_started(self, n_texts: int, model: str) -> None:
        """Emit an INFERENCE_STARTED event before an embed call."""
        self.notify(Event(
            type=EventType.INFERENCE_STARTED,
            data={"n_texts": n_texts, "model": model},
            source=self.__class__.__name__,
        ))

    def _notify_embed_completed(self, n_texts: int, model: str) -> None:
        """Emit an INFERENCE_COMPLETED event after a successful embed."""
        self.notify(Event(
            type=EventType.INFERENCE_COMPLETED,
            data={"n_texts": n_texts, "model": model},
            source=self.__class__.__name__,
        ))

    def _notify_embed_failed(self, error: str, model: str) -> None:
        """Emit an INFERENCE_FAILED event when embed raises."""
        self.notify(Event(
            type=EventType.INFERENCE_FAILED,
            data={"error": error, "model": model},
            source=self.__class__.__name__,
        ))
