from typing import List, Union
import logging
from openai import OpenAI
from core.utils import auto_config
from loguru import logger

openai_client = OpenAI(api_key=auto_config.OPENAI_API_KEY)

def text_to_embedding(texts: Union[str, List[str]]) -> List[List[float]]:
    """
    Transform text or a list of texts into embedding vectors using OpenAI's API.

    Args:
        texts (Union[str, List[str]]): The input text or list of texts to be transformed into embeddings.

    Returns:
        List[List[float]]: A list of embedding vectors, each as a list of floats.

    Raises:
        ValueError: If the API key is not set or if the API request fails.
    """
    if not auto_config.OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY in the configuration.")

    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    try:
        response = openai_client.embeddings.create(
            model=auto_config.OPENAI_TEXT_EMBEDDING_MODEL,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        logger.info(f"Successfully generated embeddings for {len(texts)} texts.")
        return embeddings
    except Exception as e:
        logger.error(f"Error while generating embeddings: {str(e)}")
        raise ValueError(f"Failed to generate embeddings: {str(e)}")
