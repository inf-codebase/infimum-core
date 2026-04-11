from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
from typing import List, Dict, Any, Optional
import asyncio
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class MilvusClient:
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.user = os.getenv("MILVUS_USER", "")
        self.password = os.getenv("MILVUS_PASSWORD", "")
        self.connect()

    def connect(self):
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            logger.info("Connected to Milvus server")
            self.ensure_index("vlm_analysis_embeddings")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus server: {str(e)}")
            raise

    def ensure_index(self, collection_name: str):
        try:
            collection = Collection(collection_name)
            logger.info(f"Collection {collection_name} exists. Schema: {collection.schema}")
        except Exception:
            logger.info(f"Collection {collection_name} does not exist. Creating it now.")
            collection = self.create_collection(collection_name)

        expected_fields = {"id", "embedding", "text"}
        actual_fields = set(field.name for field in collection.schema.fields)
        if not expected_fields.issubset(actual_fields):
            logger.warning(f"Collection {collection_name} does not have the correct schema. Expected fields: {expected_fields}, Actual fields: {actual_fields}")
            return

        if not collection.has_index():
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"Created index for collection: {collection_name}")
        else:
            logger.info(f"Index already exists for collection: {collection_name}")

        collection.load()
        logger.info(f"Collection {collection_name} loaded. Number of entities: {collection.num_entities}")

    def create_collection(self, collection_name: str) -> Collection:
        # Deprecated synchronous helper. Prefer the async variant below which
        # accepts a dynamic `fields` argument. We keep this for backward
        # compatibility by delegating to the async version.
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.create_collection_async(collection_name))

    async def create_collection_async(self, collection_name: str, fields: Optional[List[Dict[str, Any]]] = None) -> Collection:
        """
        Create a Milvus collection. If `fields` is None, a default schema is used.

        `fields` is expected to be a list of dicts like:
        [{"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": True, ...}, ...]
        """
        try:
            if fields is None:
                fields = [
                    {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": True},
                    {"name": "embedding", "type": DataType.FLOAT_VECTOR, "dim": 768},
                    {"name": "text", "type": DataType.VARCHAR, "max_length": 65535},
                ]

            field_schemas: List[FieldSchema] = []
            for f in fields:
                name = f.get("name")
                dtype = f.get("type")
                is_primary = f.get("is_primary", False)
                auto_id = f.get("auto_id", False)
                max_length = f.get("max_length")
                dim = f.get("dim")

                # FieldSchema supports `dim` for vector fields and `max_length` for varchar
                field_schema = FieldSchema(
                    name=name,
                    dtype=dtype,
                    is_primary=is_primary,
                    auto_id=auto_id,
                    max_length=max_length,
                    dim=dim,
                )
                field_schemas.append(field_schema)

            schema = CollectionSchema(field_schemas, description=f"Collection for {collection_name}")
            collection = Collection(collection_name, schema)
            logger.info(f"Created collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {str(e)}")
            raise

    # Backwards compatible async alias used by calling code
    async def create_collection(self, collection_name: str, fields: Optional[List[Dict[str, Any]]] = None) -> Collection:
        return await self.create_collection_async(collection_name, fields)

    async def search(self, collection_name: str, query_vectors: List[List[float]], top_k: int, search_params: Dict[str, Any], expr: Optional[str] = None) -> List[List[Dict[str, Any]]]:
        try:
            collection = Collection(collection_name)
            collection.load()
            logger.info(f"Collection loaded: {collection_name}")
            logger.debug(f"Collection schema: {collection.schema}")
            logger.debug(f"Collection stats: {collection.num_entities}")

            # Determine the name of the vector field and the payload/text field
            vector_field = None
            payload_field = None
            for field in collection.schema.fields:
                # FieldSchema.dtype can be compared against DataType
                if field.dtype == DataType.FLOAT_VECTOR:
                    vector_field = field.name
                if field.dtype == DataType.JSON or field.dtype == DataType.VARCHAR:
                    # prefer a JSON payload if present
                    if payload_field is None or field.dtype == DataType.JSON:
                        payload_field = field.name

            if vector_field is None:
                # fallback to common names
                vector_field = "embedding"

            if payload_field is None:
                payload_field = "text"

            # Execute the search using detected field names
            # Pass expression (filter) to search if provided
            results = collection.search(
                data=query_vectors,
                anns_field=vector_field,
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=[payload_field]
            )

            # Normalize results to the format expected by RAGQueryEngine._format_milvus_results
            normalized = []
            for result_list in results:
                hits = []
                for hit in result_list:
                    # `hit.entity` is typically a mapping of output_fields to values
                    payload = {}
                    try:
                        if hasattr(hit, "entity") and hit.entity is not None:
                            # In some pymilvus versions hit.entity is a mapping
                            payload = hit.entity.get(payload_field) if isinstance(hit.entity, dict) else hit.entity
                        else:
                            # Some versions expose 'fields' or similar; try attribute access
                            payload = {}
                    except Exception:
                        payload = {}

                    hits.append({
                        "id": getattr(hit, "id", None),
                        "distance": getattr(hit, "distance", None),
                        "payload": payload,
                    })
                normalized.append(hits)

            logger.debug(f"Normalized search results: {normalized}")
            return normalized
        except Exception as e:
            logger.error(f"Error during Milvus search: {str(e)}")
            raise

    async def insert(self, collection_name: str, entities: List[Dict[str, Any]]):
        try:
            collection = Collection(collection_name)
            insert_result = collection.insert(entities)
            logger.info(f"Inserted {len(entities)} entities into collection {collection_name}")
            logger.debug(f"Insert result: {insert_result}")
        except Exception as e:
            logger.error(f"Error during Milvus insert: {str(e)}")
            raise

    async def has_collection(self, collection_name: str) -> bool:
        """
        Async wrapper around pymilvus.utility.has_collection
        """
        try:
            from pymilvus import utility
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, utility.has_collection, collection_name)
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            raise

    async def create_index(self, collection_name: str, field_name: str, index_params: Dict[str, Any]):
        try:
            collection = Collection(collection_name)
            collection.create_index(field_name=field_name, index_params=index_params)
            logger.info(f"Created index on {field_name} for collection {collection_name}")
        except Exception as e:
            logger.error(f"Error creating index on {collection_name}.{field_name}: {str(e)}")
            raise

    async def load_collection(self, collection_name: str):
        try:
            collection = Collection(collection_name)
            collection.load()
            logger.info(f"Loaded collection {collection_name}")
        except Exception as e:
            logger.error(f"Error loading collection {collection_name}: {str(e)}")
            raise

    async def query_all(self, collection_name: str) -> List[List[Dict[str, Any]]]:
        """
        Return all entities in a collection as a single result set, normalized to the
        same structure used by `search` (List[List[Dict]]), where each inner dict
        contains `id`, `distance` (None) and `payload`.
        """
        try:
            collection = Collection(collection_name)
            collection.load()

            # Determine payload field and primary key
            vector_field = None
            payload_field = None
            primary_field = None
            for field in collection.schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    vector_field = field.name
                if field.dtype == DataType.JSON or field.dtype == DataType.VARCHAR:
                    if payload_field is None or field.dtype == DataType.JSON:
                        payload_field = field.name
                if getattr(field, 'is_primary', False):
                    primary_field = field.name

            if payload_field is None:
                payload_field = "text"
            if primary_field is None:
                primary_field = "id"

            # Query all entities. Some Milvus versions require an explicit `limit`
            # when using an empty expression; use collection.num_entities as the limit.
            total = getattr(collection, "num_entities", None)
            if total is None:
                # As a conservative fallback, request a reasonably large limit
                limit = 10000
            else:
                # Ensure limit is an int and not excessively large for safety
                try:
                    limit = int(total)
                except Exception:
                    limit = 10000

            # If collection is empty, return empty normalized list
            if limit == 0:
                return [[]]

            # Try calling query with an empty expr and a limit. If the API differs,
            # fall back to other combinations.
            query_results = None
            tried = []
            try:
                query_results = collection.query(expr="", output_fields=[primary_field, payload_field], limit=limit)
                tried.append('expr="", limit')
            except TypeError:
                # Some pymilvus signatures may not accept `limit` or expect expr=None
                try:
                    query_results = collection.query(expr=None, output_fields=[primary_field, payload_field], limit=limit)
                    tried.append('expr=None, limit')
                except TypeError:
                    try:
                        # try without limit but with a non-empty expr that matches all rows
                        query_results = collection.query(expr="", output_fields=[primary_field, payload_field])
                        tried.append('expr="" no limit')
                    except Exception as inner:
                        logger.error(f"All attempts to query all entities failed (attempts={tried}): {str(inner)}")
                        raise
            except Exception as e:
                # Other Milvus exceptions should bubble up for the caller to handle
                logger.error(f"Error querying collection with expr/limit attempt {tried}: {str(e)}")
                raise

            normalized = []
            for row in query_results:
                # row is typically a dict mapping field->value
                pid = row.get(primary_field) if isinstance(row, dict) else None
                payload = row.get(payload_field) if isinstance(row, dict) else row
                normalized.append({
                    "id": pid,
                    "distance": None,
                    "payload": payload,
                })

            return [normalized]
        except Exception as e:
            logger.error(f"Error querying all entities from {collection_name}: {str(e)}")
            raise

    def close(self):
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus server")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus server: {str(e)}")
            raise

# Usage:
# milvus_client = MilvusClient()
# try:
#     results = await milvus_client.search(...)
# finally:
#     milvus_client.close()
