"""Unit tests for core.database.milvus module."""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from core.database.milvus import MilvusManager, AsyncMilvusManager


class TestMilvusManager:
    """Test cases for MilvusManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = MilvusManager(milvus_host="localhost", milvus_port=19530)

    def test_initialization(self):
        """Test MilvusManager initialization."""
        assert self.manager.milvus_host == "localhost"
        assert self.manager.milvus_port == 19530
        assert self.manager.connection is None

    @patch('core.database.milvus.connections.connect')
    def test_connect(self, mock_connect):
        """Test connecting to Milvus."""
        mock_connection = Mock()
        mock_connect.return_value = mock_connection
        
        self.manager.connect()
        
        assert self.manager.connection is not None
        mock_connect.assert_called_once()

    def test_is_connected_false(self):
        """Test is_connected returns False when not connected."""
        assert self.manager.is_connected() is False

    def test_is_connected_true(self):
        """Test is_connected returns True when connected."""
        self.manager.connection = Mock()
        assert self.manager.is_connected() is True

    @patch('core.database.milvus.connections.disconnect')
    def test_close(self, mock_disconnect):
        """Test closing Milvus connection."""
        self.manager.connection = Mock()
        self.manager.close()
        
        mock_disconnect.assert_called_once_with("default")
        assert self.manager.connection is None

    @patch('core.database.milvus.utility.has_collection')
    @patch('core.database.milvus.Collection')
    @patch('core.database.milvus.CollectionSchema')
    @patch('core.database.milvus.FieldSchema')
    def test_ensure_collection_exists(self, mock_field, mock_schema, mock_collection, mock_has_collection):
        """Test ensure_collection when collection already exists."""
        mock_has_collection.return_value = True
        mock_coll = Mock()
        mock_coll.has_index.return_value = True
        mock_collection.return_value = mock_coll
        
        result = self.manager.ensure_collection("test_collection", has_vector=True)
        
        assert result == mock_coll
        mock_has_collection.assert_called_once_with("test_collection")

    @patch('core.database.milvus.utility.has_collection')
    @patch('core.database.milvus.Collection')
    @patch('core.database.milvus.CollectionSchema')
    @patch('core.database.milvus.FieldSchema')
    def test_ensure_collection_create_new(self, mock_field, mock_schema, mock_collection, mock_has_collection):
        """Test ensure_collection when creating new collection."""
        mock_has_collection.return_value = False
        mock_coll = Mock()
        mock_collection.return_value = mock_coll
        
        self.manager.ensure_index = Mock()
        
        result = self.manager.ensure_collection("test_collection", has_vector=True, vector_size=768)
        
        assert result == mock_coll
        self.manager.ensure_index.assert_called_once()

    @patch('core.database.milvus.Collection')
    def test_insert_points(self, mock_collection):
        """Test inserting points into collection."""
        mock_coll = Mock()
        mock_collection.return_value = mock_coll
        self.manager.ensure_collection = Mock(return_value=mock_coll)
        
        points = [
            {"vector": [0.1, 0.2, 0.3], "metadata": {"text": "test"}},
            {"vector": [0.4, 0.5, 0.6], "metadata": {"text": "test2"}}
        ]
        
        self.manager.insert_points("test_collection", points, has_vector=True)
        
        mock_coll.insert.assert_called_once()
        mock_coll.flush.assert_called_once()

    @patch('core.database.milvus.Collection')
    def test_query_points(self, mock_collection):
        """Test querying points from collection."""
        mock_coll = Mock()
        mock_collection.return_value = mock_coll
        
        mock_hit = Mock()
        mock_hit.id = 1
        mock_hit.distance = 0.5
        mock_hit.entity = {"payload": {"text": "test"}}
        
        mock_coll.search.return_value = [[mock_hit]]
        
        results = self.manager.query_points("test_collection", [0.1, 0.2, 0.3], top=10)
        
        assert len(results) == 1
        assert results[0]["id"] == 1
        mock_coll.load.assert_called_once()

    @patch('core.database.milvus.Collection')
    def test_delete_points(self, mock_collection):
        """Test deleting points from collection."""
        mock_coll = Mock()
        mock_collection.return_value = mock_coll
        
        self.manager.delete_points("test_collection", ["1", "2"])
        
        mock_coll.delete.assert_called_once()

    @patch('core.database.milvus.Collection')
    def test_update_point_payload(self, mock_collection):
        """Test updating point payload."""
        mock_coll = Mock()
        mock_collection.return_value = mock_coll
        
        payload = {"text": "updated"}
        self.manager.update_point_payload("test_collection", "1", payload)
        
        mock_coll.upsert.assert_called_once()


class TestAsyncMilvusManager:
    """Test cases for AsyncMilvusManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AsyncMilvusManager(milvus_host="localhost", milvus_port=19530)

    @pytest.mark.asyncio
    @patch('core.database.milvus.asyncio.get_event_loop')
    @patch('core.database.milvus.connections.connect')
    async def test_connect_async(self, mock_connect, mock_get_loop):
        """Test async connection to Milvus."""
        mock_loop = Mock()
        mock_connection = Mock()
        # run_in_executor returns a coroutine, so we need to make it awaitable
        mock_loop.run_in_executor = Mock(return_value=mock_connection)
        mock_get_loop.return_value = mock_loop
        
        # Since run_in_executor is called with await, we need to make it return a coroutine
        import asyncio
        async def run_executor(*args, **kwargs):
            return mock_connection
        mock_loop.run_in_executor = run_executor
        
        await self.manager.connect()
        
        assert self.manager.connection is not None

    @pytest.mark.asyncio
    @patch('core.database.milvus.asyncio.get_event_loop')
    @patch('core.database.milvus.connections.disconnect')
    async def test_close_async(self, mock_disconnect, mock_get_loop):
        """Test async closing of connection."""
        self.manager.connection = Mock()
        mock_loop = Mock()
        
        # run_in_executor returns a coroutine, so we need to make it awaitable
        import asyncio
        async def run_executor(*args, **kwargs):
            return None
        mock_loop.run_in_executor = run_executor
        mock_get_loop.return_value = mock_loop
        
        await self.manager.close()
        
        # Verify connection was cleared
        assert self.manager.connection is None
