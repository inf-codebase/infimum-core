"""Unit tests for core.database.qdrant module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from core.database.qdrant import QdrantManager, AsyncQdrantManager


class TestQdrantManager:
    """Test cases for QdrantManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = QdrantManager(qdrant_url="http://localhost:6333")

    def test_initialization(self):
        """Test QdrantManager initialization."""
        assert self.manager.qdrant_url == "http://localhost:6333"
        assert self.manager.client is None
        assert self.manager.use_memory is False

    def test_initialization_memory(self):
        """Test QdrantManager initialization with memory mode."""
        manager = QdrantManager(use_memory=True)
        assert manager.use_memory is True

    @patch('core.database.qdrant.QdrantClient')
    def test_connect(self, mock_qdrant_client):
        """Test connecting to Qdrant."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        
        self.manager.connect()
        
        assert self.manager.client == mock_client
        mock_qdrant_client.assert_called_once()

    @patch('core.database.qdrant.QdrantClient')
    def test_connect_memory(self, mock_qdrant_client):
        """Test connecting to Qdrant in memory mode."""
        manager = QdrantManager(use_memory=True)
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        
        manager.connect()
        
        mock_qdrant_client.assert_called_once_with(":memory:")

    def test_close(self):
        """Test closing Qdrant connection."""
        mock_client = Mock()
        self.manager.client = mock_client
        
        self.manager.close()
        
        mock_client.close.assert_called_once()
        assert self.manager.client is None

    @patch('core.database.qdrant.models')
    def test_ensure_collection_exists(self, mock_models):
        """Test ensure_collection when collection exists."""
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        self.manager.client = mock_client
        
        self.manager.ensure_collection("test_collection", has_vector=True, vector_size=768)
        
        mock_client.get_collection.assert_called_once_with("test_collection")

    @patch('core.database.qdrant.models')
    def test_ensure_collection_create_new(self, mock_models):
        """Test ensure_collection when creating new collection."""
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Not found")
        self.manager.client = mock_client
        
        self.manager.ensure_collection("test_collection", has_vector=True, vector_size=768)
        
        mock_client.create_collection.assert_called_once()

    def test_insert_points(self):
        """Test inserting points into collection."""
        mock_client = Mock()
        self.manager.client = mock_client
        self.manager.ensure_collection = Mock()
        
        points = [
            {"id": "1", "vector": [0.1, 0.2, 0.3], "payload": {"text": "test"}},
            {"id": "2", "vector": [0.4, 0.5, 0.6], "payload": {"text": "test2"}}
        ]
        
        self.manager.insert_points("test_collection", points, has_vector=True)
        
        mock_client.upsert.assert_called_once()
        self.manager.ensure_collection.assert_called_once()

    def test_query_points(self):
        """Test querying points from collection."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.dict.return_value = {"id": "1", "distance": 0.5}
        mock_client.search.return_value = [mock_result]
        self.manager.client = mock_client
        
        results = self.manager.query_points("test_collection", [0.1, 0.2, 0.3], top=10)
        
        assert len(results) == 1
        mock_client.search.assert_called_once()

    def test_delete_points(self):
        """Test deleting points from collection."""
        mock_client = Mock()
        self.manager.client = mock_client
        
        self.manager.delete_points("test_collection", ["1", "2"])
        
        mock_client.delete.assert_called_once()

    def test_update_point_payload(self):
        """Test updating point payload."""
        mock_client = Mock()
        self.manager.client = mock_client
        
        payload = {"text": "updated"}
        self.manager.update_point_payload("test_collection", "1", payload)
        
        mock_client.set_payload.assert_called_once()


class TestAsyncQdrantManager:
    """Test cases for AsyncQdrantManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AsyncQdrantManager(qdrant_url="http://localhost:6333")

    @patch('core.database.qdrant.AsyncQdrantClient')
    def test_connect(self, mock_async_client):
        """Test async connection to Qdrant."""
        mock_client = Mock()
        mock_async_client.return_value = mock_client
        
        self.manager.connect()
        
        assert self.manager.client == mock_client

    @pytest.mark.asyncio
    async def test_ensure_collection_async(self):
        """Test async ensure_collection."""
        mock_client = Mock()
        mock_client.get_collection = Mock(side_effect=Exception("Not found"))
        self.manager.client = mock_client
        
        await self.manager.ensure_collection("test_collection", has_vector=True)
        
        mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_points_async(self):
        """Test async insert_points."""
        mock_client = Mock()
        self.manager.client = mock_client
        self.manager.ensure_collection = Mock()
        
        points = [{"id": "1", "vector": [0.1, 0.2], "payload": {}}]
        
        await self.manager.insert_points("test_collection", points)
        
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_points_async(self):
        """Test async query_points."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.dict.return_value = {"id": "1"}
        mock_client.search = Mock(return_value=[mock_result])
        self.manager.client = mock_client
        
        results = await self.manager.query_points("test_collection", [0.1, 0.2])
        
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_close_async(self):
        """Test async close."""
        mock_client = Mock()
        self.manager.client = mock_client
        
        await self.manager.close()
        
        mock_client.close.assert_called_once()
