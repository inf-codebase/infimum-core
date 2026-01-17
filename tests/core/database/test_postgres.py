"""Unit tests for core.database.postgres module."""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from core.database.postgres import (
    DatabaseManager,
    SQLManager,
    PostgresDatabaseManagerImpl,
    DatabaseFactory,
    SyncMongoManager,
    AsyncMongoManager,
    MongoManagerBase
)
from core.base.entity import Document


class TestDatabaseManager:
    """Test cases for DatabaseManager abstract class."""

    def test_is_abstract(self):
        """Test that DatabaseManager is abstract."""
        with pytest.raises(TypeError):
            DatabaseManager()


class TestSQLManager:
    """Test cases for SQLManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SQLManager(engine_info="sqlite:///:memory:")

    @patch('core.database.postgres.create_engine')
    @patch('core.database.postgres.sessionmaker')
    def test_connect(self, mock_sessionmaker, mock_create_engine):
        """Test connecting to database."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_session = Mock()
        mock_sessionmaker.return_value = mock_session
        
        self.manager.connect()
        
        assert self.manager.engine is not None
        assert self.manager.session is not None

    def test_is_connected_false(self):
        """Test is_connected returns False when not connected."""
        assert self.manager.is_connected() is False

    def test_is_connected_true(self):
        """Test is_connected returns True when connected."""
        self.manager.engine = Mock()
        self.manager.session = Mock()
        assert self.manager.is_connected() is True

    def test_close(self):
        """Test closing database connection."""
        self.manager.session = Mock()
        self.manager.engine = Mock()
        
        self.manager.close()
        
        self.manager.session.close.assert_called_once()
        self.manager.engine.dispose.assert_called_once()

    @patch('core.database.postgres.create_engine')
    @patch('core.database.postgres.sessionmaker')
    def test_get_session_context_manager(self, mock_sessionmaker, mock_create_engine):
        """Test get_session as context manager."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_session = Mock()
        # sessionmaker returns a class, so we need to make it return a callable that returns the mock session
        mock_sessionmaker.return_value = lambda: mock_session
        
        self.manager.connect()
        
        with self.manager.get_session() as session:
            assert session is not None
        
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @patch('core.database.postgres.create_engine')
    @patch('core.database.postgres.sessionmaker')
    def test_get_session_rollback_on_error(self, mock_sessionmaker, mock_create_engine):
        """Test get_session rolls back on error."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_session = Mock()
        # sessionmaker returns a class, so we need to make it return a callable that returns the mock session
        mock_sessionmaker.return_value = lambda: mock_session
        
        self.manager.connect()
        
        with pytest.raises(ValueError):
            with self.manager.get_session() as session:
                raise ValueError("Test error")
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()

    def test_insert_or_update_new_record(self):
        """Test insert_or_update with new record."""
        self.manager.session = Mock()
        self.manager.session.query.return_value.filter_by.return_value.count.return_value = 0
        
        class TestModel:
            def __init__(self):
                self.id = None
                self.name = "Test"
        
        model = TestModel()
        result = self.manager.insert_or_update(model, auto_commit=True)
        
        assert result == model
        self.manager.session.add.assert_called_once_with(model)
        self.manager.session.commit.assert_called_once()

    def test_insert_or_update_existing_record(self):
        """Test insert_or_update with existing record."""
        self.manager.session = Mock()
        
        class TestModel:
            def __init__(self):
                self.id = 1
                self.name = "Test"
        
        existing_model = TestModel()
        existing_model.name = "Old"
        
        query_mock = Mock()
        query_mock.count.return_value = 1
        query_mock.first.return_value = existing_model
        self.manager.session.query.return_value.filter_by.return_value = query_mock
        
        new_model = TestModel()
        result = self.manager.insert_or_update(
            new_model,
            auto_commit=True,
            update_if_true_conditions={"id": 1}
        )
        
        assert result == existing_model
        assert existing_model.name == "Test"
        self.manager.session.commit.assert_called_once()

    def test_query_or_create_new_existing(self):
        """Test query_or_create_new with existing record."""
        self.manager.session = Mock()
        
        class TestModel:
            def __init__(self, **kwargs):
                self.id = kwargs.get('id')
                self.name = kwargs.get('name')
        
        existing = TestModel(id=1, name="Existing")
        
        query_mock = Mock()
        query_mock.count.return_value = 1
        query_mock.first.return_value = existing
        self.manager.session.query.return_value.filter_by.return_value = query_mock
        
        result = self.manager.query_or_create_new(TestModel, {"id": 1})
        
        assert result == existing

    def test_query_or_create_new_not_existing(self):
        """Test query_or_create_new with non-existing record."""
        self.manager.session = Mock()
        
        class TestModel:
            def __init__(self, **kwargs):
                self.id = kwargs.get('id')
                self.name = kwargs.get('name')
        
        query_mock = Mock()
        query_mock.count.return_value = 0
        self.manager.session.query.return_value.filter_by.return_value = query_mock
        
        result = self.manager.query_or_create_new(TestModel, {"id": 1, "name": "New"})
        
        assert result.id == 1
        assert result.name == "New"


class TestPostgresDatabaseManagerImpl:
    """Test cases for PostgresDatabaseManagerImpl class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PostgresDatabaseManagerImpl(engine_info="sqlite:///:memory:")

    def test_get_collection(self):
        """Test get_collection returns collection name."""
        result = self.manager.get_collection("test_table")
        assert result == "test_table"

    @patch('core.database.postgres.pd.read_sql_query')
    def test_to_dataframe(self, mock_read_sql):
        """Test converting SQL query to DataFrame."""
        mock_df = Mock()
        mock_read_sql.return_value = mock_df
        self.manager.engine = Mock()
        
        result = self.manager.to_dataframe("SELECT * FROM test")
        
        assert result == mock_df
        mock_read_sql.assert_called_once()

    @patch('core.database.postgres.auto_config')
    def test_add_limit_if_debug(self, mock_auto_config):
        """Test add_limit_if_debug adds limit in debug mode."""
        mock_auto_config.DEBUG = True
        mock_auto_config.DB_LIMIT_QUERY_RECORDS = 100
        
        result = self.manager.add_limit_if_debug("SELECT * FROM test")
        assert "limit" in result.lower()

    @patch('core.database.postgres.auto_config')
    def test_add_limit_if_not_debug(self, mock_auto_config):
        """Test add_limit_if_debug doesn't add limit in non-debug mode."""
        mock_auto_config.DEBUG = False
        
        sql = "SELECT * FROM test"
        result = self.manager.add_limit_if_debug(sql)
        assert result == sql


class TestDatabaseFactory:
    """Test cases for DatabaseFactory class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.factory = DatabaseFactory()

    def test_register_config(self):
        """Test registering database configuration."""
        config = {
            "type": DatabaseFactory.DatabaseType.POSTGRES,
            "host": "localhost",
            "port": "5432"
        }
        
        self.factory.register_config("test_db", config)
        
        assert "test_db" in self.factory._db_store
        stored_config, _ = self.factory._db_store["test_db"]
        assert stored_config["type"] == DatabaseFactory.DatabaseType.POSTGRES

    def test_build_connection_string_postgres(self):
        """Test building PostgreSQL connection string."""
        conn_str = DatabaseFactory.build_connection_string(
            DatabaseFactory.DatabaseType.POSTGRES,
            "localhost", "5432", "testdb", "user", "pass"
        )
        
        assert "postgresql://" in conn_str
        assert "localhost" in conn_str
        assert "5432" in conn_str
        assert "testdb" in conn_str

    def test_build_connection_string_mysql(self):
        """Test building MySQL connection string."""
        conn_str = DatabaseFactory.build_connection_string(
            DatabaseFactory.DatabaseType.MYSQL,
            "localhost", "3306", "testdb", "user", "pass"
        )
        
        assert "mysql+pymysql://" in conn_str

    def test_build_connection_string_sqlite(self):
        """Test building SQLite connection string."""
        conn_str = DatabaseFactory.build_connection_string(
            DatabaseFactory.DatabaseType.SQLITE,
            "", "", "test.db", "", ""
        )
        
        assert "sqlite:///test.db" == conn_str

    def test_create_manager_not_registered(self):
        """Test creating manager for non-registered database raises error."""
        with pytest.raises(KeyError):
            self.factory.create_manager("non_existent")

    def test_close(self):
        """Test closing specific database manager."""
        mock_manager = Mock()
        config = {"type": "test"}
        self.factory._db_store["test_db"] = (config, mock_manager)
        
        self.factory.close("test_db")
        
        mock_manager.close.assert_called_once()
        assert "test_db" not in self.factory._db_store

    def test_close_all(self):
        """Test closing all database managers."""
        mock_manager1 = Mock()
        mock_manager2 = Mock()
        self.factory._db_store = {
            "db1": ({"type": "test"}, mock_manager1),
            "db2": ({"type": "test"}, mock_manager2)
        }
        
        self.factory.close_all()
        
        mock_manager1.close.assert_called_once()
        mock_manager2.close.assert_called_once()
        assert len(self.factory._db_store) == 0


class TestMongoManagerBase:
    """Test cases for MongoManagerBase class."""

    def test_get_collection_name_string(self):
        """Test getting collection name from string."""
        manager = MongoManagerBase("mongodb://localhost", "testdb")
        assert manager._get_collection_name("users") == "users"

    def test_get_collection_name_document_class(self):
        """Test getting collection name from Document class."""
        class User(Document):
            pass
        
        manager = MongoManagerBase("mongodb://localhost", "testdb")
        assert manager._get_collection_name(User) == "users"

    def test_get_collection_name_document_instance(self):
        """Test getting collection name from Document instance."""
        class User(Document):
            pass
        
        user = User()
        manager = MongoManagerBase("mongodb://localhost", "testdb")
        assert manager._get_collection_name(user) == "users"

    def test_get_collection_name_invalid_type(self):
        """Test getting collection name with invalid type raises error."""
        manager = MongoManagerBase("mongodb://localhost", "testdb")
        with pytest.raises(TypeError):
            manager._get_collection_name(123)
