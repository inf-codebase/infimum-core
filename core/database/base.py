from abc import ABC, abstractmethod


class DatabaseManager(ABC):
    """Base class for database managers"""
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def insert_or_update(self, model, auto_commit=True, update_if_true_conditions=None):
        """
        Insert a new record or update an existing one based on conditions.

        Args:
            model: The model instance to insert or update
            auto_commit: Whether to commit the transaction immediately
            update_if_true_conditions: Dictionary of field-value pairs to use for finding existing records

        Returns:
            The inserted or updated model instance
        """
        pass

    @abstractmethod
    def query_or_create_new(self, model_class, query_conditions=None):
        """
        Query for an existing record or create a new instance if not found.

        Args:
            model_class: The model class to query or instantiate
            query_conditions: Dictionary of field-value pairs to use for finding existing records

        Returns:
            An existing record instance or a new (unsaved) instance
        """
        pass
