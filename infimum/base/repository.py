"""
Repository base that works with any ORM model: one instance, multiple entity types.

Each method takes the model class (and id or payload) so the same repository
can be used for User, Order, etc. Uses a single session per operation.
"""
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from infimum.database.base import DatabaseManager

ModelType = TypeVar("ModelType")


class BaseRepository:
    """
    Multi-model repository: one instance can perform CRUD on any ORM entity type.

    Pass the model class into each method, e.g. repo.get_by_id(User, 1),
    repo.create(Order, order_data). Each operation uses a single session.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def get_by_id(self, model: Type[ModelType], id_: Any) -> Optional[ModelType]:
        """Get a record by ID. Uses a single session for the operation."""
        with self.db.get_session() as session:
            return session.query(model).filter(model.id == id_).first()

    def create(
        self,
        model: Type[ModelType],
        obj_in: Union[BaseModel, Dict[str, Any]],
    ) -> ModelType:
        """Create a new record for the given model."""
        data = obj_in.model_dump() if isinstance(obj_in, BaseModel) else obj_in
        db_obj = model(**data)
        with self.db.get_session() as session:
            session.add(db_obj)
            session.commit()
            session.refresh(db_obj)
        return db_obj

    def update(
        self,
        model: Type[ModelType],
        id_: Any,
        obj_in: Union[BaseModel, Dict[str, Any]],
    ) -> Optional[ModelType]:
        """Update an existing record. Load and update run in the same session."""
        update_data = (
            obj_in.model_dump(exclude_unset=True)
            if isinstance(obj_in, BaseModel)
            else obj_in
        )
        with self.db.get_session() as session:
            db_obj = session.query(model).filter(model.id == id_).first()
            if not db_obj:
                return None
            for field, value in update_data.items():
                setattr(db_obj, field, value)
            session.commit()
            session.refresh(db_obj)
        return db_obj

    def delete(self, model: Type[ModelType], id_: Any) -> bool:
        """Delete a record. Load and delete run in the same session."""
        with self.db.get_session() as session:
            db_obj = session.query(model).filter(model.id == id_).first()
            if not db_obj:
                return False
            session.delete(db_obj)
            session.commit()
        return True
