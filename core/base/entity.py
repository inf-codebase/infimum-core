from typing import Optional, Any
from pydantic import BaseModel
from datetime import datetime

from core.utils import string_utils
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.orm import DeclarativeBase

# Use the modern approach only (no decorator needed)
class BaseEntity(DeclarativeBase):
    """Base entity for SQLAlchemy ORM models"""
    # This flag tells SQLAlchemy to accept normal type annotations without Mapped[]
    __allow_unmapped__ = True
    
    # Generate __tablename__ automatically
    @classmethod
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __init__(self, **kwargs):
        """Initialize entity with keyword arguments
        This allows entities to be created with attribute values passed as keywords
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        super().__init__()

class Document(BaseModel):
    """BaseModel as document. It is appropriate for document database such as MongoDB, CouchDB, Couchbase, ...
       The document name will automatically extract base on class name. eg: `document --> documents, IssuerProfile --> issuer_profiles, AiRatingStrengthsChallenge --> ai_rating_strengths_challenges`
    Args:
        BaseModel: pydantic base class. 

    Returns:
        Document: Based entity for document.
    """
    
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def get_collection_name(cls):
        collection_name = string_utils.camel_to_plural_underscore(cls.__name__) 
        return collection_name
        
    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)
    
    