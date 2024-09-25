from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone

class Movie(BaseModel):
    movie_title: str
    rotten_tomatoes_link: str
    description: Optional[str]
    critics_consensus: Optional[str]
    genres: List[str]
    actors: List[str]
    runtime: Optional[int]
    rank: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Actor(BaseModel):
    name: str
    movies: List[str]

class Message(BaseModel):
    role: str  # 'user' or 'system'
    content: str
    timestamp: datetime = datetime.now(timezone.utc)

class Conversation(BaseModel):
    conversation_id: str
    messages: List[Message]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))