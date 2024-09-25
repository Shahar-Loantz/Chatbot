from pydantic import BaseModel
from typing import List, Optional, Dict

class MovieQuery(BaseModel):
    fields: Optional[List[str]] = None # The fields to display for each movie
    search_criteria: Optional[Dict[str, str]] = None # Input for each selected field

