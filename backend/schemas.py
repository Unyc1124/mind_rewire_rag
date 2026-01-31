# backend/schemas.py

from pydantic import BaseModel

class UserInput(BaseModel):
    text: str
    age_group: str  # "teen" | "adult"

class NavigatorResponse(BaseModel):
    status: str
    message: str
