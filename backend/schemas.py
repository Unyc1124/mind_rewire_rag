# backend/schemas.py

from pydantic import BaseModel
from typing import List

class UserInput(BaseModel):
    text: str
    age_group: str  # "teen" | "adult"
class NavigatorResponse(BaseModel):
    status: str
    summary: str                # main result
    focus_areas: List[str]      # key domains
    plan_today: List[str]       # immediate actions
    plan_week: List[str]        # 7-day plan
