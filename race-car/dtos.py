from pydantic import BaseModel
from typing import Dict, Optional


class RaceCarPredictRequestDto(BaseModel):
    did_crash: bool
    elapsed_time_ms: int
    distance: int
    velocity: Dict[str, int]  
    sensors: Dict[str, Optional[int]]  

class RaceCarPredictResponseDto(BaseModel):
    action_type: str
    action_amount: Optional[int] = None    # Possible types:
    # 'ACCELERATE'
    # 'DECELERATE'
    # 'STEER_LEFT'
    # 'STEER_RIGHT'
    # 'NOTHING''
