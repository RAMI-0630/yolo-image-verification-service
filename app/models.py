from pydantic import BaseModel
from typing import Optional

class VerifyRequest(BaseModel):
    expectedCars: int  # 1 or 2

class VerifyResponse(BaseModel):
    isValid: bool
    carsDetected: int
    confidenceScore: float
    message: str
