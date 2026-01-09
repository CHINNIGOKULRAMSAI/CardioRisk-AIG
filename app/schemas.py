from pydantic import BaseModel, Field

class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: int = Field(..., ge=1, le=2)
    height: int = Field(..., ge=120, le=220)
    weight: float = Field(..., ge=30, le=200)
    ap_hi: int = Field(..., ge=80, le=250)
    ap_lo: int = Field(..., ge=40, le=150)
    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)
