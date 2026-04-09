from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Form
from typing import Optional
import config
from app.models import VerifyResponse
from app.detector import CarDetector

app = FastAPI(title="InstaClaim Car Verification Service", version="1.0.0")

detector = CarDetector()

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "car-verification", "model": "yolov8n"}

@app.post("/api/verify", response_model=VerifyResponse)
async def verify_cars(
    file: UploadFile = File(...),
    expectedCars: int = Form(...),
    api_key: str = Depends(verify_api_key)
):
    if expectedCars not in [1, 2]:
        raise HTTPException(
            status_code=400, 
            detail=f"expectedCars must be 1 or 2 (received: {expectedCars})"
        )
    
    if file.size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    if file.size > config.MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image too large")
    
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images allowed")
    
    try:
        image_bytes = await file.read()
        image = detector.preprocess_image(image_bytes)
        cars_detected, confidence_score = detector.count_cars(image)
        
        # Allow up to 3 extra cars (background/passing cars)
        tolerance = 3
        is_valid = expectedCars <= cars_detected <= expectedCars + tolerance
        
        if is_valid:
            if cars_detected == expectedCars:
                message = f"Valid: Found {cars_detected} car(s) as expected"
            else:
                message = f"Valid: Found {cars_detected} car(s) (expected {expectedCars}, within tolerance)"
        else:
            message = f"Invalid: Expected {expectedCars} car(s), but found {cars_detected} (exceeds tolerance)"
        
        return VerifyResponse(
            isValid=is_valid,
            carsDetected=cars_detected,
            confidenceScore=confidence_score,
            message=message
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
