# YOLOv8n Car Verification Microservice - Context

## Overview
A standalone Python microservice using YOLOv8n to verify car presence in images. Returns boolean validation based on expected car count (1 or 2).

## Purpose
Part of InstaClaim project - validates user-uploaded claim images contain the correct number of vehicles before processing.

## Architecture

### Technology Stack
- **Framework:** FastAPI (Python 3.10+)
- **AI Model:** YOLOv8n (Ultralytics)
- **Image Processing:** OpenCV
- **Deployment:** Docker-ready

### Project Structure
```
YOLO/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI endpoints
│   ├── models.py         # Pydantic request/response models
│   ├── detector.py       # YOLOv8n car detection logic
│   └── fraud_analyzer.py # (Not used - legacy)
├── config.py             # Environment configuration
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── .env                 # Environment variables
└── README.md            # Documentation
```

## API Specification

### Endpoint: POST /api/verify

**Purpose:** Verify if image contains expected number of cars

**Request:**
- Headers: `X-API-Key: instaclaim-dev-key-2024`
- Body (multipart/form-data):
  - `file`: Image file (JPG/PNG, max 10MB)
  - `expectedCars`: Integer (1 or 2)

**Response:**
```json
{
  "isValid": true,
  "carsDetected": 1,
  "confidenceScore": 0.95,
  "message": "Valid: Found 1 car(s) as expected"
}
```

**Response Fields:**
- `isValid`: Boolean - true if detected count matches expected
- `carsDetected`: Integer - actual number of cars detected
- `confidenceScore`: Float (0.0-1.0) - average confidence of detections
- `message`: String - human-readable result

### Endpoint: GET /health

**Purpose:** Health check

**Response:**
```json
{
  "status": "healthy",
  "service": "car-verification",
  "model": "yolov8n"
}
```

## How It Works

1. **Image Upload:** Client sends image + expected car count
2. **Preprocessing:** Image decoded and prepared for YOLO
3. **Detection:** YOLOv8n detects all objects, filters for "car" class
4. **Counting:** Counts detected cars with confidence > 0.5 (configurable)
5. **Validation:** Compares detected count with expected count
6. **Response:** Returns boolean result + metadata

## Configuration

Environment variables (`.env`):
```
API_KEY=instaclaim-dev-key-2024
MODEL_PATH=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
MAX_IMAGE_SIZE=10485760
```

## Running the Service

**Local:**
```bash
pip install -r requirements.txt
python -m app.main
```
Service runs on `http://localhost:8000`

**Docker:**
```bash
docker build -t instaclaim-detector .
docker run -p 8000:8000 instaclaim-detector
```

## Spring Boot Integration

### Service Client Example
```java
@Service
public class CarVerificationService {
    
    @Value("${car.verification.url}")
    private String verificationServiceUrl;
    
    @Value("${car.verification.api-key}")
    private String apiKey;
    
    private final RestTemplate restTemplate;
    
    public VerifyResponse verifyCarCount(MultipartFile file, int expectedCars) {
        HttpHeaders headers = new HttpHeaders();
        headers.set("X-API-Key", apiKey);
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", file.getResource());
        body.add("expectedCars", expectedCars);
        
        HttpEntity<MultiValueMap<String, Object>> request = 
            new HttpEntity<>(body, headers);
        
        return restTemplate.postForObject(
            verificationServiceUrl + "/api/verify",
            request,
            VerifyResponse.class
        );
    }
}
```

### Response Model
```java
public class VerifyResponse {
    private boolean isValid;
    private int carsDetected;
    private double confidenceScore;
    private String message;
    // getters/setters
}
```

### application.properties
```properties
car.verification.url=http://localhost:8000
car.verification.api-key=instaclaim-dev-key-2024
```

## Usage Recommendations

### Process Images One-by-One
**Recommended approach:** Verify each image immediately after upload

**Why:**
- Immediate user feedback
- User can retake failed photos while still in context
- Better UX - progressive validation
- Simpler error handling

**Example Flow:**
```
1. User uploads front damage photo → Verify (expectedCars=1) → ✅ Pass
2. User uploads rear damage photo → Verify (expectedCars=1) → ❌ Fail
   → Prompt: "No vehicle detected. Please retake rear photo."
3. User retakes photo → Verify → ✅ Pass
4. Continue with claim submission
```

### Performance
- Single image verification: ~50-200ms (CPU)
- Network overhead: ~10-50ms (local network)
- 5 sequential requests: ~300ms-1.25s total
- Negligible impact on user experience

### When to Use Batch Processing
Only if:
- Processing 10+ images at once
- Images already stored (not real-time upload)
- Generating summary reports

## Key Implementation Details

### Car Detection Logic (`app/detector.py`)
```python
def count_cars(self, image: np.ndarray) -> Tuple[int, float]:
    results = self.model(image, conf=config.CONFIDENCE_THRESHOLD)
    car_count = 0
    confidences = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            
            if class_name == 'car':
                car_count += 1
                confidences.append(float(box.conf[0]))
    
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    return car_count, avg_confidence
```

**Detection Logic:**
- Uses YOLOv8n pre-trained on COCO dataset
- Filters for "car" class only (not truck, bus, motorcycle)
- Applies confidence threshold (default 0.5)
- Returns count + average confidence

### API Endpoint (`app/main.py`)
```python
@app.post("/api/verify", response_model=VerifyResponse)
async def verify_cars(
    file: UploadFile = File(...),
    expectedCars: int = Form(...),
    api_key: str = Depends(verify_api_key)
):
    # Validate expectedCars is 1 or 2
    # Process image
    # Count cars
    # Return boolean validation
```

## Testing

**Interactive API Docs:**
Visit `http://localhost:8000/docs` after starting service

**Test Script:**
```bash
python test_simple.py
```

**Manual curl:**
```bash
curl -X POST "http://localhost:8000/api/verify" \
  -H "X-API-Key: instaclaim-dev-key-2024" \
  -F "file=@test_car.jpg" \
  -F "expectedCars=1"
```

## Dependencies

**Core:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `ultralytics` - YOLOv8n implementation
- `opencv-python-headless` - Image processing
- `torch` - Deep learning framework (PyTorch)

**Why PyTorch:**
- YOLOv8n model runs on PyTorch
- Loads `.pt` model files
- Handles neural network inference
- Required dependency for YOLO

## Security

- API key authentication via `X-API-Key` header
- File size validation (10MB limit)
- Input validation with Pydantic
- No data persistence (stateless service)

## Limitations

- Only detects "car" class (not trucks, motorcycles, buses)
- Confidence threshold may need tuning for specific use cases
- CPU inference only (GPU support requires additional setup)
- Expected car count limited to 1 or 2

## Future Enhancements (Not Implemented)

- Support for other vehicle types
- Damage severity assessment
- Fraud detection patterns
- Batch processing endpoint
- S3/URL image support
- GPU acceleration

## Notes for Integration

1. **Service should be running** before Spring Boot app starts
2. **Network latency:** Deploy on same server/network for best performance
3. **Error handling:** Handle 401 (invalid API key), 400 (invalid input), 500 (processing error)
4. **Timeouts:** Set reasonable timeout (5-10 seconds) in RestTemplate
5. **Retry logic:** Consider retry for transient failures
6. **Monitoring:** Log verification results for debugging

## Contact & Support

- Service runs independently of main application
- Can be scaled horizontally (multiple instances)
- Stateless design allows load balancing
- Docker deployment recommended for production
