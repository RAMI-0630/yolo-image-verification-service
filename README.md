# InstaClaim Car Verification Service

YOLOv8n-based microservice to verify if an image contains exactly 1 or 2 cars.

## Features

- Car detection using YOLOv8n
- Boolean verification (returns true/false)
- Configurable expected car count (1 or 2)
- REST API with authentication

## Quick Start

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Run the service:**
```bash
python -m app.main
```

Service runs on `http://localhost:8000`

### Docker Deployment

1. **Build image:**
```bash
docker build -t instaclaim-detector .
```

2. **Run container:**
```bash
docker run -p 8000:8000 -e API_KEY=your-key instaclaim-detector
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Verify Car Count
```bash
POST /api/verify
Headers: X-API-Key: your-api-key
Body: multipart/form-data with 'file' and 'expectedCars' fields
```

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/api/verify" \
  -H "X-API-Key: instaclaim-dev-key-2024" \
  -F "file=@car_image.jpg" \
  -F "expectedCars=1"
```

## Response Format

```json
{
  "isValid": true,
  "carsDetected": 1,
  "confidenceScore": 0.95,
  "message": "Valid: Found 1 car(s) as expected"
}
```

## Spring Boot Integration

### Add RestTemplate Configuration

```java
@Configuration
public class RestTemplateConfig {
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
```

### Create Service Client

```java
@Service
public class CarVerificationService {
    
    @Value("${car.verification.url}")
    private String verificationServiceUrl;
    
    @Value("${car.verification.api-key}")
    private String apiKey;
    
    private final RestTemplate restTemplate;
    
    public CarVerificationService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }
    
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

### application.properties

```properties
car.verification.url=http://localhost:8000
car.verification.api-key=instaclaim-dev-key-2024
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| API_KEY | instaclaim-dev-key-2024 | Authentication key |
| MODEL_PATH | yolov8n.pt | YOLOv8 model file |
| CONFIDENCE_THRESHOLD | 0.5 | Detection confidence threshold |
| MAX_IMAGE_SIZE | 10485760 | Max image size (10MB) |
| AWS_REGION | us-east-1 | AWS region for S3 |

## Development

**Run tests:**
```bash
python test_simple.py
```

**API Documentation:**
Visit `http://localhost:8000/docs` for interactive Swagger UI

## Production Deployment

1. Set strong API_KEY in environment
2. Configure AWS credentials for S3 access
3. Use reverse proxy (nginx) for SSL termination
4. Set up monitoring and logging
5. Scale horizontally with load balancer

## License

Proprietary - InstaClaim Project
