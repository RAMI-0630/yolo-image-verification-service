@echo off
echo Testing InstaClaim Accident Detection Service
echo.

echo 1. Testing Health Endpoint...
curl -X GET http://localhost:8000/health
echo.
echo.

echo 2. Testing Image Analysis (requires test_image.jpg)...
if exist test_image.jpg (
    curl -X POST "http://localhost:8000/api/analyze/image" ^
      -H "X-API-Key: instaclaim-dev-key-2024" ^
      -F "file=@test_image.jpg"
    echo.
) else (
    echo test_image.jpg not found, skipping...
)
echo.

echo 3. Testing URL Analysis...
curl -X POST "http://localhost:8000/api/analyze/s3" ^
  -H "X-API-Key: instaclaim-dev-key-2024" ^
  -H "Content-Type: application/json" ^
  -d "{\"imageUrl\": \"https://images.unsplash.com/photo-1449965408869-eaa3f722e40d\"}"
echo.
echo.

echo 4. Testing API Documentation...
echo Open http://localhost:8000/docs in your browser
echo.

pause
