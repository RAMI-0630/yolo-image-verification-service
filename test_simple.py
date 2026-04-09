import requests

BASE_URL = "http://localhost:8000"
API_KEY = "instaclaim-dev-key-2024"
HEADERS = {"X-API-Key": API_KEY}

def test_health():
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_verify_cars(image_path, expected_cars):
    print(f"Testing /api/verify with {image_path}, expecting {expected_cars} car(s)...")
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'expectedCars': expected_cars}
        response = requests.post(
            f"{BASE_URL}/api/verify",
            headers=HEADERS,
            files=files,
            data=data
        )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

if __name__ == "__main__":
    test_health()
    
    # Test with your image
    # test_verify_cars("test_image.jpg", 1)  # Expecting 1 car
    # test_verify_cars("test_image.jpg", 2)  # Expecting 2 cars
