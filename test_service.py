import requests
import os

BASE_URL = "http://localhost:8000"
API_KEY = "instaclaim-dev-key-2024"
HEADERS = {"X-API-Key": API_KEY}

def test_health():
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_analyze_image(image_path):
    print(f"Testing /api/analyze/image with {image_path}...")
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{BASE_URL}/api/analyze/image",
            headers=HEADERS,
            files=files
        )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_analyze_url(image_url):
    print(f"Testing /api/analyze/s3 with URL...")
    data = {"imageUrl": image_url}
    response = requests.post(
        f"{BASE_URL}/api/analyze/s3",
        headers=HEADERS,
        json=data
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

if __name__ == "__main__":
    test_health()
    
    # Test with local image if exists
    if os.path.exists("test_image.jpg"):
        test_analyze_image("test_image.jpg")
    
    # Test with online image
    test_analyze_url("https://images.unsplash.com/photo-1449965408869-eaa3f722e40d")
