import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple
import config

class CarDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format")
        return img
    
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
