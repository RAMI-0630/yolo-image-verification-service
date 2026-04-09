from typing import List
import numpy as np

class FraudAnalyzer:
    @staticmethod
    def analyze(detections: List[dict], image_shape: tuple) -> List[str]:
        indicators = []
        
        vehicle_detections = [d for d in detections if d['class'] in ['car', 'truck', 'bus', 'motorcycle']]
        person_detections = [d for d in detections if d['class'] == 'person']
        
        # Check for minimal damage with multiple vehicles
        if len(vehicle_detections) > 1:
            damage_levels = [d.get('damageLevel') for d in vehicle_detections]
            if all(level == 'minor' for level in damage_levels if level):
                indicators.append('minimal_damage_multiple_vehicles')
        
        # Check for staged positioning (vehicles too perfectly aligned)
        if len(vehicle_detections) >= 2:
            centers = []
            for d in vehicle_detections:
                bbox = d['boundingBox']
                center_x = bbox['x'] + bbox['width'] / 2
                center_y = bbox['y'] + bbox['height'] / 2
                centers.append((center_x, center_y))
            
            if len(centers) >= 2:
                distances = []
                for i in range(len(centers) - 1):
                    dist = np.sqrt((centers[i][0] - centers[i+1][0])**2 + 
                                 (centers[i][1] - centers[i+1][1])**2)
                    distances.append(dist)
                
                avg_distance = np.mean(distances)
                if avg_distance < 100:  # Very close positioning
                    indicators.append('staged_positioning')
        
        # Check for inconsistent damage (no visible damage but severe classification)
        severe_count = sum(1 for d in vehicle_detections if d.get('damageLevel') == 'severe')
        if severe_count == 0 and len(vehicle_detections) > 0:
            high_conf_vehicles = [d for d in vehicle_detections if d['confidence'] > 0.9]
            if len(high_conf_vehicles) > 1:
                indicators.append('inconsistent_damage')
        
        # Check for suspicious person count
        if len(person_detections) > 4:
            indicators.append('excessive_people_present')
        
        # No damage detected
        if len(vehicle_detections) > 0 and all(d.get('damageLevel') == 'minor' for d in vehicle_detections):
            indicators.append('minimal_visible_damage')
        
        return indicators
