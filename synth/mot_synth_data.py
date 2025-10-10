#!/usr/bin/env python3
import json
import random
import argparse
from datetime import datetime, timedelta
import math

class COCOMaritimeDataGenerator:
    def __init__(self):
        # Match your real dataset categories
        self.categories = [
            {"id": 1, "name": "swimmer", "supercategory": "person"},
            {"id": 2, "name": "swimmer with life jacket", "supercategory": "person"},
            {"id": 3, "name": "boat", "supercategory": "boat"},
            {"id": 6, "name": "life jacket", "supercategory": "lifejacket"}
        ]
        
    def generate_coco_dataset(self, num_images=100, scenario_type='rescue'):
        """Generate COCO-format synthetic maritime data"""
        dataset = {
            "info": {
                "year": "2025",
                "version": "1.0",
                "description": f"Synthetic maritime data - {scenario_type}",
                "date_created": datetime.utcnow().isoformat()
            },
            "licenses": [{"id": 1, "url": "", "name": "Synthetic"}],
            "categories": self.categories,
            "images": [],
            "annotations": []
        }
        
        annotation_id = 1
        base_time = datetime.utcnow()
        
        # Realistic GPS bounds (similar to your real data)
        base_lat = 47.67
        base_lon = 9.27
        
        for img_id in range(num_images):
            timestamp = base_time + timedelta(seconds=img_id * 0.033)  # ~30fps
            
            # Drone metadata matching your real data structure
            altitude = random.uniform(5, 50)
            gimbal_pitch = random.uniform(-90, 0)
            compass_heading = random.uniform(0, 360)
            
            # Slight GPS drift
            lat = base_lat + random.uniform(-0.001, 0.001)
            lon = base_lon + random.uniform(-0.001, 0.001)
            
            image_info = {
                "id": img_id,
                "file_name": f"synthetic_{img_id:06d}.png",
                "height": 2160,
                "width": 3840,
                "date_time": timestamp.isoformat(),
                "meta": {
                    "date_time": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "gps_latitude": lat,
                    "gps_latitude_ref": "N",
                    "gps_longitude": lon,
                    "gps_longitude_ref": "E",
                    "altitude": altitude,
                    "gimbal_pitch": gimbal_pitch,
                    "compass_heading": compass_heading,
                    "gimbal_heading": compass_heading + random.uniform(-5, 5),
                    "speed": random.uniform(0, 2),
                    "xspeed": random.uniform(-1, 1),
                    "yspeed": random.uniform(-1, 1),
                    "zspeed": random.uniform(-0.5, 0.5)
                },
                "source": {
                    "drone": "synthetic_mavic",
                    "scenario": scenario_type
                }
            }
            
            dataset["images"].append(image_info)
            
            # Generate annotations based on scenario
            annotations = self._generate_annotations_for_scenario(
                img_id, scenario_type, annotation_id
            )
            
            dataset["annotations"].extend(annotations)
            annotation_id += len(annotations)
        
        return dataset
    
    def _generate_annotations_for_scenario(self, image_id, scenario_type, start_annotation_id):
        """Generate realistic bounding box annotations"""
        annotations = []
        annotation_id = start_annotation_id
        
        # Scenario-specific object distributions
        if scenario_type == 'rescue':
            # 1-3 swimmers, possibly with life jackets
            num_swimmers = random.randint(1, 3)
            for i in range(num_swimmers):
                has_life_jacket = random.random() < 0.6
                category_id = 2 if has_life_jacket else 1
                
                # Realistic bounding box (swimmers are small)
                x = random.uniform(500, 3000)
                y = random.uniform(500, 1800)
                width = random.uniform(30, 80)
                height = random.uniform(40, 100)
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "confidence": random.uniform(0.7, 0.98)
                })
                annotation_id += 1
                
        elif scenario_type == 'patrol':
            # 0-2 boats
            num_boats = random.randint(0, 2)
            for i in range(num_boats):
                # Boats are larger
                x = random.uniform(200, 3200)
                y = random.uniform(300, 1600)
                width = random.uniform(150, 400)
                height = random.uniform(100, 300)
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 3,  # boat
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "confidence": random.uniform(0.75, 0.95),
                    "attributes": {
                        "vessel_type": random.choice(["small_craft", "fishing_vessel", "patrol_boat"]),
                        "speed_estimate_knots": random.uniform(5, 25)
                    }
                })
                annotation_id += 1
                
        elif scenario_type == 'threat':
            # Fast approaching boats (threat scenario)
            num_threats = random.randint(1, 4)
            for i in range(num_threats):
                x = random.uniform(1000, 3500)
                y = random.uniform(500, 1500)
                width = random.uniform(100, 250)
                height = random.uniform(80, 200)
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 3,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "confidence": random.uniform(0.7, 0.9),
                    "attributes": {
                        "vessel_type": "small_fast_craft",
                        "speed_estimate_knots": random.uniform(20, 35),
                        "is_threat": True,
                        "threat_indicators": {
                            "high_speed": True,
                            "intercept_course": random.random() < 0.7,
                            "no_ais": random.random() < 0.8
                        }
                    }
                })
                annotation_id += 1
        
        return annotations
    
    def save_to_json(self, dataset, output_file):
        """Save COCO dataset to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset['images'])} images with {len(dataset['annotations'])} annotations to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate COCO-format synthetic maritime data')
    parser.add_argument('--num-images', type=int, default=1000)
    parser.add_argument('--scenarios', default='rescue,patrol,threat')
    parser.add_argument('--output-dir', default='./synthetic_coco')
    
    args = parser.parse_args()
    
    generator = COCOMaritimeDataGenerator()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    for scenario in args.scenarios.split(','):
        print(f"Generating {scenario} scenario...")
        dataset = generator.generate_coco_dataset(args.num_images, scenario.strip())
        output_file = os.path.join(args.output_dir, f'synthetic_{scenario}.json')
        generator.save_to_json(dataset, output_file)

if __name__ == '__main__':
    main()