'''
References: 
Ultralytics (2024). Predict - YOLOv8 Docs. [online] Ultralytics. 
Available at: https://docs.ultralytics.com/modes/predict/ [Accessed 1st april 2025].
'''

import os
from ultralytics import YOLO
import cv2

# initialisation for image path, YOLO model result and input image patch
image_path = "dataset/original_images/img12.jpg" 

# load trained YOLO model for detecting players. 
model = YOLO("YOLOV8N_BEST.pt")

# player class id = 2 because the model also detects goalkeepers, referees and footballs. 
PLAYER_CLASS_ID = 2

# store results from YOLO and image to extract cropped player images
results = model(image_path)
input_image = cv2.imread(image_path)

# Create the desired folder if it doesn't exist
output_folder = "dataset/extracted_players/test"
os.makedirs(output_folder, exist_ok=True) 

# Process the detected bounding boxes
for r in results:
    # Access bounding boxes
    for i, box in enumerate(r.boxes):  
        class_id = int(box.cls[0])

        # Only process "player" detections based on class_id 
        if class_id == PLAYER_CLASS_ID:  
            # Extract bounding box coordinates and convert to integers. 
            xyxy = box.xyxy[0].tolist()  
            x_min, y_min, x_max, y_max = map(int, xyxy) 

            # Crop the bounding box from the original image
            cropped_image = input_image[y_min:y_max, x_min:x_max]

            # Save the cropped image and print confirmation message.
            cropped_image_path = os.path.join(output_folder, f"cropped_player_{i}.jpg")
            cv2.imwrite(cropped_image_path, cropped_image)

            print(f"Cropped player image saved at: {cropped_image_path}")
