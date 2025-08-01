import os
import re
import requests
import base64
import cv2
import numpy as np
from utils import decode_request, plot_prediction, dice_score

img_dir = r"C:\Users\theod\Documents\nmki\NM-i-AI-2025-Neural-Networks-Enjoyers\tumor-segmentation\datasets\val\imgs"         # folder with patient_*.png images
label_dir = r"C:\Users\theod\Documents\nmki\NM-i-AI-2025-Neural-Networks-Enjoyers\tumor-segmentation\datasets\val\labels"        # folder with segmentation_*.png labels
url = "http://localhost:9051/predict"

img_filenames = [f for f in os.listdir(img_dir) if f.endswith('.png')]

dice_scores = []

for img_name in img_filenames:
    # Extract patient number from image filename
    match = re.search(r'patient_(\d+)', img_name)
    if not match:
        print(f"Skipping {img_name}, can't extract patient number.")
        continue
    patient_num = match.group(1)
    
    # Build label filename using extracted number
    label_name = f"segmentation_{patient_num}.png"
    
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, label_name)
    
    # Load the image and label
    img = cv2.imread(img_path)
    seg = cv2.imread(label_path)
    
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue
    if seg is None:
        print(f"Failed to load label: {label_path}")
        continue
    
    # Encode image and prepare payload
    success, encoded_img = cv2.imencode('.png', img)
    if not success:
        print(f"Failed to encode image {img_path}")
        continue
    b64_img_str = base64.b64encode(encoded_img).decode()
    payload = {"img": b64_img_str}
    
    # Send to API
    response = requests.post(url, json=payload)
    if not response.ok:
        print(f"Request failed for {img_name} with status code: {response.status_code}")
        continue
    
    # Decode predicted segmentation
    response_json = response.json()
    pred_b64 = response_json.get('img')
    if not pred_b64:
        print(f"No 'img' in response for {img_name}")
        continue
    pred_np = decode_request(type('obj', (object,), {'img': pred_b64}))
    
    # Compute and store dice score
    score = dice_score(seg, pred_np)
    dice_scores.append(score)
    print(f"{img_name} vs {label_name}: Dice score = {score:.4f}")
    
    # Visualize
    plot_prediction(img, seg, pred_np)

if dice_scores:
    avg_score = sum(dice_scores) / len(dice_scores)
    print(f"\nAverage Dice score for {len(dice_scores)} samples: {avg_score:.4f}")
else:
    print("No valid samples processed.")
