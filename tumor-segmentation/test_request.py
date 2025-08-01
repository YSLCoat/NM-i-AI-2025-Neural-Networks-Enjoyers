import requests
import base64
import cv2
import numpy as np
from utils import decode_request, plot_prediction

# Load your test image (PET MIP)
img_path = r"C:\Users\theod\Documents\nmki\NM-i-AI-2025-Neural-Networks-Enjoyers\tumor-segmentation\data\patients\imgs\patient_000.png"
pet_mip = cv2.imread(img_path)

# Load your ground truth segmentation image path
labels_path = r"C:\Users\theod\Documents\nmki\NM-i-AI-2025-Neural-Networks-Enjoyers\tumor-segmentation\data\patients\labels\segmentation_000.png"
seg = cv2.imread(labels_path)

# Encode image as PNG and base64 string
success, encoded_img = cv2.imencode('.png', pet_mip)
if not success:
    raise Exception("Failed to encode image")
b64_img_str = base64.b64encode(encoded_img).decode()

# Prepare JSON payload for the API
payload = {
    "img": b64_img_str
}

# Send POST request to your API
url = "http://localhost:9051/predict"
response = requests.post(url, json=payload)

if response.ok:
    response_json = response.json()
    b64_pred_img = response_json['img']  # predicted segmentation base64 string

    # Decode predicted segmentation to NumPy array
    class DummyRequest:
        def __init__(self, img):
            self.img = img

    pred_seg = decode_request(DummyRequest(b64_pred_img))

    # Visualize prediction + dice score
    plot_prediction(pet_mip, seg, pred_seg)

else:
    print("Request failed with status code:", response.status_code)
    print("Response:", response.text)