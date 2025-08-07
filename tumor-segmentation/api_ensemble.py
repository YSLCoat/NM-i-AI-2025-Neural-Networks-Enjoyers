import uvicorn
import time
import datetime
import os
import numpy as np
import imageio

from fastapi import Body, FastAPI
from dtos import TumorPredictRequestDto, TumorPredictResponseDto
# from inference_unet import predict
from ensemble_unet import load_ensemble, predict

from utils import validate_segmentation, encode_request, decode_request

HOST = "0.0.0.0"
PORT = 8000

app = FastAPI()
start_time = time.time()

@app.on_event("startup")
def load_models():
    load_ensemble([
        "tumor-segmentation/models/unet_model_7_0.pth",
        "tumor-segmentation/models/unet_model_6_8.pth",
        "tumor-segmentation/models/unet_model_6_7.pth"
    ])


@app.post('/predict', response_model=TumorPredictResponseDto)
def predict_endpoint(request: TumorPredictRequestDto):
    img: np.ndarray = decode_request(request)

    save_dir = "received_images"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"image_{timestamp}.png"
    image_path = os.path.join(save_dir, filename)
    # imageio.imwrite(image_path, img)

    predicted_segmentation = predict(img)

    validate_segmentation(img, predicted_segmentation)
    encoded_segmentation = encode_request(predicted_segmentation)

    response = TumorPredictResponseDto(
        img=encoded_segmentation
    )
    return response


@app.get('/api')
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT,
        reload=True
    )
