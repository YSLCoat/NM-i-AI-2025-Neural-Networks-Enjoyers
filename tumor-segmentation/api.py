import uvicorn
import time
import datetime
import os
import numpy as np
import imageio

from fastapi import Body, FastAPI
from dtos import TumorPredictRequestDto, TumorPredictResponseDto
# from example import predict
# from inference_mrcnn import predict
from inference_unet import predict
from utils import validate_segmentation, encode_request, decode_request


HOST = "0.0.0.0"
PORT = 8000


app = FastAPI()
start_time = time.time()


@app.post('/predict', response_model=TumorPredictResponseDto)
def predict_endpoint(request: TumorPredictRequestDto):
    # Decode request str to numpy array
    img: np.ndarray = decode_request(request)

    # Save image locally
    save_dir = "received_images"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"image_{timestamp}.png"
    image_path = os.path.join(save_dir, filename)
    # imageio.imwrite(image_path, img)

    # Obtain segmentation prediction
    predicted_segmentation = predict(img)

    # Validate segmentation format
    validate_segmentation(img, predicted_segmentation)

    # Encode the segmentation array to a str
    encoded_segmentation = encode_request(predicted_segmentation)

    # Return the encoded segmentation to the validation/evalution service
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
