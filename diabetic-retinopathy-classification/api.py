import time
import uvicorn
import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from model import predict
from utils import load_sample

HOST = "0.0.0.0"
PORT = 4321


# Images are loaded via cv2, encoded via base64 and sent as strings
# See utils.py for details
class DiabeticRetinopathyPredictRequestDto(BaseModel):
    fundus_image: str


class DiabeticRetinopathyPredictResponseDto(BaseModel):
    """
    severity_level should be an integer between 0 and 4, where:
    - 0: No DR
    - 1: Mild non-proliferative DR
    - 2: Moderate non-proliferative DR
    - 3: Severe non-proliferative DR
    - 4: Proliferative DR
    """
    severity_level: int


app = FastAPI()
start_time = time.time()


@app.get('/api')
def hello():
    return {
        "service": "diabetic-retinopathy-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


@app.post('/predict', response_model=DiabeticRetinopathyPredictResponseDto)
def predict_endpoint(request: DiabeticRetinopathyPredictRequestDto):

    # Decode request
    image_id = load_sample(request.fundus_image)

    predicted_severity_level = predict(image_id)

    # Return the encoded image to the validation/evalution service
    response = DiabeticRetinopathyPredictResponseDto(
        severity_level=predicted_severity_level
    )

    return response


if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
