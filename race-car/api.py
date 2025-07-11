import time
import uvicorn
import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from model import predict
from utils import load_sample
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from test_endpoint import return_random
HOST = "0.0.0.0"
PORT = 4321


@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    action = return_random(request.dict())
    return RaceCarPredictResponseDto(action_type=action)


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
