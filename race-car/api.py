import time
import uvicorn
import datetime
from fastapi import Body, FastAPI
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from test_endpoint import return_random
HOST = "0.0.0.0"
PORT = 4243


app = FastAPI()
start_time = time.time()

@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    action = return_random(request.dict())
    return RaceCarPredictResponseDto(
        action_type=action['action_type'],
        action_amount=action['action_amount']
    )

@app.get('/api')
def hello():
    return {
        "service": "diabetic-retinopathy-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }


@app.get('/')
def index():
    return "Your endpoint is running!"




if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
