# api.py

import uvicorn
import datetime
import time
from contextlib import asynccontextmanager
from fastapi import Body, FastAPI

# Import the DTOs and the new model service
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
import model_service

HOST = "0.0.0.0"
PORT = 8001

# Use FastAPI's lifespan manager to load the model on startup
# and clean up on shutdown. This is the modern replacement for
# on_event("startup") and on_event("shutdown").
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model and other resources
    model_service.load_model()
    yield
    # Shutdown: Clean up resources
    model_service.close_model()

app = FastAPI(lifespan=lifespan)
start_time = time.time()


@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    """
    Receives game state, uses the ML model to predict an action,
    and returns it.
    """
    # The heavy lifting is now done in the model_service
    predicted_actions = model_service.predict_action(request.dict())
    
    return RaceCarPredictResponseDto(
        actions=predicted_actions
    )

@app.get('/api')
def api_info():
    return {
        "service": "race-car-usecase-model-server",
        "uptime": f"{datetime.timedelta(seconds=time.time() - start_time)}"
    }

@app.get('/')
def index():
    return "Your model endpoint is running! Send POST requests to /predict."


if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )