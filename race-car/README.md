# Race Car

Ready, set, GO! It's racing time. 

Race against the competition to go the furthest in the allotted time, but be careful, one small crash can end your run!

Image
/home/rebecca/projects/dm-i-ai-2025-evaluation-service_new/race_car_game/cf7a5fb356344b6bb9acd94940a84d72/frame_0348.png

## About the game
You control the yellow car. Red and blue cars will spawn in random lanes - it is your job to dodge them. The car is equipped with 8 evenly spaced sensors - each being able to find obstacles within a 1000px. Figure 2 shows an image of the sensors with names.

Each tick the game is updated. The game runs with 60 ticks per second. A queue stores future actions, and each tick, an action is popped from the queue and applied to the car. If there are no actions in the queue, it will do the 'NOTHING' action. 

A game runs for up to 60 seconds - 3600 ticks. If you crash, the game will immediately end. Once the game ends, the final score will be the distance you achieved. 

### Environment
The game runs with 5 lanes of equal size. Your car will spawn in the center lane, while adversarial cars will spawn randomly in other lanes. The adversarial cars will not leave their lane, and only one adversarial car can be in each lane at a time. They can spawn in front or behind your car, and spawn with a speed relative to yours. 

On the top and bottom of the screens are walls. If you hit the walls your car will crash, so no off roading in this one. 

### Your Goal
1. dont crash into adversarial cars
2. dont crash into walls
3. go fast
4. go far


Your goal is to go as far as you can in one minute. Your game will **end** if you crash into adversarial cars or into walls. Your final score will be based on your distance.

Train a model to interpret the sensor input and respond with commands for your car.

### Controls

Pygame has been used to setup visualisation of the game locally. Initial controls using arrowkeys have been added. Change this to your won logic. 

To communicate with the server for validation and evaluation, use the functions found in dtos.py. You can test if these work using the *test connection* button on cases.dmiai.dk #FIX




### Sensors

Sensor output is your information from the game. There are 8 sensors on the car, each is positioned at a specific angle (in degrees) relative to the center of the car and has a reach of 1000 pixels Figure X #FIX shows the sensors. Below all sensors are listed. 

![Sensors](race_car_game/cf7a5fb356344b6bb9acd94940a84d72/frame_0425.png)

**List of Sensors (angle, name):**

| Angle   | Name               |
|---------|--------------------|
| 0       | left_side          |
| 22.5    | left_side_front    |
| 45      | left_front         |
| 67.5    | front_left_front   |
| 90      | front              |
| 112.5   | front_right_front  |
| 135     | right_front        |
| 157.5   | right_side_front   |
| 180     | right_side         |
| 202.5   | right_side_back    |
| 225     | right_back         |
| 247.5   | back_right_back    |
| 270     | back               |
| 292.5   | back_left_back     |
| 315     | left_back          |
| 337.5   | left_side_back     |

Each sensor is positioned at the specified angle (in degrees) relative to the center of the car and has a reach of 1000 pixels

## Scoring

Your score will be based on your distance. Scores will be normalised, lowest will recieve 0 and highest 1. Only scores above the baseline will count. If your score is below the baseline, it will auromatically get 0. 

## Evaluation

Once you are ready to evaluate your final model, start your evaluation attempt. You only have **ONE** try, so make sure the model is ready for the final test. Your score from the evaluation is the one you will be judged on. Remember you can so a validation check as many times as you need.

The evaluation opens up on Thursday the 7th, and will have a preset seed.

## Quickstart

```cmd
git clone https://github.com/amboltio/DM-i-AI-2025
cd DM-i-AI-2024/race-car
```

Install dependencies
```cmd
pip install -r requirements.txt
```

### Serve your endpoint
Serve your endpoint locally and test that everything starts without errors

```cmd
python api.py
```
Open a browser and navigate to http://localhost:XXXX. #FIX You should see a message stating that the endpoint is running. 
Feel free to change the `HOST` and `PORT` settings in `api.py`. 

You can send the following action responses:
- NOTHING
- ACCELERATE
- DECELERATE
- STEER_RIGHT
- STEER_LEFT

If you do not add an action amount, it will default to None, and one action will be added to the queue. 

### Run the simulation locally
```cmd
cd src/game
python core.py
```
By default the action input will use arrowkeys. 


**We recommend you do not change the amount of lanes or the size of the game during training.**