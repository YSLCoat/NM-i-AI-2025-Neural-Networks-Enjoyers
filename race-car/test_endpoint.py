import random

def return_random(state):
    random_number = random.randint(0,4)
    response_list = ['ACCELERATE',
        'DECELERATE',
        'STEER_LEFT',
        'STEER_RIGHT',
        'NOTHING']
    response = response_list[random_number]
    return response