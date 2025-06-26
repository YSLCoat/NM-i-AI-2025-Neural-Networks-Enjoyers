import numpy as np
import random

### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(fundus_image: np.ndarray) -> int:

    severity_level = example_model(fundus_image)

    """
    severity_level should be an integer between 0 and 4, where:
    - 0: No DR
    - 1: Mild non-proliferative DR
    - 2: Moderate non-proliferative DR
    - 3: Severe non-proliferative DR
    - 4: Proliferative DR
    """

    return severity_level


def example_model(fundus_image: np.ndarray) -> int:

    severity_level = random.randint(0, 4) # Randomly generate a severity level between 0 and 4

    return severity_level