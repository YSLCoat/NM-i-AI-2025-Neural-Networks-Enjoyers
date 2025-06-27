from utils import png_to_array, show_image
from model import predict

FUNDUS_IMG = "data/training/0a4e1a29ffff.png" 

image = png_to_array(FUNDUS_IMG)
sample_prediction = predict(image)

print("Predicted class:", sample_prediction)
show_image(image, str(sample_prediction))

