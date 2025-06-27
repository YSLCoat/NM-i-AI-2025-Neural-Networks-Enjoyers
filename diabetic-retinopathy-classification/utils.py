import numpy as np
import cv2
import base64


def base64_to_array(encoded_img: str) -> np.ndarray:
    """
    Converts a base64 encoded image string to a numpy array using OpenCV.
    """
    np_img = np.fromstring(base64.b64decode(encoded_img), np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_ANYCOLOR)


def png_to_array(png_path: str) -> np.ndarray:
    """
    Converts a PNG image file to a numpy array using OpenCV.
    """
    img_array = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    return img_array

def show_image(image: np.ndarray, title: str = "Image"):
    """
    Displays an image using OpenCV.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
