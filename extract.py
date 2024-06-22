import cv2
import numpy as np

def is_grayscale(image):
    return image.shape[2] == 1

def calculate_brightness(image):
    if is_grayscale(image):
        return np.mean(image)
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_image)

def calculate_contrast(image):
    if is_grayscale(image):
        return np.std(image)
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray_image)

def main():
    
    image = cv2.imread('group 2.jpg')

    grayscale = is_grayscale(image)
    if grayscale:
        print("Image is grayscale")
    else:
        print("Image is not grayscale")
    
    brightness = calculate_brightness(image)
    print("Brightness:", brightness)

    contrast = calculate_contrast(image)
    print("Contrast:", contrast)

if __name__ == "__main__":
    main()
