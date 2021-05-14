from keras.models import load_model
from helpers import resize_to_fit
import numpy as np
import cv2
import pickle


MODEL_FILENAME = "number_model.hdf5"
MODEL_LABELS_FILENAME = "number_model_labels.dat"

with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
number_model = load_model(MODEL_FILENAME)


def predict(image_file):
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 28, 70, L2gradient=True)
    dilated = cv2.dilate(canny, (2, 2), iterations=1)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_image_regions = []

    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if (5 <= w <= 20) and (4 <= h <= 21):
            letter_image_regions.append((x, y, w, h))
        elif w > 20:
            half_width = int(w / 2) + 3
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))

    if len(letter_image_regions) != 3:
        print("image contours not 3 path: ", image_file)
        return
    

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    predictions = []

    # loop over the lektters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        margin = 1 
        letter_image = gray[y - margin : y + h + margin, x - margin : x + w + margin]

        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = number_model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)


    # Print the captcha's text
    captcha_text = "".join(predictions)
    value = eval(captcha_text)
    print("CAPTCHA text is: {} = {}".format(captcha_text, value))
    return value

if __name__ == "__main__":
    try:
        while True:
            name = input("Enter file name: ")
            predict("numbers/" + name + ".png")
    except KeyboardInterrupt:
        pass