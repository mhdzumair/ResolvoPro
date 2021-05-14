import tensorflow.keras
import numpy as np
import cv2
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def image_resize(image, height, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# this function crops to the center of the resize image


def cropTo(img):
    size = 224
    height, width = img.shape[:2]

    sideCrop = (width - 224) // 2
    return img[:, sideCrop : (width - sideCrop)]


def predict(image_path):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model("keras_model.h5", compile=False)

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    # image = Image.open(image_path)

    image = cv2.imread(image_path)
    image = image_resize(image, height=224)
    image = cropTo(image)

    # flips the image
    image = cv2.flip(image, 1)

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    # size = (224, 224)
    # image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    # image_array = np.asarray(image)

    # display the resized image
    # image.show()

    # Normalize the image
    normalized_image_array = (image.astype(np.float32) / 127.0) - 1
    # print(normalized_image_array)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    print(prediction)

    prediction = "number" if prediction[0][0] > prediction[0][1] else "star"
    return prediction


if __name__ == "__main__":
    print(predict("captcha_solved/captcha0.png"))
