from collections import Counter
import pickle

import cv2
import imutils
from keras.models import load_model
import numpy as np
from sklearn.cluster import KMeans


class CaptchaSolver:
    def __init__(self):
        with open("number_model_labels.dat", "rb") as f:
            self.label = pickle.load(f)

        self.number_model = load_model("number_model.hdf5")
        self.captcha_model = load_model("keras_model.h5", compile=False)

    @staticmethod
    def solve_star(gray):
        # image = cv2.imread(image_path)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 28, 47, L2gradient=True)
        dilated = cv2.dilate(canny, (2, 2), iterations=5)
        contours, _ = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        count = 0
        for contour in contours:
            *_, width, height = cv2.boundingRect(contour)
            if (12 <= width <= 19) and (12 <= height <= 19):
                count += 1
            elif width > 19:
                count += round(width / 16)
            elif height > 19:
                count += round(height / 16)
        print("star count: ", count)
        return count

    def solve_number(self, gray):
        # image = cv2.imread(image_file)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 28, 70, L2gradient=True)
        dilated = cv2.dilate(canny, (2, 2), iterations=1)
        contours, _ = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        letter_image_regions = []

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if (5 <= w <= 20) and (4 <= h <= 21):
                letter_image_regions.append((x, y, w, h))
            elif w > 20:
                half_width = int(w / 2) + 3
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))

        if len(letter_image_regions) != 3:
            print("image contours not 3 path: ")
            return 0

        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
        predictions = []

        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box
            margin = 1
            letter_image = gray[
                y - margin : y + h + margin, x - margin : x + w + margin
            ]
            letter_image = resize_to_fit(letter_image, 20, 20)
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            prediction = self.number_model.predict(letter_image)
            letter = self.label.inverse_transform(prediction)[0]
            predictions.append(letter)

        captcha_text = "".join(predictions)
        try:
            value = eval(captcha_text)
        except SyntaxError:
            print("predict failed!")
            value = 0
        print("Maths : {} = {}".format(captcha_text, value))
        return value

    def predict_captcha(self, gray):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # image = cv2.imread(image_path)
        image = image_resize(gray, height=224)
        image = cropTo(image)
        image = cv2.flip(image, 1)
        normalized_image_array = (image.astype(np.float32) / 127.0) - 1

        data[0] = normalized_image_array
        prediction = self.captcha_model.predict(data)
        prediction = "number" if prediction[0][0] > prediction[0][1] else "star"
        return prediction


class Preprocessor:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=3)

    def find_background(self, image):
        modified_image = image.reshape(image.shape[0] * image.shape[1], 3)
        labels = self.kmeans.fit_predict(modified_image)
        counts = Counter(labels)
        counts = {
            key: value
            for key, value in sorted(
                counts.items(), key=lambda item: item[1], reverse=True
            )
        }
        center_colors = self.kmeans.cluster_centers_
        ordered_colors = [center_colors[i] for i in counts.keys()]
        rgb_colors = [rgbvalue(color) for color in ordered_colors]
        background = rgb_colors[1] if rgb_colors[0] == [0, 0, 0] else rgb_colors[0]
        return background

    def argonclick(self, image):
        background = self.find_background(image.copy())
        red = background[0]
        moded = None
        if red in range(80, 85):
            # purple background reduce noise
            moded = change_color(image.copy(), (0, 0, 0), (100, 99, 149))
            change_color(moded, (61, 60, 109), (100, 99, 149))
            change_color(moded, [109, 140, 95], (100, 99, 149))
            change_color(moded, (91, 91, 139), (100, 99, 149))
            change_color(moded, (95, 95, 144), (100, 99, 149))
            change_color(moded, (68, 68, 116), (100, 99, 149))
            change_color(moded, (61, 61, 109), (100, 99, 149))
            change_color(moded, (79, 78, 126), (100, 99, 149))
            change_color(moded, (31, 31, 31), (100, 99, 149))
            change_color(moded, (62, 62, 62), (100, 99, 149))

        elif red in range(108, 113) or red in range(98, 100):
            # green background reduce noice
            moded = change_color(image.copy(), (0, 0, 0), [87, 118, 74])
            change_color(moded, (138, 168, 125), [87, 118, 74])
            change_color(moded, (78, 78, 126), [87, 118, 74])
            change_color(moded, (110, 140, 96), [87, 118, 74])
            change_color(moded, (31, 31, 31), [87, 118, 74])
            change_color(moded, (62, 62, 62), [87, 118, 74])

        else:
            print(f"new color: {background}")

        return moded


def image_resize(image, height, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def cropTo(img):
    _, width = img.shape[:2]
    sideCrop = (width - 224) // 2
    return img[:, sideCrop : (width - sideCrop)]


def resize_to_fit(image, width, height):
    (h, w) = image.shape[:2]
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)

    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image


def change_color(image, old_rgb, new_rgb):
    image[np.where((image == old_rgb).all(axis=2))] = new_rgb
    return image


def rgbvalue(colors):
    return [int(color) for color in colors]
