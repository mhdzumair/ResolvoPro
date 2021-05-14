from captcha_solver import CaptchaSolver
from flask import Flask, request, jsonify
import cv2
import numpy
from tempfile import NamedTemporaryFile
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
num = 1
solver = CaptchaSolver()


@app.route("/")
def success():
    return jsonify("success")


@app.route("/solve", methods=["POST"])
def solve():
    file = request.files["file"]
    if file:
        npimg = numpy.fromfile(file, numpy.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        prediction = solver.predict_captcha(image)
        result = 0
        if prediction == "star":
            result = solver.solve_star(gray)
        else:
            result = solver.solve_number(gray)

        if result > 8 or result < 0:
            result = 0

        return jsonify({"predict": prediction, "data": str(result)})
    return jsonify({"data": "image not found"})


if __name__ == "__main__":
    app.run()
