from os import listdir

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy
import uvicorn

from captcha_solver import CaptchaSolver, Preprocessor

app = FastAPI()
solver = CaptchaSolver()
preprocessor = Preprocessor()
error_count = len(listdir("captcha_error/")) + 1

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def solve_captcha(image, bgr=False):
    global error_count
    if bgr:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    prediction = solver.predict_captcha(image)
    result = 0
    if prediction == "star":
        result = solver.solve_star(gray)
    else:
        result = solver.solve_number(gray)

    if result > 8 or result < 0:
        result = 0

    if result == 0:
        cv2.imwrite("captcha_error/" + str(error_count).zfill(3) + ".png", image)
        error_count += 1
    return {"predict": prediction, "data": str(result)}


@app.post("/solver/argonclick")
async def solve_argonclick(file: UploadFile = File(...)):
    npimg = numpy.frombuffer(await file.read(), numpy.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocessor.argonclick(image)
    if isinstance(image, numpy.ndarray):
        return await solve_captcha(image)
    return {"predict": "New Background Color", "data": str(0)}
    


@app.post("/solve")
async def solve_general(file: UploadFile = File(...)):
    npimg = numpy.frombuffer(await file.read(), numpy.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return await solve_captcha(image, True)


if __name__ == "__main__":
    uvicorn.run("fastapp:app", port=5000)
