import cv2


def solve_star(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 28, 47, L2gradient=True)
    dilated = cv2.dilate(canny, (2, 2), iterations=5)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


if __name__ == '__main__':
    import time
    start = time.perf_counter()
    solve_star("stars/captcha1.png")
    print("time taken: ", time.perf_counter() - start)
