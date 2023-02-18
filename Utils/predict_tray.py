import argparse
import cv2
import numpy as np


def detect(path):
    img = cv2.imread(path)
    h, w, _ = img.shape
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

    canny = cv2.Canny(blurred, 40, 100)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Scharr(canny, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Scharr(canny, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    seed = (int(w/2), int(h/2))

    ret, out_img, mask, rect = cv2.floodFill(grad, np.zeros((h+2,w+2), dtype=np.uint8), seed, 255, 7, 7)

    return (rect[0], rect[1]), ((rect[0] + rect[2]), (rect[1] + rect[3]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--out_path", default="", type=str)
    args = parser.parse_args()
    coord1, coord2 = detect(args.img_path)
    image = cv2.imread(args.img_path)
    out_image = cv2.rectangle(image, coord1, coord2, (0, 0, 255), 7)

    if args.out_path == "":
        cv2.imshow("Tray Detection Output", out_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(args.out_path, out_image)
