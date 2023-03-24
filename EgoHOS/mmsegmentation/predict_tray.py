import argparse
import cv2
import numpy as np


def detect(img, seed=(960, 540)):
    h, w, _ = img.shape
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

    canny = cv2.Canny(blurred, 20, 35)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Scharr(canny, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Scharr(canny, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    ret, out_img, mask, rect = cv2.floodFill(grad, np.zeros((h+2, w+2), dtype=np.uint8), seed, 255, 7, 7)

    return (rect[0], rect[1]), ((rect[0] + rect[2]), (rect[1] + rect[3]))


def detect_video(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:

            # Display the resulting frame
            (x0, y0), (x1, y1), _ = predict_tray(frame)
            # (x0, y0), (x1, y1) = (500, 250), (1350, 880)
            out_image = cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 7)
            out_image = cv2.resize(out_image, (960, 540))
            print((x0, y0), (x1, y1))
            cv2.imshow("Tray Detection Output", out_image)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def predict_tray(img):
    """
    API for the rest of the program. Returns (x0,y0),(x1,y1) of the bounding box that represents the tray area. And
    a boolean value for the source of the result.
    Args:
        img: the image containing the tray. Preferably without hand/object.
    Return:
        The two coordinate representing the bounding box of the tray.
        A boolean value indicating if the bbox is predicted from image OR guessed (due to recognition failure).
            True represents that the bbox is predicted.
    """
    if img is None:
        return (500, 250), (1350, 880), False
    h, w, _ = img.shape
    seed = (int(w / 2), int(h / 2))
    (x0, y0), (x1, y1) = detect(img, seed=seed)

    if x0 < 390 or x0 > 700 or y0 < 200 or y0 > 450 or x1 < 1150 or x1 > 1450 or y1 < 780 or y1 > 1080:
        # a re-detect is needed
        seed = (seed[0] - 50, seed[1] - 50)
        (x0, y0), (x1, y1) = detect(img, seed=seed)
    else:
        return (x0, y0), (x1, y1), True

    if x0 < 400 or x0 > 700 or y0 < 200 or y0 > 350 or x1 < 1150 or x1 > 1450 or y1 < 780 or y1 > 950:
        # detect fail, wait for next
        (x0, y0), (x1, y1) = (500, 250), (1350, 880)
    else:
        return (x0, y0), (x1, y1), True

    return (x0, y0), (x1, y1), False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--out_path", default="", type=str)
    args = parser.parse_args()

    # Uncomment next line for video, supply with img_path argument
    # detect_video(args.img_path)

    # Detecting in images
    image = cv2.imread(args.img_path)
    coord1, coord2, _ = predict_tray(image)
    out_image = cv2.rectangle(image, coord1, coord2, (0, 0, 255), 7)

    if args.out_path == "":
        cv2.imshow("Tray Detection Output", out_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(args.out_path, out_image)
