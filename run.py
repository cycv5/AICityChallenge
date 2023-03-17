import argparse
import os


def main(img_dir, save_dir, weights):
    for filename in os.listdir(img_dir):
        f = os.path.join(img_dir, filename)
        # checking if it is a file
        if "gitkeep" not in f:
            os.system("python yolov8_tracking/track.py --yolo-weights " + weights
                      + " --tracking-method strongsort --source " + f +
                      " --save-vid --result-dir " + save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--save_dir", default="", type=str)
    parser.add_argument("--weights", type=str)
    args = parser.parse_args()
    main(args.img_dir, args.save_dir, args.weights)
