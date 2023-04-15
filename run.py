import argparse
import os


def main(img_dir, save_dir, weights):
    for filename in os.listdir(img_dir):
        f = os.path.join(img_dir, filename)
        # checking if it is a file
        if "gitkeep" not in f:
            os.system("python yolov8_tracking/track.py --yolo-weights " + weights
                      + " --tracking-method strongsort --source " + f +
                      " --save-vid --imgsz 960 --deblur-path \"/content/TBD/NAFNet/experiments/pretrained_models/NAFNet-REDS-width64.pth\" --result-dir " + save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--img_dir", default="data/test/video", type=str)
    parser.add_argument("--save_dir", default="data/test/out", type=str)
    parser.add_argument("--weights", default="yolov8_tracking/weights/bestl-100.pt", type=str, required=True)
    args = parser.parse_args()
    main(args.img_dir, args.save_dir, args.weights)
