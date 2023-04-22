# yolo task=detect mode=train model=yolov8l.pt data=data.yaml epochs=100 imgsz=640 save=true device=\'0,1\' batch=16 optimizer='Adam'
# python train.py -data data.yaml -epochs 100 -imgsz 640 -save True -device '0,1' -batch 16 -optimizer Adam

from ultralytics import YOLO
import argparse

model = YOLO('yolov8l.pt')
parser = argparse.ArgumentParser(description="train yolov8")
parser.add_argument("-data", default="data.yaml", type=str)
parser.add_argument("-epochs", default=100, type=int)
parser.add_argument("-imgsz", default=640, type=int)
parser.add_argument("-save", default=True, type=bool)
parser.add_argument("-device", default="0,1", type=str)
parser.add_argument("-batch", default=16, type=int)
parser.add_argument("-optimizer", default="Adam", type=str)

args = parser.parse_args()

results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, save=args.save, device=args.device, batch=args.batch, optimizer=args.optimizer)
