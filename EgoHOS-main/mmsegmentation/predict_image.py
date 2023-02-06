from mmseg.apis import inference_segmentor, init_segmentor
import os
import argparse
from PIL import Image
import numpy as np
import cv2
import sys


parser = argparse.ArgumentParser(description="")
parser.add_argument("--config_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K.py', type=str)
parser.add_argument("--checkpoint_file", default='./work_dirs/upernet_swin_base_patch4_window12_512x512_160k_egohos_handobj2_pretrain_480x360_22K/best_mIoU_iter_42000.pth', type=str)
parser.add_argument("--video_dir", default='../../data/test/video', type=str)
parser.add_argument("--out_dir", default='../../data/test/out', type=str)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# build the model from a config file and a checkpoint file
model = init_segmentor(args.config_file, args.checkpoint_file, device='cuda:0')

for filename in os.listdir(args.video_dir):
    f_path = os.path.join(args.video_dir, filename)
    vidcap = cv2.VideoCapture(f_path)
    width, height, fps = int(vidcap.get(3)), int(vidcap.get(4)), vidcap.get(5)
    out = cv2.VideoWriter(os.path.join(args.out_dir, "{}_masked.avi".format(filename)), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    success, image = vidcap.read()
    count = 1

    alpha = 0.5
    while success:
        seg_result = inference_segmentor(model, image)[0]
        inv_seg_result = np.where(seg_result == 0, 1, 0)
        masked_image = (image.transpose() * inv_seg_result.transpose()).transpose()
        out.write(masked_image.astype(np.uint8))
        # imsave(os.path.join("/content/output/", str(count) + '.png'), seg_result.astype(np.uint8))
        if count % 60 == 0:
            print("{} second of video {} processed.".format(count//60, filename))
        count += 1
        success, image = vidcap.read()
    vidcap.release()
    out.release()
