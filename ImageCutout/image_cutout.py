import os
import argparse
import cv2
import random
import numpy
import tqdm
from tqdm import tqdm


def apply_mask(img, h_mask, w_mask, mask_size):
    left_right = random.randint(0, 1)
    if left_right == 0:
        img[h_mask:h_mask+mask_size, mask_size:2*mask_size, :] = 0
    else:
        img[h_mask:h_mask + mask_size, -1-2*mask_size:-1-mask_size, :] = 0
    top_bot = random.randint(0, 1)
    if top_bot == 0:
        img[mask_size:2*mask_size, w_mask:w_mask + mask_size, :] = 0
    else:
        img[-1-2*mask_size:-1-mask_size, w_mask:w_mask + mask_size, :] = 0
    return img


def main(path, out_path, mask_size_multiplier):
    if out_path == "":
        out_path = path
    count = 0
    with tqdm(total=100) as pbar:
        for filename in sorted(os.listdir(path)):
            f_path = os.path.join(args.img_path, filename)
            img = cv2.imread(f_path)
            height, width, c = img.shape
            mask_size = (min(height, width)) * mask_size_multiplier
            h_mask = random.randint(0, height-mask_size-1)
            w_mask = random.randint(0, width-mask_size-1)
            new_img = apply_mask(img, h_mask, w_mask, mask_size)
            save_path = os.path.join(out_path, filename[:-4] + "_cutout.jpg")
            cv2.imwrite(save_path, new_img)
            count += 1
            if count % 1065 == 0:
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--out_path", default="", type=str)
    parser.add_argument("--mask_size_multiplier", default=0.2, type=int)
    args = parser.parse_args()
    main(args.img_path, args.out_path, args.mask_size_multiplier)
