import argparse
import glob
import math
import os.path
import random
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance

slash = "/"

def IOU(boxA, boxB):
    # Calculate the IOU ratio between two bounding boxes
    # boxA = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # intersection area with smoothing factor
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # union area with smoothing factor to avoid division by 0
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the IOU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def checkIOUwithList(bbx, bbx_list, iou_threshold=0.2):
    # check if the current bbx have a higher IOU with any of the pervious bbx
    if len(bbx_list) == 0:
        return True
    for a in bbx_list:
        if IOU(a, bbx) > iou_threshold:
            return False
    return True


def main(bkg, src, target, seg):
    images = glob.glob(src + "/*")
    img_count = 100000
    count = 0
    bkg_img = Image.open(bkg)
    
    # randomize the order of images
    random.shuffle(images)
    iou_threshold=0.1
    num_try = 8
    bbx_l = [] # keep a list of bounding boxes (top_left_x, top_left_y, bottom_right_x, bottom_right_y) in the curr background image
    
    for image_path in tqdm(images, total=len(images)):
        label_path = seg + slash + image_path.split(slash)[-1].split(".jpg")[0] + '_seg.jpg'
        filename = image_path.split(slash)[-1]
        object_class = filename.split("_")[0]
        object_class = object_class.lstrip("0")
        
        im = Image.open(image_path)
        w, h = im.size
        mask_im = Image.open(label_path)
        
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(im)
        brightness_factor = random.uniform(0.9, 1.1)
        im = enhancer.enhance(brightness_factor)
        
        # randomize the position of product images on the ROI of background images
        p = random.randint(-30, 30)
        q = random.randint(-50, 50)
        top_left_x = 50 + (300 * count) + p
        top_left_y = 400 + q
        bottom_right_x = 50 + (300 * count) + p + w
        bottom_right_y = 400 + q + h
        
        bbx = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        
        if count == 0 or checkIOUwithList(bbx, bbx_l):
            bbx_l.append(bbx)
        else:
            for i in range(num_try):
                if checkIOUwithList(bbx, bbx_l):
                    break
                elif bbx[1] > 0:
                    bbx[1] -= 30 # move the object 30 pixels up to reduce IOU as much as possible
                    
        # now deal with the potential outside the coords boundaries
        while bbx[2] > 1920:
            bbx[2] -= 5
            bbx[0] -= 5
            
        while bbx[3] > 1080:
            bbx[3] -= 5
            bbx[1] -= 5
                
        bkg_img.paste(im, (bbx[0], bbx[1]), mask_im) # top left corner (x, y)
        with open(target + slash + "labels" + slash + str(img_count) + ".txt", "a") as f:
            x = (bbx[0] + (w / 2.0)) / 1920.0 # center (x, y)
            y = (bbx[1] + (h / 2.0)) / 1080.0
            width = w / 1920.0
            height = h / 1080.0
            s = "{} {} {} {} {}\n".format(int(object_class)-1, x, y, width, height)
            f.write(s)
            
        count += 1
        if count >= 6:
            bbx_l = []
            bkg_img.save(target + slash + "images" + slash + str(img_count) + ".jpg", quality=95)
            img_count += 1
            count = 0
            bkg_img = Image.open(bkg)
            
            # flip the background image horizontally, vertically, or in both ways at random to increase diversity
            r = random.randint(0, 3) 
            if r == 0:
                bkg_img = ImageOps.flip(bkg_img)
            elif r == 1:
                bkg_img = ImageOps.mirror(bkg_img)  
            elif r == 2:
                bkg_img = ImageOps.flip(ImageOps.mirror(bkg_img))

    if count > 0:
        bkg_img.save(target + slash + "images" + slash + str(img_count) + ".jpg", quality=95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="copy-pasta")
    parser.add_argument("--source_dir", required=True, help="The raw image data folder on disk.")
    parser.add_argument("--background_dir", required=True, help="The background.")
    parser.add_argument("--target_dir", required=True, help="Directory for saving background replaced files.")
    parser.add_argument("--segmentation_dir", required=True,
                        help="Directory where segmentation label images are stored.")
    args = parser.parse_args()
    main(args.background_dir, args.source_dir, args.target_dir, args.segmentation_dir)
