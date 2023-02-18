A simple tool for creating image cutout.

Run by command:
python image_cutout.py --img_path <path-to-image-folder> --out_path <path-to-output-folder> --mask_size_multiplier <number>

If no output path is specified, it will go into the img_path folder. The new file names will be appended with "_cutout" at the end.

The "mask_size_multiplier" controls the size of the mask. Mask_size = minimum(height, width) * mask_size_multiplier.
The default value is 0.2. That means the side length of the mask is 1/5 of the height or width, whichever is smaller.