# AI City Challenge 2023 - Track 4 - Team 13
## DACNet: A Deep Automated Checkout Network with Selective Deblurring

## Our software pipeline
![software_stack](./assets/software_stack.jpg)
As shown in our software stack diagram, we have three main modules: preprocessing, training, and postprocessing. In preprocessing, we augmente the synthetic data to bridge the domain gap between the training and test sets. For the training module, our model utilizes a yolov8 network. For postprocessing, we propose novel methods such as retrospective rectification and selective deblurring to further improve the model's accuracy.

## Inference Instructions
The inference can be run with some simple commands. The code is tested on Google Colab.
1. Clone the repo and cd into the root directory:\
  `git clone https://github.com/cycv5/AICityChallenge`\
  `cd AICityChallenge`
2. Run the setup code:\
  `./setup.sh`
3. Place the input videos under `AICityChallenge/data/test/video`, make sure the video is named in the format of `xxx_n.mp4` where n is the video number.
4. Download the pre-trained weight [here](https://drive.google.com/file/d/1NTQMpd_z1neLgDQSPoyiFjPvk7iMNop6/view?usp=share_link), or prepare your own weight file in pt format. To reproduce the leaderboard results precisely, please use the pretrained weight as the training process is non-deterministic (with variable parameters).
5. Place the weight under `AICityChallenge/yolov8_tracking/weights`, or a place of your choice.
6. Run the `run.py` file as following, `img_dir` and `save_dir` are optional, by default they will be `AICityChallenge/data/test/video` and `AICityChallenge/data/test/out`:\
  `python3 run.py --img_dir <dir/containing/videos> --save_dir <dir/containing/output> --weights <dir/to/weights>`
7. Your result.txt will be in `AICityChallenge/data/test/out` or any directory you specify in 6.

## Training Instructions
1. Download the AI City Track 4 raw data and put under `AICityChallenge/yolov8_train/raw_data`
2. Run the data augmentation script `data_gen.py` with the following parameters: source_dir, background_dir, target_dir, segmentation_dir. 

    - `cd AICityChallenge/yolov8_train`

    - `python3 data_gen.py --source_dir raw_data/train --background_dir frame0.jpg --target_dir aug_data --segmentation_dir raw_data/segmentation_labels`
3. Split the test val dataset with the splitfolders tool
    `pip3 install split-folders[full]` and then 
    `splitfolders --ratio .9 .1 --output custom_data aug_data`
4. Now you can run the python train.py script with the input parameters of your choice such as the number of epochs, batch size, GPU devices, learning rate, etc. In addition, you could also try ensemble learning by averaging the confidence score of different models, but we have the best model available in the shared link. Otherwise, you could use the default settings: `python3 train.py`
5. The results will be saved in the `/run` directory.  

**Enjoy** 😄
