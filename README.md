# AI City Challenge 2023 - Track 4 - Team 13
## The code repository is still *under construction*. Available Apr 21, 2023
## Inference Instructions
The inference can be run with some simple commands. The code is tested on Google Colab.
1. Clone the repo and cd into the root directory:\
  `git clone https://github.com/cycv5/AICityChallenge`\
  `cd AICityChallenge`
2. Run the setup code:\
  `./setup.sh`
3. Place the input videos under `AICityChallenge/data/test/video`, make sure the video is named in the format of `xxx_n.mp4` where n is the video number.
4. Download the pre-trained weight [here](https://drive.google.com/file/d/1NTQMpd_z1neLgDQSPoyiFjPvk7iMNop6/view?usp=share_link), or prepare your own weight file in pt format.
5. Place the weight under `AICityChallenge/yolov8_tracking/weights`, or a place of your choice.
6. Run the `run.py` file as following, `img_dir` and `save_dir` are optional, by default they will be `AICityChallenge/data/test/video` and `AICityChallenge/data/test/out`:\
  `python3 run.py --img_dir <dir/containing/videos> --save_dir <dir/containing/output> --weights <dir/to/weights>`
7. Your result.txt will be in `AICityChallenge/data/test/out` or any directory you specify in 6.

**Enjoy** 😄
