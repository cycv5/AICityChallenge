# Ultralytics YOLO 🚀, GPL-3.0 license

import glob
import math
import os
import time
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils import LOGGER, ROOT, is_colab, is_kaggle, ops
from ultralytics.yolo.utils.checks import check_requirements

# For Colab
import sys
base_path = Path(__file__).parent.resolve()
root_path = (base_path / "../../../../../../").resolve()
print("Root directory for project: " + str(root_path))
sys.path.append(str(root_path))

import EgoHOS.mmsegmentation.predict_image as handseg
from EgoHOS.mmsegmentation.predict_tray import predict_tray


class LoadStreams:
    # YOLOv8 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='file.streams', imgsz=640, stride=32, auto=True, transforms=None, vid_stride=1):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.imgsz = imgsz
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [ops.clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/Zgi9g1ksQHc'
                check_requirements(('pafy', 'youtube_dl==2020.12.2'))
                import pafy  # noqa
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0:
                assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
                assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([LetterBox(imgsz, auto, stride=stride)(image=x).shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f = 0, self.frames[i]  # frame number, frame array
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([LetterBox(self.imgsz, self.auto, stride=self.stride)(image=x) for x in im0])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadScreenshots:
    # YOLOv8 screenshot dataloader, i.e. `python detect.py --source "screen 0 100 100 512 256"`
    def __init__(self, source, imgsz=640, stride=32, auto=True, transforms=None):
        # source = [screen_number left top width height] (pixels)
        check_requirements('mss')
        import mss  # noqa

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.imgsz = imgsz
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        return self

    def __next__(self):
        # mss screen capture: get raw pixels from the screen as np array
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]  # [:, :, :3] BGRA to BGR
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=im0)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        self.frame += 1
        return str(self.screen), im, im0, None, s  # screen, img, original img, im0s, s


class LoadImages:
    # YOLOv8 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, imgsz=640, stride=32, auto=True, transforms=None, vid_stride=1, deblur_path=""):
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.imgsz = imgsz
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        # added attributes
        self.tray = ((500, 250), (1350, 880))
        self.first_found = False
        self.counter = 0
        base_path = Path(__file__).parent.resolve()
        opt_path = (base_path / '../../../../../../NAFNet/options/test/REDS/NAFNet-width64.yml').resolve()
        opt = parse(str(opt_path), is_train=False)
        opt['dist'] = False
        opt['path']['pretrain_network_g'] = deblur_path
        print(opt)
        self.NAFNet = create_model(opt)
        self.deblur = False
        work_path = (base_path / "../../../../../../EgoHOS/mmsegmentation/work_dirs").resolve()
        config_path = (work_path / "seg_twohands_ccda/seg_twohands_ccda.py").resolve()
        checkpt_path = (work_path / "seg_twohands_ccda/best_mIoU_iter_56000.pth").resolve()
        self.h = handseg.HandSegmentor(str(config_path), str(checkpt_path))
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def set_deblur(self, value):
        self.deblur = value

    def imread(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def img2tensor(self, img, bgr2rgb=False, float32=True):
        img = img.astype(np.float32) / 255.
        return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

    # def display(img1, img2):
    #     fig = plt.figure(figsize=(25, 10))
    #     ax1 = fig.add_subplot(1, 2, 1)
    #     plt.title('Input image', fontsize=16)
    #     ax1.axis('off')
    #     ax2 = fig.add_subplot(1, 2, 2)
    #     plt.title('NAFNet output', fontsize=16)
    #     ax2.axis('off')
    #     ax1.imshow(img1)
    #     ax2.imshow(img2)

    def single_image_inference(self, model, img):
        model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if model.opt['val'].get('grids', False):
            model.grids()

        model.test()

        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        return sr_img

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()

            # TODO: Image Preprocess to im0 Here!!! #####
            # TODO: after hand segmentation, determine if predict tray should kick in
            # If current tray prediction is based on guesses, predict_tray must kick in
            # Keep running predict_tray until we find a frame where hand/object is not covering the tray AND
            # the prediction result is not a guess.
            # Every 5 second (tentative) after, refresh the tray prediction if hand/object is not covering the tray.
            def hand_on_tray(img):
                ret = False
                for i in range(250, 880):
                    for j in range(500, 1350):
                        if (img[i][j] == np.array([0, 0, 0])).all():
                            ret = True
                return ret
            if self.deblur and im0 is not None:
                print("Deblurring at work")
                inp = self.img2tensor(im0)
                im0 = self.single_image_inference(self.NAFNet, inp)
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                # do deblur work

            if im0 is not None:
                im1 = self.h.process_video_frame(im0).astype('float32')

            print("first_found: ", self.first_found)
            if not self.first_found:
                coord1, coord2, pred = predict_tray(im0)
                # if (not hand_on_tray(im1)) and pred:
                if pred:  # not a guess
                    self.first_found = True
                self.tray = (coord1, coord2)
            else:
                if self.counter == 300:
                    coord1, coord2, pred = predict_tray(im0)
                    if pred:
                        self.tray = (coord1, coord2)

            if self.counter >= 300:
                self.counter = 0
            else:
                self.counter += 1
            print("Current tray: ", self.tray)
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(im1)  # transforms
        else:
            im = LetterBox(self.imgsz, self.auto, stride=self.stride)(image=im1)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s, self.tray, self.deblur

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files


class LoadPilAndNumpy:

    def __init__(self, im0, imgsz=640, stride=32, auto=True, transforms=None):
        if not isinstance(im0, list):
            im0 = [im0]
        self.im0 = [self._single_check(im) for im in im0]
        self.imgsz = imgsz
        self.stride = stride
        self.auto = auto
        self.transforms = transforms
        self.mode = 'image'
        # generate fake paths
        self.paths = [f"image{i}.jpg" for i in range(len(self.im0))]

    @staticmethod
    def _single_check(im):
        assert isinstance(im, (Image.Image, np.ndarray)), f"Expected PIL/np.ndarray image type, but got {type(im)}"
        if isinstance(im, Image.Image):
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def _single_preprocess(self, im, auto):
        if self.transforms:
            im = self.transforms(im)  # transforms
        else:
            im = LetterBox(self.imgsz, auto=auto, stride=self.stride)(image=im)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def __len__(self):
        return len(self.im0)

    def __next__(self):
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        auto = all(x.shape == self.im0[0].shape for x in self.im0) and self.auto
        im = [self._single_preprocess(im, auto) for im in self.im0]
        im = np.stack(im, 0) if len(im) > 1 else im[0][None]
        self.count += 1
        return self.paths, im, self.im0, None, ''

    def __iter__(self):
        self.count = 0
        return self


if __name__ == "__main__":
    img = cv2.imread(str(ROOT / "assets/bus.jpg"))
    dataset = LoadPilAndNumpy(im0=img)
    for d in dataset:
        print(d[0])
