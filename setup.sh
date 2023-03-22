echo "Setting up the environment"
cd EgoHOS
pip install -r requirements.txt
pip install -U openmim
mim install mmcv-full==1.6.0
cd mmsegmentation
pip install -v -e .

echo "Downloading weights"
curl -L -s -o work_dirs.zip "https://drive.google.com/uc?id=1LNMQ6TGf1QaCjMgTExPzl7lFFs-yZyqX&confirm=t"
unzip work_dirs.zip
rm work_dirs.zip

echo "Setting up deblur"
cd ../../NAFNet
pip install -r requirements.txt
pip install --upgrade --no-cache-dir gdown
python3 setup.py develop --no_cuda_ext

echo "Setting up YOLOv8"
cd ../../yolov8_tracking
pip install -r requirements.txt