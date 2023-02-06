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

echo "Preprocessing the videos"
python predict_image.py \
       --config_file ./work_dirs/seg_twohands_ccda/seg_twohands_ccda.py \
       --checkpoint_file ./work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth \
       --video_dir ../../data/test/video \
       --out_dir ../../data/test/out