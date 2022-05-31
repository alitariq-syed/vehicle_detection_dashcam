
import os
os.chdir("G:\Motive AI Challenge dataset\Motive AI Challenge\mmdetection-master")
os.listdir()


# import required functions, classes
from sahi.model import MmdetDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

# import os
# os.getcwd()

# model_path="/content/drive/MyDrive/Motive AI Challenge/tutorial_swin/kaggle_output/epoch_60.pth"
# config_path="/content/drive/MyDrive/Motive AI Challenge/faster_rcnn_swin-t-p4-w7_fpn_1x_motive_challenge_fp16_mscale_fullCFG.py"
model_path="tutorial_swin/kaggle_output/epoch_60.pth"
config_path="configs/swin/faster_rcnn_swin-t-p4-w7_fpn_1x_motive_challenge_fp16_mscale_fullCFG.py"

model_type = "mmdet"
model_path = model_path
model_config_path = config_path
model_device = "cuda" # cpu or 'cuda:0'
model_confidence_threshold = 0.4

slice_height = 180
slice_width = 320
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

# source_image_dir = "data/train/train_images" #"demo_data/"
# source_image_dir = "/content/vehicle_detection_dashcam/mmdetection-master/data/train/train_images"
#source_image_dir = "/content/vehicle_detection_dashcam/mmdetection-master/data/public_test/test2_images"
source_image_dir = "data/public_test/test2_images"

predict(
    model_type=model_type,
    model_path=model_path,
    model_config_path=config_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    
    dataset_json_path = "data/public_test/test_gt.json",#"/content/vehicle_detection_dashcam/mmdetection-master/data/public_test/test_gt.json", #"data/train/train_gt.json",
    verbose=0,
    export_pickle=False,
    novisual=True,
    project="/content/drive/MyDrive/Motive AI Challenge/tutorial_swin/kaggle_output/runs/predict",
    name= "exp")
