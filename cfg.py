import os
import sys
import shutil
from pathlib import Path
import subprocess
import zipfile

MODEL_ARCHITECTURES = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolov8n-seg",
    "yolov8s-seg",
    "yolov8m-seg",
    "yolov8l-seg",
    "yolov8x-seg",
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov9c",
    "yolov9e",
    "yolov9t-seg",
    "yolov9s-seg",
    "yolov9m-seg",
    "yolov9c-seg",
    "yolov9e-seg",
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10b",
    "yolov10l",
    "yolov10x",
    "yolov10n-seg",
    "yolov10s-seg",
    "yolov10m-seg",
    "yolov10l-seg",
    "yolov10x-seg",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
    "yolo11n-seg",
    "yolo11s-seg",
    "yolo11m-seg",
    "yolo11l-seg",
    "yolo11x-seg"
]

def setup_env(model_name: str, data_dir: str = "/kaggle/working/data"):
    """Main function of the module. Sets up the whole environment."""
    install_ultralytics()
    # import_data()
    setup_kaggle()
    return init_cfg(data_dir, model_name)

def install_ultralytics():
    """Installs ultralytics if it is not already installed."""
    try:
        import ultralytics
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

# def import_data():
#     """Loads dataset and prepares it"""
#     subprocess.run(["wget", "--no-check-certificate", "https://drive.google.com/uc?export=download&id=1ElVOmuWiAS1bOVgXvH7iNKzRKrkS_duk", "-O", "datasets.zip"], check=True)
#     with zipfile.ZipFile("/kaggle/working/datasets.zip", 'r') as zip_ref:
#         zip_ref.extractall()

def setup_kaggle():
    """Sets up kaggle environment."""
    os.environ['WANDB_MODE'] = 'disabled'
    # shutil.move('/kaggle/working/YOLOcfg/dataset.yaml', '/kaggle/working/dataset.yaml')

def check_model_name(model_name: str):
    """Checks whether model of correct type was specified."""
    assert model_name[-3:] == ".pt", "Model has incorrect extension"
    assert "pose" not in model_name, "Model of incorrect task type was specified"

def init_cfg(data_dir: str, model_name: str):
    """
    Initializes default configs for the YOLO model.
    
    Input args:
    - data_dir (str): path to dataset directory in which train/val/test folders are stored;
    - model_name (str): name of the YOLO model to use.
    """
    # setting up configs for data augmentation
    flipud=0.5
    fliplr=0.5
    
    # checking model name and setting corresponding model configs
    check_model_name(model_name)
    if Path(model_name).stem in MODEL_ARCHITECTURES:
        print("Init architecture is used. Not a checkpoint...")
        pretrained = False
        resume=False
        epochs=500
    else:
        print("Pretrained model checkpoint is used...")
        pretrained = True
        resume=True
        epochs=500
    
    patience = 10
    max_det = 5000
    data = "/kaggle/input/livecell-raw/dataset.yaml"
    batch = 4
    imgsz = 512
    single_cls = True
    split = "val"
    plots = False

    cfg = {
        "epochs": epochs,
        "patience": patience,
        "max_det": max_det,
        "data": data,
        "pretrained": pretrained,
        "single_cls": single_cls,
        "split": split,
        "plots": plots,
        "flipud": flipud,
        "fliplr": fliplr,
        "batch": batch,
        "imgsz": imgsz,
        "resume": resume
    }

    return cfg