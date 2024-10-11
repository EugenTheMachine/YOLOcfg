# YOLOcfg
Contains YOLO config initializer for our ultralytics YOLO models.

## Instruction
Create a Kaggle kernel (Jupyter notebook) with the following code cells:

```bash
!git clone https://github.com/EugenTheMachine/YOLOcfg.git
from YOLOcfg.cfg import setup_env

MODEL_NAME = "yolov8m.pt"
config = setup_env(MODEL_NAME)
```

```bash
EPOCHS = config['epochs']
PATIENCE = config['patience']
MAX_DET = config['max_det']
DATA = config['data']
PRETRAINED = config['pretrained']
SINGLE_CLS = config['single_cls']
SPLIT = config['split']
PLOTS = config['plots']
FLIPUD = config['flipud']
FLIPLR = config['fliplr']
BATCH = config['batch']
IMGSZ = config['imgsz']
RESUME = config['resume']
# MODEL_NAME was defined at the beginning

print(config)
```

```bash
from ultralytics import YOLO

model = YOLO(MODEL_NAME)
results = model.train(epochs=EPOCHS, patience=PATIENCE, max_det=MAX_DET,
                      data=DATA, pretrained=PRETRAINED, flipud=FLIPUD,
                      fliplr=FLIPLR, batch=BATCH, imgsz=IMGSZ, resume=RESUME,
                     single_cls=SINGLE_CLS, split=SPLIT, plots=PLOTS)
```