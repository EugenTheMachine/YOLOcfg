# YOLOcfg
Contains YOLO config initializer for our ultralytics YOLO models.

## General description
**YOLOcfg** is a repository which automatically sets the main model hyperparams. Some of the values may differ depending on whether a starting model architecture checkpoint is used (directly from the *ultralytics* library) or a customly-obtained model checkpoint with its weights partially pre-trained on the custom data already.

The configs listed in this repository were particularly used for training YOLO11x models for cell segmentation (both L929 and spherical cells). They can be used as a reference point for further project work, but in fact those values were originally set based on theoretical assumptions only, thus any experiments and modifications are welcome.

Please, note that for Cell Spheroid Tracking task the hyperparams were slightly modified, for instance: 1) larger input image size was used (704 instead of 512); 2) model pre-trained on MS COCO dataset was used as a starting point, not raw model architecture with randomly initialized initial weights; 3) smaller batch size was utilized (1 instead of 4) in order to deal with GPU memory limitations.

For a better understanding what each of the hyperparams is used for, and for seeing the full list of available hyperparams, please refer to [ultralytics documentation](https://docs.ultralytics.com/usage/cfg/).

For a raw model architecture, the following values are returned:
```
flipud = 0.5
fliplr = 0.5
pretrained = False
resume = False
epochs = 500
patience = 10
max_det = 5000
data = "/kaggle/input/livecell-raw/dataset.yaml"
batch = 4
imgsz = 512
single_cls = True
split = "val"
plots = False
cache = True
```

For a partially pre-trained model checkpoint, the following values are returned:
```
flipud = 0.5
fliplr = 0.5
pretrained = True
resume = True
epochs = 500
patience = 10
max_det = 5000
data = "/kaggle/input/livecell-raw/dataset.yaml"
batch = 4
imgsz = 512
single_cls = True
split = "val"
plots = False
```

## General model training procedure
Training YOLO models is a huge computationally expensive task, so unless you are a lucky guy who owns at least a 15 GB GPU, you will fail to do it locally. Therefore, you need to use cloud computing services. Most of them do cost money (GCP, AWS, Azure), so our only option are free-of-charge services (Google Colab, Kaggle kernels). We strongly recommend using Kaggle for model training and Google Colab for some other smaller tasks (mostly - quick experiments), as Kaggle kernels provide you with the most preferable cpmputational quotas.

To start using Kaggle, you need register your account and after that verify it through your mobile phone number. **NOTE HERE**: if you have several phone numbers, you can register several accounts and verify them all - in that case, your resulting quota will be several times as large.

Having completed your account verification, create an empty Kaggle kernel (Jupyter notebook on Kaggle platform) and go to the guidelines below.

### Using initial architecture checkpoint
When you first start training your model, you should follow the following steps:
1. Follow steps 1 and 2 from the instruction section below;
2. Choose the appropriate model type and assign its checkpoint name to `MODEL_NAME` variable. We recommend you first try `yolo11x-seg.pt`;
3. Run all the code cells and wait till the model finishes training. **NOTE!!!**: in case model fails to complete training by meeting early stopping criterion in 12 hours (max session duration limit in Kaggle kernels), you may want to continue the model training. For that, complete the remaining steps in this section and go to the next section.
4. Complete steps 3 and 4 from the instruction section below.

### Using partially pre-trained model
In case you need to continue model training having a partially pre-trained model checkpoint, follow the steps below:
1. Upload your latest `last.pt` model checkpoint to your Kaggle models (you can name it whatever you want);
2. Add it to the Kaggle kernel you use for model training;
3. Copy its path and assign it to the `MODEL_NAME` variable;
4. Run all the code cells and wait till the model finishes training. **NOTE!!!**: in case model fails to complete training by meeting early stopping criterion in 12 hours (max session duration limit in Kaggle kernels), you may want to continue the model training. For that, complete the remaining steps in this section and then repeat the steps listed in this section once again.
5. Complete steps 3 and 4 from the instruction section below.

## Instruction
1. Set up virtual environment with P100 GPU accelerator;

2. Create a Kaggle kernel (Jupyter notebook) with the following code cells and run them all:

```bash
!git clone https://github.com/EugenTheMachine/YOLOcfg.git
from YOLOcfg.cfg import setup_env

MODEL_NAME = "yolo11x-seg.pt"
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

3. After the model finished training, download and save the following files:
- *weights/last.pt*: the latest obtained checkpoint - it will be used as a starting point in case of further model training;
- *weights/best.pt*: the best model checkpoint obtained during training, which can be used as a production model;
- *results.csv*: values of loss functions, model quality metrics and optimizer's learning rates values during training;
- *args.yaml*: YAML file with hyperparam values of the model;
- *events.out.tfevents.......*: TensorBoard log file containing information about training (the same as *results.csv*);

4. In the end, turn off the current session (in order not to spoil computational quota).

*Note: added cache*