# Radiology Guardians: Adaptive AI for SIIM COVID-19 Detection

![Alt text](./images/header.png?raw=true "Optional Title")

Radiology Guardians is an end-to-end research workspace for learning, validating, and packaging thoracic imaging detectors tailored to the SIIM-FISABIO-RSNA challenge. The codebase orchestrates classification, segmentation, and localization models while tracking every artifact needed for reproducible experiments.

## Highlights
- Full training recipes covering multi-task classification, lung localization, and opacity detection.
- Seamless pseudo-labeling loops that recycle confident predictions and strengthen supervision.
- Modular data preparation utilities that transform raw DICOM studies into curated model inputs.
- Built-in visual diagnostics, including a demo notebook and exported flowcharts, to accelerate review cycles.

## Environment Setup
The project was originally developed on multi-GPU Linux hosts with CUDA-enabled PyTorch. For local workstations:
- Recommended: Python 3.8+, CUDA 11.x compatible GPUs, 64 GB RAM minimum when generating pseudo-labels.
- Create an isolated environment and install dependencies:
  ```
  conda create -n radiology-guardians python=3.8
  conda activate radiology-guardians
  pip install -r requirements.txt
  ```
- Optional native libraries such as GDCM may be required to read certain DICOM encodings. Install them through your OS package manager before running conversion scripts.

## Data Preparation
1. Obtain the SIIM-FISABIO-RSNA COVID-19 Detection competition data and place the extracted structure under `dataset/siim-covid19-detection`.
2. (Optional) Augment with external studies such as RSNA Pneumonia, VinBigData, NIH ChestXray14, CheXpert, PadChest, or pneumothorax datasets. Store each set within `dataset/external_dataset/<dataset_name>`.
3. Use the preprocessing utilities in `src/prepare` to convert DICOM files to PNG/JPEG, align annotations, and generate stratified folds:
   ```
   cd src/prepare
   python dicom2image_siim.py
   python kfold_split.py
   python prepare_siim_annotation.py
   ```
4. Execute analogous scripts for any external source you enable (for example `dicom2image_pneumonia.py`, `prepare_vinbigdata.py`, `refine_data.py`). When integrating additional public test predictions for pseudo-labeling, copy the derived lung crops into `dataset/lung_crop`.
5. Confirm your folder tree matches the reference outline in `dataset/dataset_structure.txt`.

## Pipeline Overview
![Alt text](./images/flowchart.png?raw=true "Optional Title")

The processing graph illustrates how raw studies feed into dedicated model families (classification, lung detection, opacity detection) before merging into ensemble inference and pseudo-labeling cycles.

## Training Playbooks
### Multi-Head Classification with Segmentation Heads
1. Navigate to `src/classification_aux`.
2. Pretrain backbones on the chosen external sets using `train_chexpert_chest14.sh` and `train_rsnapneu.sh`.
3. Fine-tune on the SIIM challenge data via `train_siim.sh`.
4. Produce refined soft labels and masks using `generate_pseudo_label.sh <checkpoint_dir>`.
5. Iterate pseudo-labeling with `train_pseudo.sh <old_ckpt> <new_ckpt>` until validation metrics plateau.
6. Evaluate single models or ensembles with `evaluate.py --cfg <config_file> --num_tta <count>`.

Classification head performance snapshot (mAP@0.5) across negative, typical, indeterminate, atypical findings:
| Model Variant | No TTA | 8× TTA |
| :-- | :-- | :-- |
| SeResNet152d + UNet decoder | 0.575 | 0.584 |
| EfficientNet-B5 + DeeplabV3+ | 0.583 | 0.592 |
| EfficientNet-B6 + LinkNet | 0.580 | 0.587 |
| EfficientNet-B7 + UNet++ | 0.589 | 0.595 |
| Geometric ensemble | 0.595 | 0.598 |

### Lung Field Localization
1. Switch to `src/detection_lung_yolov5`.
2. Download COCO initialization weights by running `cd weights && bash download_coco_weights.sh`.
3. Train folds through `bash train.sh`.
4. Average precision summary:
   | Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
   | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
   | mAP@0.5:0.95 | 0.921 | 0.931 | 0.926 | 0.923 | 0.922 | 0.925 |
   | mAP@0.5 | 0.997 | 0.998 | 0.997 | 0.996 | 0.998 | 0.997 |

### Opacity Detection
#### YOLOv5x6 @ 768 px
```
cd src/detection_yolov5
cd weights && bash download_coco_weights.sh && cd ..
bash train_rsnapneu.sh
bash train_siim.sh
bash generate_pseudo_label.sh
bash warmup_ext_dataset.sh
bash train_final.sh
```

#### EfficientDet-D7 @ 768 px
```
cd src/detection_efffdet
bash train_rsnapneu.sh
bash train_siim.sh
bash generate_pseudo_label.sh
bash warmup_ext_dataset.sh
bash train_final.sh
```

#### Faster R-CNN FPN (ResNet101d & ResNet200d)
```
cd src/detection_fasterrcnn
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_chexpert_chest14.py --steps 0 1 --cfg configs/resnet200d.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_chexpert_chest14.py --steps 0 1 --cfg configs/resnet101d.yaml
CUDA_VISIBLE_DEVICES=0 python train_rsnapneu.py --cfg configs/resnet200d.yaml
CUDA_VISIBLE_DEVICES=0 python train_rsnapneu.py --cfg configs/resnet101d.yaml
CUDA_VISIBLE_DEVICES=0 python train_siim.py --cfg configs/resnet200d.yaml --folds 0 1 2 3 4 --SEED 123
CUDA_VISIBLE_DEVICES=0 python train_siim.py --cfg configs/resnet101d.yaml --folds 0 1 2 3 4 --SEED 123
CUDA_VISIBLE_DEVICES=0 python warmup_ext_dataset.py --cfg configs/resnet200d.yaml
CUDA_VISIBLE_DEVICES=0 python warmup_ext_dataset.py --cfg configs/resnet101d.yaml
CUDA_VISIBLE_DEVICES=0 python train_final.py --cfg configs/resnet200d.yaml
CUDA_VISIBLE_DEVICES=0 python train_final.py --cfg configs/resnet101d.yaml
```
If you encounter file descriptor limits on some platforms, enable the optional snippet inside the training scripts to raise the open-file threshold.

### Pseudo-Label Filtration
Use `src/detection_make_pseudo` to aggregate predictions across detectors:
```
cd src/detection_make_pseudo
python make_pseudo.py
python make_annotation.py
```
Default thresholds retain studies with negative probability below 0.3 and at least one positive phenotype above 0.7, keeping the two highest-confidence boxes per image. Expect memory usage that peaks around 128 GB when combining every fold and detector.

### Detection Benchmarks
| Detector | mAP@0.5 (TTA) |
| :-- | :-- |
| YOLOv5x6 768 | 0.580 |
| EfficientDet-D7 768 | 0.594 |
| Faster R-CNN ResNet200d 768 | 0.592 |
| Faster R-CNN ResNet101d 1024 | 0.596 |

## Evaluation and Reporting
- `src/demo_notebook/demo.ipynb` visualizes predictions, masks, and box overlays for sanity checks.
- `src/classification_aux/evaluate.py` consolidates per-fold metrics and renders CSV summaries.
- `src/detection_yolov5/test.py`, `src/detection_efffdet/validate.py`, and `src/detection_fasterrcnn/utils.py` provide detector-specific evaluation hooks with optional test-time augmentation.
- Final challenge submissions combined ensemble logits and weighted boxes, yielding 0.658 public leaderboard and 0.635 private leaderboard scores.

## Repository Map
- `dataset/` – raw and processed imagery plus fold metadata.
- `src/prepare/` – ingestion, conversion, and deduplication tools.
- `src/classification_aux/` – hybrid classification/segmentation models, configs, and pseudo-label scripts.
- `src/detection_*` – detectors (YOLOv5, EfficientDet, Faster R-CNN) with training, inference, and warmup utilities.
- `src/detection_make_pseudo/` – pseudo-label curation utilities.
- `images/` – diagrams and marketing assets.

## License
This project is released under the terms stated in `LICENSE`. Please review the file before redistributing models or derived datasets.
