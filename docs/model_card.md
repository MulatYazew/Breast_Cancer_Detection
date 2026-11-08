# Model card

Research use only. Not a medical device. Not validated for diagnosis or patient management.

## Detection

- Algorithm: YOLO26n (Ultralytics), initialized from `yolo26n.pt`.
- Data: BUSI tumor boxes derived from lesion masks. One class, `tumor`.
- Image size 640, 150 epochs configured, early-stopping patience 20.
- Best validation epoch in `checkpoints/yolo/busi_tumor/results.csv`: epoch 86, mAP@0.50 0.907, mAP@0.50:0.95 0.597. This is the training log, not the test set.
- Test log 2026-10-07: precision 0.884, recall 0.770, F1 0.823, mAP@0.50 0.734, mAP@0.50:0.95 0.545.
- The test set shares near-duplicate frames with training. See `docs/dataset.md`.

## Segmentation

- Algorithm: U-Net, ImageNet ResNet34 encoder, 3 input channels, 1 output channel.
- Loss: 0.5 Dice + 0.5 BCE. Focal weight 0.
- Test log 2026-10-07: Dice 0.936, IoU 0.882, precision 0.927, recall 0.949, specificity 0.918.
- HD95 and ASSD were not computed for that run.

## Classification

- Algorithm: ImageNet ResNet50, 2-class head, 5 epochs with the backbone frozen, then fine-tuning at 1e-5.
- Input that was actually trained: masked ROI (`roi_masked` in `resnet/input_modes.py`). Ground-truth boxes and masks were used for the isolated test log. The cascade log uses predicted boxes and masks.
- Label: BUSI folder name. Not a BI-RADS category. A pathology table is not in the repository.
- Isolated test log 2026-10-07: accuracy 0.93, precision 0.962, malignant recall 0.806, F1 0.877, ROC-AUC 0.984, PR-AUC 0.968.
- Cascade test log the same day: accuracy 0.830, precision 0.688, recall 0.815, F1 0.746, ROC-AUC 0.892, PR-AUC 0.752, 9 detection failures, mean detection IoU 0.670, mean segmentation Dice 0.807.
- Calibration, Brier score, and a confidence interval were not in that log. Threshold selection on a validation operating point was not re-done in the 2026-11-08 session.
- DenseNet121, EfficientNet-B0, and ConvNeXt-Tiny can be constructed. They have no BUSI checkpoint. Full-image, unmasked-ROI, and four-channel ROI+mask modes are implemented and untrained. No architecture is selected.

## Descriptor and assessment models

Not trained. Required labels are absent.

## Language model

Configured id: `Qwen/Qwen2.5-3B-Instruct` (Qwen research license, non-commercial). It was not re-run on 2026-11-08.

`google/medgemma-27b-text-it` and `google/medgemma-1.5-4b-it` were not loaded. MedGemma is gated. The 27B checkpoint is refused below 48 GiB RAM. This machine has 24 GiB and MPS, and no CUDA device.

## Known limits

Small single-source set, class imbalance (210 malignant / 437 benign images), no patient ids, cross-split near-duplicates, no external test set, no reader study, no descriptor labels, pixel units only.
