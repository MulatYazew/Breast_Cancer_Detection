# Hardware audit

## End-to-end notebook, 2026-11-08

Measured while running `notebooks/Breast_Ultrasound_AI_End_to_End.ipynb` on this MacBook Air.

- Device: Apple Silicon, MPS. CUDA: false. PyTorch 2.12.1. RAM 24.00 GiB.
- YOLO26n, U-Net, and ResNet50 were loaded one at a time from the checkpoints listed in `results/notebook_end_to_end/execution_summary.json`.
- TinyLlama 1.1B was loaded from the local Hugging Face cache. Parameter count: 1,100,048,384. Process RSS while it was loaded: 2.47 GiB.
- RSS samples taken after each model had been released peaked at 1.38 GiB.
- `google/medgemma-27b-text-it` was refused and not downloaded.
- `google/medgemma-1.5-4b-it` was refused and not downloaded. Published weight files are 4.96 GB and 3.64 GB, 8.60 GB together.

## Earlier classifier run

Measured on this machine during the removed `BREAST_CANCER_AI.ipynb` classifier run. That notebook is no longer in the project.

- Architecture: Apple Silicon. CUDA available: False. MPS available: True.
- RAM: 24.00 GiB. CPUs: 10.
- PyTorch: 2.12.1.
- Free disk at startup: 168.8 GiB.
- Optional checkpoint cap: 8 GB. Process RSS cap: 18 GiB.

## Models that were not loaded

- `google/medgemma-27b-text-it`: refused. A 27B checkpoint is outside a 24 GB laptop.
- `google/medgemma-1.5-4b-it`: refused without download. Published shards are 4.96 GB and 3.64 GB, summing to 8.60 GB.

## Training peak memory

| Backbone | Peak RSS GiB | Epochs | Batch |
| --- | --- | --- | --- |
| efficientnet_b0 | 1.05 | 5 | 8 |
| densenet121 | 1.28 | 6 | 8 |
| mobilenet_v3_small | 1.56 | 6 | 8 |
| resnet18 | 1.65 | 6 | 8 |

## Selected checkpoint

Selected backbone: `densenet121`.

| Measurement | Value |
| --- | --- |
| Selected checkpoint size | 0.028 GB |
| RSS before reload | 1.48 GiB |
| RSS after load | 1.48 GiB |
| RSS after 20 inferences | 1.51 GiB |
| Seconds per image | 0.0226 |
| Under 8 GB checkpoint | True |
| Under 18 GiB process | True |
