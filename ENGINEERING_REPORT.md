# Engineering report

Session date: 2026-11-08. Research use only.

## Notebook execution, 2026-11-08

- Path: `notebooks/Breast_Ultrasound_AI_End_to_End.ipynb`.
- Project-generated notebooks: 1.
- The notebook was executed from a fresh kernel with `jupyter nbconvert --execute --inplace`. No cell raised an error.
- Device: MPS. CUDA was not available.
- Checkpoints loaded: `checkpoints/yolo/busi_tumor/weights/best.pt` (5,384,133 bytes), `checkpoints/unet/best_unet.pth` (97,916,747 bytes), `checkpoints/resnet/best_resnet.pth` (94,363,799 bytes).
- YOLO26n, U-Net, and ResNet50 each ran on `benign (100).png` and on 12 test-split images (6 benign, 6 malignant). Training stayed off because `TRAIN_MODE` is false.
- On those 12 images, YOLO26n precision and recall were both 0.917 at IoU 0.5 (11 true positives, 1 false positive, 1 false negative). U-Net mean Dice was 0.921 and mean IoU was 0.856, using the ground-truth ROI. ResNet50 malignant sensitivity was 0.833, benign specificity was 1.0, ROC-AUC was 0.972, and the confusion matrix was [[6, 0], [1, 5]]. These are sample counts, not a full-test or clinical result.
- RAG ran. The two PDFs were embedded with `sentence-transformers/all-MiniLM-L6-v2` and searched with FAISS. Recall@5 on the 10 prepared questions was 0.60.
- TinyLlama 1.1B (1,100,048,384 parameters) ran. Process RSS while it was loaded was 2.47 GiB. Its text did not contain a findings or impression JSON object, so the report fields stayed empty and the limitation states that the model output was not used.
- MedGemma 27B was refused. MedGemma 1.5 4B was refused without download; the published shards are 8.60 GB.
- The end-to-end draft validated against `StructuredStudyReport`. `results/notebook_end_to_end/draft_report.json` was written. The approved export was withheld because physician review was still pending.
- Samples taken after each model was released peaked at 1.38 GiB. The highest RSS measured with a model resident was the 2.47 GiB TinyLlama reading.

## Earlier audit in this session

This session audited the existing BUSI cascade and added the pieces that the files can support. It did not retrain YOLO26n, U-Net, or ResNet50, and it did not declare a new winning classifier.

## What was already in the repository

- YOLO26n, U-Net (ResNet34), and ResNet50 training and evaluation code, with checkpoints and an October 2026 test log.
- A Streamlit reviewer, radiomics export, BM25 retrieval, and a heuristic lexicon that is hidden unless a config flag is turned on.
- A 152-cell notebook, `notebooks/Breast_CAD_Master.ipynb`.

Those paths were kept. `python train.py` still trains the same three stages.

## What this session changed

- `config.yaml` now indexes `dataset/BIRADS Reference Card 1.pdf` and `dataset/usg-16034 (1).pdf`. The second card copy under `dataset/BUSI_Jpeg/` is documented and not indexed again.
- `bus_research/rag.py` stores a section label and, for the Lee article, the printed page (PDF page + 70).
- Dataset audit module `dataset/inspect.py`, writing `reports/dataset_audit.json`, `reports/dataset_audit.md`, and `reports/figures/class_distribution.png`.
- Review state machine, structured report schema, mask geometry, surface-distance functions, classification input modes, and untrained backbone constructors.
- FastAPI service in `api/` (not a package named `app`, because `app.py` is the Streamlit entry point).
- MedGemma load guard. 27B is refused under 48 GiB. No fallback model is called.
- Tests `tests/test_research_gates.py` and `tests/test_rag_documents.py`.
- Notebook `notebooks/Breast_Ultrasound_AI_End_to_End.ipynb`.
- Docs under `docs/` and this audit in `PROJECT_AUDIT.md`.

## Dataset limits, measured

647 RGB PNGs, 437 benign / 210 malignant, 665 masks, 0 unreadable files. No patient id. No BI-RADS table. Exact duplicate: benign (433) with malignant (145). 128 near-duplicate pairs, 63 of them across the saved split, 5 across labels.

## Models

| Model | Configuration in the checkpoint or YAML | This session |
| --- | --- | --- |
| YOLO26n | imgsz 640, class `tumor` | Checkpoint file present (5.4 MB). Test metrics taken from the 2026-10-07 log |
| U-Net | ResNet34, Dice 0.5 + BCE 0.5 | State dict loads. Keys: `config`, `model_state` |
| ResNet50 | ImageNet init, 5 frozen epochs, masked ROI | State dict loads. Other backbones and input modes were constructed in a unit test and not trained |

Logged test metrics are copied in `reports/logged_baseline_metrics.json`. They were not recomputed.

## Retrieval, measured this session

10 questions, k = 5, BM25, phrase match against the expected source: recall@5 = 1.00, MRR = 0.608, nDCG@5 = 0.706, precision@5 = 0.28. File: `reports/rag_retrieval.json`.

## Tests executed

`pytest tests/` → **69 passed**, about 6.7 seconds, on Python 3.12. The API test uses a fake detector. The PDF test reads the two reference files. GPU training was not part of the test run.

After a later section-label tweak, `pytest tests/test_rag_documents.py tests/test_research_gates.py` → **9 passed**.

## Hardware

arm64, 10 CPUs, 24 GiB RAM, PyTorch 2.12.1, CUDA unavailable, MPS available. MedGemma 27B was not loaded. MedGemma 4B was not downloaded. Qwen was not run.

## Incomplete, on purpose

- No new training, so no claim that DenseNet, EfficientNet, ConvNeXt, or a different ROI input beats ResNet50.
- No HD95 or ASSD on the test set.
- No descriptor head and no assessment head.
- No patient-level bootstrap interval.
- The Streamlit UI was not rewritten around the new state machine. The API uses the state machine. Streamlit still has its own accept/reject controls.
- The saved BM25 index in `results/rag/` was not rebuilt. Call `bus_research.rag_eval.build_project_index` to refresh it.
- Full `CADPipeline` inference on a BUSI image was not re-run. Checkpoint tensors were loaded.

## How to run

Notebook, inference only:

```bash
jupyter lab notebooks/Breast_Ultrasound_AI_End_to_End.ipynb
```

API:

```bash
export REVIEWER_TOKEN=local-research-token
uvicorn api.main:app --reload
```

Streamlit:

```bash
streamlit run app.py
```

Existing training:

```bash
python train.py yolo
python train.py unet
python train.py resnet
```

## Clinical limit

Nothing in this repository is cleared to read a clinical study, assign BI-RADS, or recommend treatment. A qualified physician remains the only person who can approve a finding or enter an assessment.
