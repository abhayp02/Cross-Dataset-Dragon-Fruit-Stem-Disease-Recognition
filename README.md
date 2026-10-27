# Dragon Fruit Stem Analysis

A local research dashboard using your saved MobileNetV3 Small, EfficientNet-B0,
and ResNet18 checkpoints. Training code and model files are not modified by the app.

## Start the app

Double-click `start_app.bat`, keep its terminal open, then visit
http://127.0.0.1:5000 in your browser. Or run from this folder:

```powershell
.\.venv\Scripts\python.exe app.py
```

Stop the server with Ctrl+C in its terminal. No GPU or internet connection is
needed after setup. The server listens only on this laptop's loopback address.

## First-time setup on another laptop

Use Python 3.10 (tested) or 3.11/3.12. Create a new environment on that laptop;
do not copy `.venv` from your friend's device.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
```

The application environment is separate from the GPU training dependencies in
`requirements-training.txt`. The CPU app uses torch 2.6 and torchvision 0.21.

## Features

- **Predict:** select an architecture and experiment, choose or drop a JPG/PNG/WebP
  image, preview it, and run the actual checkpoint. All class scores are shown.
- **Model Comparison:** choose one experiment to compare the three architectures,
  inspect per-class metrics and confusion matrices, or download the displayed
  comparison as CSV (metrics in CSV are fractions from 0 to 1).
- **About:** CNN workflow, datasets, scope, and privacy details.

Default prediction uses MobileNetV3 Small, Dataset C, with six classes. This is
a convenient demonstration default, not a claim of overall model superiority.
Selecting another experiment changes the supported labels. The binary experiments
recognize only Healthy and Stem Canker.

Uploads stay in memory and are not saved. Maximum upload: under 12 MB and
20 megapixels. EXIF orientation is applied before the training evaluation
pipeline: RGB, resize 256, center crop 224, ImageNet normalization. Scores are
softmax outputs, not calibrated certainty or a confirmed diagnosis. An unrelated
image will still receive one of the known labels.

## Required artifacts

Each experiment needs `best_model.pt` and `results.json` under:

```text
model/
  MobileNetV3/<experiment>/
  efficientnet_b0/<experiment>/
  resnet18/<experiment>/
```

The standard `mobilenet_v3_small` directory is also supported. Checkpoints load
strictly on CPU with `weights_only=True`; no pretrained weights are downloaded.
The app reads label ordering from each checkpoint and verifies it against its
results. Up to three models are cached in memory.

## Research interpretation

Multiclass experiments use each source's own test split. Cross-dataset binary
experiments train on A, B, or A+B and test the compatible classes across C's
splits. Source validation is used for checkpoint selection. The C multiclass
experiment is a separate within-dataset experiment, not cross-dataset evidence.

Recorded scores come from single runs. Before paper-level claims, review label
compatibility, related-image leakage and dataset independence, and quantify
uncertainty with appropriate repeated runs or intervals. Perfect scores on a
small test set do not imply perfect performance in the field.

## Presentation graphs

Figures for slides and papers are in `reports/presentation_graphs/`, with a short
explanation of each chart. To regenerate them from the saved experiment results:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-plots.txt
.\.venv\Scripts\python.exe src\create_presentation_graphs.py
```

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The integration suite checks all 18 local checkpoints with an existing Dataset C
test image, verifies score and label consistency, checks result counts, and
exercises invalid/missing uploads and missing-checkpoint errors. It needs the
local model and dataset folders and does not train or overwrite models.
