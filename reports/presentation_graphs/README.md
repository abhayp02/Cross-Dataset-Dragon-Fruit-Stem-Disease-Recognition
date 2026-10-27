# Presentation graphs

Generated from the 18 saved `results.json` files. PNG files are 300 DPI for
PowerPoint; PDF and SVG files are vector graphics for papers or resizing.
`experiment_metrics.csv` contains the plotted summary metrics as percentages.

Suggested presentation order:

1. **01_multiclass_comparison** — accuracy and macro F1 for the three models on
   each source's own test split. Compare architectures within each dataset group;
   A, B, and C have different classes and test images.
2. **02_cross_dataset_comparison** — models trained on A, B, or A+B and evaluated
   on the same 155 Healthy / Stem Canker images from C. This addresses transfer
   between datasets for the two compatible labels.
3. **03_dataset_c_learning_curves** — training loss, validation accuracy, and
   validation macro F1 over 12 epochs for C multiclass. These are validation
   curves, not repeated test evaluations; no validation loss was saved.
4. **04_dataset_c_confusion_matrices** — C multiclass test predictions; rows are
   actual classes and columns are predicted classes. Counts and row percentages
   are shown, with the same colour scale for all models.
5. **05_dataset_c_class_recall** — how many examples of each C class were found.
   `n` is the number of test images in that class.

Explain accuracy as the proportion of correct predictions. Macro F1 gives equal
weight to every class and combines precision and recall. Recall measures the
share of actual class examples correctly recognized.

An important finding: on C multiclass, Stem Canker recall is 40% for MobileNetV3
and 20% for EfficientNet-B0 and ResNet18, despite high overall accuracy. There are
only 5 test examples of Stem Canker. This six-class task differs from the binary
external experiment, which evaluates 29 Stem Canker images and 126 Healthy images.

These are single-run recorded results. Error bars are not fabricated; repeated
runs or an appropriate uncertainty analysis are needed to quantify variation.
Label compatibility, related-image leakage, and source independence still need
review before strong paper claims. ROC/PR curves need per-image scores; model
speed, memory, parameter counts, and training duration are not measured by these
saved result files, so those benchmarks are not claimed here.

To regenerate from the project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-plots.txt
.\.venv\Scripts\python.exe src\create_presentation_graphs.py
```

The generator validates matrix totals against test counts and checks that each
experiment has matching class mappings and sample counts across architectures.
It reads saved results without training models or changing checkpoints.
