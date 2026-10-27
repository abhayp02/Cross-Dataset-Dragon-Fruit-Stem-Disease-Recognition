# Working project screenshots

These are unmodified browser captures of the running local application.

Suggested PPT captions:

1. **01_upload_dashboard.jpg** — Image upload and model/experiment selection.
2. **02_disease_prediction.jpg** — MobileNetV3 Small predicts Anthracnose on a Dataset C test image; all six class scores are displayed.
3. **03_healthy_prediction.jpg** — The same model predicts Healthy on a different Dataset C test image.
4. **04_multiclass_results.jpg** — Saved accuracy, macro F1, precision, recall, and detailed results for Dataset C multiclass.
5. **05_cross_dataset_results.jpg** — Train on A, evaluate Healthy / Stem Canker on C; compare three CNN architectures.
6. **06_project_overview.jpg** — Explanation of the CNN workflow, datasets, and research scope.

Use 02, 03, and 05 for a short demonstration section. Full-page captures 04–06
are tall; crop or split them in PowerPoint for readable text. Image 04 includes
the complete per-class metrics and confusion matrix.

Prediction screenshots demonstrate software operation using existing test data;
they are not additional independent evaluations. The displayed model score is
a softmax output, not a confirmed diagnosis or an overall accuracy metric.
