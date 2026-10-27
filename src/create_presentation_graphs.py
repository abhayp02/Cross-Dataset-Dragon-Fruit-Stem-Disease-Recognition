"""Export presentation figures from saved results; never runs training."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'presentation_graphs'
MODELS = [('MobileNetV3', 'MobileNetV3 Small'), ('efficientnet_b0', 'EfficientNet-B0'), ('resnet18', 'ResNet18')]
EXPERIMENTS = ['a_multiclass', 'b_multiclass', 'c_multiclass', 'a_to_c_binary', 'b_to_c_binary', 'ab_to_c_binary']
COLORS = ['#237A57', '#377BB5', '#CB7741']
NOTE = 'Recorded single runs; no uncertainty estimates. Scores do not establish real-world diagnostic accuracy.'


def read_results():
    results = {}
    for folder, _ in MODELS:
        for experiment in EXPERIMENTS:
            base = ROOT / 'model' / folder / experiment
            if folder == 'MobileNetV3' and not base.exists():
                base = ROOT / 'model' / 'mobilenet_v3_small' / experiment
            record = json.loads((base / 'results.json').read_text(encoding='utf-8'))
            labels = sorted(record['labels'], key=record['labels'].get)
            matrix = np.array(record['test_metrics']['confusion_matrix'])
            assert matrix.shape == (len(labels), len(labels)), base
            assert matrix.sum() == record['record_counts']['test'], base
            for key in ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']:
                assert 0 <= record['test_metrics'][key] <= 1, (base, key)
            assert np.isclose(np.trace(matrix) / matrix.sum(), record['test_metrics']['accuracy']), base
            results[folder, experiment] = record
    for experiment in EXPERIMENTS:
        reference = results[MODELS[0][0], experiment]
        assert all(results[folder, experiment]['labels'] == reference['labels'] and
                   results[folder, experiment]['record_counts'] == reference['record_counts'] for folder, _ in MODELS)
    return results


def save(fig, name, title, subtitle, note=NOTE, left=.07, right=.97):
    fig.suptitle(title, fontsize=20, fontweight='bold', x=.07, ha='left', y=.985, color='#173B2D')
    fig.text(.07, .925, subtitle, fontsize=11, color='#535F59')
    fig.text(.07, .025, note, fontsize=9, color='#626C66')
    fig.subplots_adjust(top=.81, bottom=.2, left=left, right=right, wspace=.28)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{extension}', dpi=300, facecolor='white')
    plt.close(fig)


def bars(results, experiments, titles, name, title, subtitle):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, (metric, heading) in zip(axes, [('accuracy', 'Test accuracy'), ('macro_f1', 'Test macro F1')]):
        x = np.arange(len(experiments))
        for j, (folder, label) in enumerate(MODELS):
            values = [100 * results[folder, exp]['test_metrics'][metric] for exp in experiments]
            bars = ax.bar(x + (j - 1) * .24, values, width=.22, label=label, color=COLORS[j], zorder=3)
            ax.bar_label(bars, labels=[f'{v:.1f}' for v in values], padding=4, fontsize=9)
        ax.set(title=heading, ylim=(0, 112), ylabel='Score (%)', xticks=x, xticklabels=titles)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis='y', alpha=.2, zorder=0)
    fig.legend([label for _, label in MODELS], loc='lower center', bbox_to_anchor=(.52, .09), ncol=3, frameon=False)
    save(fig, name, title, subtitle)


def training_curves(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    for folder_index, (folder, label) in enumerate(MODELS):
        history = results[folder, 'c_multiclass']['history']
        epochs = [row['epoch'] for row in history]
        for ax, key, heading in zip(axes, ['train_loss', 'accuracy', 'macro_f1'], ['Training loss', 'Validation accuracy', 'Validation macro F1']):
            multiplier = 1 if key == 'train_loss' else 100
            ax.plot(epochs, [row[key] * multiplier for row in history], marker='o', markersize=4, linewidth=2, color=COLORS[folder_index], label=label)
            ax.set(title=heading, xlabel='Epoch', ylabel='Cross-entropy loss' if key == 'train_loss' else 'Score (%)')
            ax.set_xticks([1, 3, 6, 9, 12]); ax.grid(alpha=.2)
            if key != 'train_loss':
                ax.set_ylim(0, 105)
    fig.legend([label for _, label in MODELS], loc='lower center', bbox_to_anchor=(.52, .09), ncol=3, frameon=False)
    save(fig, '03_dataset_c_learning_curves', 'How the CNN models learned',
         'Dataset C multiclass · 385 training images / 82 validation images · 12 epochs',
         'Accuracy and F1 curves use validation data. Test-set scores are not plotted across epochs.')


def confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.8))
    short = ['Anthracnose', 'Brown spot', 'Gray blight', 'Healthy', 'Soft rot', 'Stem canker']
    for ax, (folder, label) in zip(axes, MODELS):
        matrix = np.array(results[folder, 'c_multiclass']['test_metrics']['confusion_matrix'])
        percentages = matrix / matrix.sum(axis=1, keepdims=True) * 100
        ax.imshow(percentages, cmap='Greens', vmin=0, vmax=100)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f'{value}\n({percentages[i,j]:.0f}%)', ha='center', va='center', fontsize=9,
                    color='white' if percentages[i,j] > 55 else '#173B2D')
        ax.set(title=label, xticks=range(6), yticks=range(6), xticklabels=short, yticklabels=short,
               xlabel='Predicted class', ylabel='Actual class')
        plt.setp(ax.get_xticklabels(), rotation=40, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)
    save(fig, '04_dataset_c_confusion_matrices', 'Where each model makes mistakes',
         'Dataset C held-out test split · 84 images · Cells show image counts and percentages within each actual class',
         'Shared colour scale: 0–100% of each row. Brown spot = Brown Stem Spot. Small classes have limited test support.')


def recall_heatmap(results):
    fig, ax = plt.subplots(figsize=(12, 7))
    reference = results[MODELS[0][0], 'c_multiclass']
    labels = sorted(reference['labels'], key=reference['labels'].get)
    values = np.array([[100 * results[folder, 'c_multiclass']['test_metrics']['classification_report'][label]['recall'] for folder, _ in MODELS] for label in labels])
    image = ax.imshow(values, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f'{value:.1f}%', ha='center', va='center', fontsize=14, color='white' if value > 65 else '#173B2D')
    ax.set_xticks(range(3), [name for _, name in MODELS])
    ax.set_yticks(range(len(labels)), [label.replace('_', ' ').title() + f"  (n={int(reference['test_metrics']['classification_report'][label]['support'])})" for label in labels])
    ax.tick_params(length=0, pad=10)
    colorbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.025)
    colorbar.set_label('Recall (%)')
    save(fig, '05_dataset_c_class_recall', 'Accuracy alone can hide missed disease cases',
         'Dataset C · Recall is the share of actual examples correctly recognized in each class',
         'Stem Canker has only 5 test images: each missed image changes recall by 20 percentage points.', left=.28, right=.91)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.titlepad': 14,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    results = read_results()
    group_titles = []
    for letter, experiment in zip('ABC', EXPERIMENTS[:3]):
        record = results[MODELS[0][0], experiment]
        group_titles.append(f"Dataset {letter}\n{len(record['labels'])} classes · n={record['record_counts']['test']:,}")
    bars(results, EXPERIMENTS[:3], group_titles,
         '01_multiclass_comparison', 'CNN comparison within each dataset',
         'Each group uses its own held-out test split. Compare architectures within a group; the tasks differ across datasets.')
    bars(results, EXPERIMENTS[3:], ['Train A → Test C', 'Train B → Test C', 'Train A+B → Test C'],
         '02_cross_dataset_comparison', 'Can the CNN models generalize to Dataset C?',
         'Binary task: Healthy / Stem Canker · Same 155 external test images (126 healthy, 29 stem canker) in every group')
    training_curves(results)
    confusion_matrices(results)
    recall_heatmap(results)
    with (OUT / 'experiment_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'experiment', 'train_images', 'validation_images', 'test_images', 'accuracy_percent', 'macro_f1_percent', 'macro_precision_percent', 'macro_recall_percent'])
        for folder, name in MODELS:
            for experiment in EXPERIMENTS:
                result = results[folder, experiment]
                writer.writerow([name, experiment, *[result['record_counts'][k] for k in ['train', 'valid', 'test']], *[result['test_metrics'][k] * 100 for k in ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']]])
    print(f'Created 5 figures in PNG / PDF / SVG and one metrics CSV in {OUT}')


if __name__ == '__main__':
    main()
