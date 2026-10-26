"""Local, CPU-only prediction dashboard. Never modifies training artifacts."""
from __future__ import annotations

import io
import json
import threading
import warnings
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
torch.set_num_threads(min(4, torch.get_num_threads()))
INFERENCE_LOCK = threading.Lock()
ARCHITECTURES = {
    'mobilenet_v3_small': ('MobileNetV3 Small', ['mobilenet_v3_small', 'MobileNetV3']),
    'efficientnet_b0': ('EfficientNet-B0', ['efficientnet_b0']),
    'resnet18': ('ResNet18', ['resnet18']),
}
EXPERIMENTS = {
    'a_multiclass': 'Dataset A · 5 classes',
    'b_multiclass': 'Dataset B · 4 classes',
    'c_multiclass': 'Dataset C · 6 classes',
    'a_to_c_binary': 'A → C · Healthy / Stem Canker',
    'b_to_c_binary': 'B → C · Healthy / Stem Canker',
    'ab_to_c_binary': 'A + B → C · Healthy / Stem Canker',
}
NORMALIZE = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def ordered_labels(mapping):
    if not isinstance(mapping, dict) or not mapping or any(type(v) is not int for v in mapping.values()):
        raise ValueError('Invalid label mapping.')
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError('Label indices must be consecutive.')
    return sorted(mapping, key=mapping.get)


def catalog():
    entries = []
    for architecture, (name, folders) in ARCHITECTURES.items():
        for experiment, title in EXPERIMENTS.items():
            folder = next((ROOT / 'model' / f / experiment for f in folders
                           if (ROOT / 'model' / f / experiment / 'results.json').is_file()), None)
            if folder is None:
                continue
            try:
                result = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
                labels = ordered_labels(result['labels'])
                entries.append({
                    'id': architecture + '/' + experiment, 'architecture': architecture,
                    'name': name, 'experiment': experiment, 'title': title, 'labels': labels,
                    'available': (folder / 'best_model.pt').is_file(),
                    'metrics': result['test_metrics'], 'counts': result['record_counts'],
                    'history': result.get('history', []), 'folder': folder,
                })
            except (OSError, ValueError, KeyError, TypeError):
                app.logger.warning('Could not read result file in %s', folder)
    return entries


@lru_cache(maxsize=3)
def load_model(path, architecture, modified_ns):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    labels = ordered_labels(checkpoint['labels'])
    if architecture == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(labels))
    elif architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(labels))
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, labels


def decode_image(data):
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as original:
            if original.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise ValueError('Please choose a JPG, PNG, or WebP image.')
            original.load()
            return ImageOps.exif_transpose(original).convert('RGB')


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/models')
def model_list():
    return jsonify(models=[{k: v for k, v in entry.items() if k != 'folder'} for entry in catalog()])


@app.post('/api/predict')
def predict():
    entry = next((e for e in catalog() if e['id'] == request.form.get('model')), None)
    if entry is None:
        return jsonify(error='Select an available model and experiment.'), 400
    if not entry['available']:
        return jsonify(error='The checkpoint is missing. Copy best_model.pt into this experiment folder.'), 404
    upload = request.files.get('image')
    if upload is None or not upload.filename:
        return jsonify(error='Choose a stem image first.'), 400
    try:
        image = decode_image(upload.read())
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return jsonify(error='Cannot read this image. Use a valid JPG, PNG, or WebP under 12 MB and 20 megapixels.'), 400
    try:
        path = entry['folder'] / 'best_model.pt'
        with INFERENCE_LOCK, torch.inference_mode():
            model, labels = load_model(str(path), entry['architecture'], path.stat().st_mtime_ns)
            if labels != entry['labels']:
                raise ValueError('Checkpoint labels do not match results.json.')
            scores = torch.softmax(model(NORMALIZE(image).unsqueeze(0)), dim=1)[0]
            if not torch.isfinite(scores).all():
                raise ValueError('The model produced non-finite scores.')
            probabilities = [{'label': label, 'score': float(scores[i])} for i, label in enumerate(labels)]
        probabilities.sort(key=lambda x: x['score'], reverse=True)
        return jsonify(model=entry['name'], experiment=entry['title'],
                       prediction=probabilities[0]['label'], score=probabilities[0]['score'],
                       probabilities=probabilities)
    except Exception:
        app.logger.exception('Prediction failed for %s', entry['id'])
        return jsonify(error='This checkpoint could not be loaded or run. Check the terminal for details and verify the model files.'), 500


@app.errorhandler(413)
def too_large(error):
    return jsonify(error='The image is too large. Please choose a file under 12 MB.'), 413


if __name__ == '__main__':
    from waitress import serve
    print('Dragon Fruit Stem Analysis: http://127.0.0.1:5000', flush=True)
    serve(app, host='127.0.0.1', port=5000, threads=4)
