"""Integration checks against the user's real, locally saved checkpoints."""
import csv
import io
import unittest
from unittest.mock import patch

from PIL import Image
from app import ROOT, app, catalog, decode_image, load_model


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.model_id = 'mobilenet_v3_small/c_multiclass'

    def test_pages_and_catalog(self):
        for url in ['/', '/static/app.js', '/static/style.css']:
            with self.client.get(url) as response:
                self.assertEqual(response.status_code, 200)
        entries = self.client.get('/api/models').json['models']
        self.assertEqual(len(entries), 18)
        self.assertTrue(all(e['available'] for e in entries))
        self.assertTrue(all('folder' not in e for e in entries))
        for entry in entries:
            matrix = entry['metrics']['confusion_matrix']
            self.assertEqual(sum(map(sum, matrix)), entry['counts']['test'])

    def test_bad_requests(self):
        self.assertEqual(self.client.post('/api/predict', data={'model': '../bad'}).status_code, 400)
        self.assertEqual(self.client.post('/api/predict', data={'model': self.model_id}).status_code, 400)
        response = self.client.post('/api/predict', data={'model': self.model_id, 'image': (io.BytesIO(b'bad'), 'bad.jpg')})
        self.assertEqual(response.status_code, 400)
        with patch.dict(app.config, MAX_CONTENT_LENGTH=50):
            self.assertEqual(self.client.post('/api/predict', data=b'x' * 100).status_code, 413)
        entry = next(e for e in catalog() if e['id'] == self.model_id)
        with patch('app.catalog', return_value=[dict(entry, available=False)]):
            self.assertEqual(self.client.post('/api/predict', data={'model': self.model_id}).status_code, 404)

    def test_orientation_and_formats(self):
        image = Image.new('RGB', (60, 30), 'green')
        exif = Image.Exif(); exif[274] = 6
        data = io.BytesIO(); image.save(data, format='JPEG', exif=exif)
        self.assertEqual(decode_image(data.getvalue()).size, (30, 60))
        for mode, fmt in [('RGBA', 'PNG'), ('L', 'PNG'), ('RGB', 'WEBP')]:
            data = io.BytesIO(); Image.new(mode, (60, 30)).save(data, format=fmt)
            self.assertEqual(decode_image(data.getvalue()).mode, 'RGB')

    def test_real_inference_all_18_checkpoints(self):
        root = ROOT / 'data/sources/dragon_fruit_stem_disease_v1'
        with (root / 'manifest.csv').open(encoding='utf-8') as handle:
            row = next(r for r in csv.DictReader(handle) if r['split'] == 'test')
        data = (root / row['file']).read_bytes()
        for entry in catalog():
            with self.subTest(model=entry['id']):
                response = self.client.post('/api/predict', data={'model': entry['id'], 'image': (io.BytesIO(data), 'sample.jpg')})
                self.assertEqual(response.status_code, 200, response.json)
                result = response.json
                self.assertEqual(set(p['label'] for p in result['probabilities']), set(entry['labels']))
                self.assertAlmostEqual(sum(p['score'] for p in result['probabilities']), 1, places=5)
                self.assertEqual(result['prediction'], result['probabilities'][0]['label'])
                print('PASS real inference:', entry['id'], flush=True)
        load_model.cache_clear()


if __name__ == '__main__':
    unittest.main(verbosity=2)
