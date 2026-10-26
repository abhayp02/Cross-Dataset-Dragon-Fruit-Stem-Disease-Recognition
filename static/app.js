'use strict';
const $ = id => document.getElementById(id);
const pretty = text => text.split('_').map(w => w === 'x' ? 'X' : w[0].toUpperCase() + w.slice(1)).join(' ');
const pct = value => (value * 100).toFixed(2) + '%';
let models = [], selectedFile = null, previewURL = null, busy = false;
const selection = () => models.find(m => m.architecture === $('architecture').value && m.experiment === $('experiment').value);
function option(select, value, title, disabled = false) { const node = new Option(title, value); node.disabled = disabled; select.add(node); }
function showError(id, text) { $(id).textContent = text; $(id).hidden = !text; }
function clearResult() { $('result').hidden = true; $('empty-result').hidden = false; $('result-badge').textContent = 'Ready when you are'; showError('prediction-error', ''); }
function buttonState() { $('analyze').disabled = busy || !selectedFile || !selection()?.available; }
function updateSelection() {
  clearResult(); const m = selection();
  $('model-detail').textContent = m ? 'Model: ' + m.name : 'No model available';
  $('task-detail').textContent = m ? 'Task: ' + m.labels.length + '-class classification' : '';
  $('dataset-detail').textContent = m ? 'Experiment: ' + m.title : '';
  $('class-list').textContent = m ? 'Recognized classes: ' + m.labels.map(pretty).join(' · ') : '';
  if (m && !m.available) showError('prediction-error', 'The checkpoint for this experiment is missing.');
  buttonState();
}
function updateExperiments() {
  const previous = $('experiment').value || 'c_multiclass'; $('experiment').replaceChildren();
  models.filter(m => m.architecture === $('architecture').value).forEach(m => option($('experiment'), m.experiment, m.title));
  if ([...$('experiment').options].some(o => o.value === previous)) $('experiment').value = previous;
  updateSelection();
}
function takeFile(file) {
  if (busy) return;
  selectedFile = null; clearResult();
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null; $('preview').hidden = true; $('preview').removeAttribute('src'); $('upload-placeholder').hidden = false;
  $('filename').textContent = 'No image selected'; buttonState();
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return showError('prediction-error', 'Choose a JPG, PNG, or WebP image.');
  if (file.size >= 12 * 1024 * 1024) return showError('prediction-error', 'Choose an image under 12 MB.');
  const url = URL.createObjectURL(file); previewURL = url;
  const probe = new Image();
  probe.onload = () => {
    if (url !== previewURL) return;
    if (probe.naturalWidth * probe.naturalHeight > 20000000) { URL.revokeObjectURL(url); previewURL = null; return showError('prediction-error', 'Choose an image under 20 megapixels.'); }
    selectedFile = file; $('preview').src = url; $('preview').hidden = false; $('upload-placeholder').hidden = true;
    $('filename').textContent = file.name; buttonState();
  };
  probe.onerror = () => { if (url !== previewURL) return; URL.revokeObjectURL(url); previewURL = null; showError('prediction-error', 'This image is damaged or unsupported. Choose another file.'); };
  probe.src = url;
}
$('choose').onclick = () => { $('file').value = ''; $('file').click(); };
$('file').onchange = event => takeFile(event.target.files[0]);
['dragenter', 'dragover'].forEach(name => $('drop-zone').addEventListener(name, event => { event.preventDefault(); if (!busy) $('drop-zone').classList.add('dragging'); }));
['dragleave', 'drop'].forEach(name => $('drop-zone').addEventListener(name, event => { event.preventDefault(); $('drop-zone').classList.remove('dragging'); }));
$('drop-zone').addEventListener('drop', event => takeFile(event.dataTransfer.files[0]));
$('architecture').onchange = updateExperiments; $('experiment').onchange = updateSelection;
$('analyze').onclick = async () => {
  const m = selection(); if (!selectedFile || !m || busy) return;
  busy = true; clearResult(); buttonState();
  ['choose', 'architecture', 'experiment'].forEach(id => $(id).disabled = true);
  $('analyze').textContent = 'Analyzing…'; $('result-badge').textContent = 'Running on CPU…'; document.querySelector('.results').setAttribute('aria-busy', 'true');
  const data = new FormData(); data.append('model', m.id); data.append('image', selectedFile);
  try {
    const response = await fetch('/api/predict', { method: 'POST', body: data });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Prediction failed.');
    $('predicted-class').textContent = pretty(result.prediction); $('top-score').textContent = pct(result.score);
    $('scores').replaceChildren();
    result.probabilities.forEach(item => {
      const row = document.createElement('div'); row.className = 'score';
      const name = document.createElement('span'); name.textContent = pretty(item.label);
      const track = document.createElement('div'); track.className = 'track';
      const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = (item.score * 100) + '%'; track.append(fill);
      const value = document.createElement('span'); value.className = 'score-value'; value.textContent = (item.score * 100).toFixed(1) + '%';
      row.append(name, track, value); $('scores').append(row);
    });
    $('result-context').textContent = result.model + ' · ' + result.experiment;
    $('empty-result').hidden = true; $('result').hidden = false; $('result-badge').textContent = 'Analysis complete';
  } catch (error) { showError('prediction-error', error.message === 'Failed to fetch' ? 'Cannot reach the local app. Make sure its terminal is still running.' : error.message); $('result-badge').textContent = 'Please try again'; }
  finally { busy = false; ['choose', 'architecture', 'experiment'].forEach(id => $(id).disabled = false); $('analyze').textContent = 'Analyze image'; document.querySelector('.results').setAttribute('aria-busy', 'false'); buttonState(); }
};
document.querySelectorAll('.nav').forEach(button => button.onclick = () => {
  document.querySelectorAll('.page').forEach(page => page.hidden = page.id !== button.dataset.page);
  document.querySelectorAll('.nav').forEach(nav => { nav.classList.toggle('active', nav === button); if (nav === button) nav.setAttribute('aria-current', 'page'); else nav.removeAttribute('aria-current'); });
});
function addRow(parent, values, header = false) { const row = document.createElement('tr'); values.forEach(value => { const cell = document.createElement(header ? 'th' : 'td'); cell.textContent = value; row.append(cell); }); parent.append(row); }
const comparisonModels = () => models.filter(m => m.experiment === $('compare-experiment').value);
function showComparison() {
  const group = comparisonModels(); $('comparison-rows').replaceChildren(); $('detail-model').replaceChildren();
  group.forEach(m => { addRow($('comparison-rows'), [m.name, ...['accuracy', 'macro_f1', 'macro_precision', 'macro_recall'].map(k => pct(m.metrics[k])), m.counts.test]); option($('detail-model'), m.id, m.name); });
  const external = $('compare-experiment').value.endsWith('_to_c_binary');
  $('test-context').textContent = external ? 'External evaluation on the shared Healthy and Stem Canker images from all C splits. Model selection uses source-dataset validation.' : 'Evaluation on the selected dataset’s held-out test split. Model selection uses its validation split.';
  $('download').disabled = !group.length; showDetail();
}
function showDetail() {
  const m = models.find(m => m.id === $('detail-model').value); $('class-rows').replaceChildren(); $('confusion').replaceChildren(); if (!m) return;
  $('sample-counts').textContent = `Training: ${m.counts.train} images · Validation: ${m.counts.valid} · Test: ${m.counts.test}`;
  m.labels.forEach(label => { const r = m.metrics.classification_report[label]; addRow($('class-rows'), [pretty(label), pct(r.precision), pct(r.recall), pct(r['f1-score']), r.support]); });
  const table = document.createElement('table'); const head = document.createElement('thead'); const body = document.createElement('tbody');
  addRow(head, ['Actual / Predicted', ...m.labels.map(pretty)], true);
  m.metrics.confusion_matrix.forEach((row, i) => addRow(body, [pretty(m.labels[i]), ...row])); table.append(head, body); $('confusion').append(table);
}
$('compare-experiment').onchange = showComparison; $('detail-model').onchange = showDetail;
$('download').onclick = () => {
  const columns = ['model', 'experiment', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'test_images'];
  const rows = comparisonModels().map(m => [m.name, m.experiment, m.metrics.accuracy, m.metrics.macro_f1, m.metrics.macro_precision, m.metrics.macro_recall, m.counts.test]);
  const csv = [columns, ...rows].map(row => row.map(v => '"' + String(v).replaceAll('"', '""') + '"').join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' })); const link = document.createElement('a'); link.href = url; link.download = $('compare-experiment').value + '_comparison.csv'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
async function init() {
  try {
    const response = await fetch('/api/models'); if (!response.ok) throw new Error('Cannot read model results.');
    models = (await response.json()).models; if (!models.length) throw new Error('No saved results found. Copy your model folders into this project and reload.');
    [...new Map(models.map(m => [m.architecture, m.name]))].forEach(([id, name]) => option($('architecture'), id, name));
    [...new Map(models.map(m => [m.experiment, m.title]))].forEach(([id, title]) => option($('compare-experiment'), id, title));
    $('compare-experiment').value = 'c_multiclass'; updateExperiments(); showComparison();
  } catch (error) { showError('global-error', error.message); $('download').disabled = true; }
}
init();
