class HFCodeGenerator {
    constructor() {
        this.dataset = [];
        this.labelStats = {};
        this.codeEditor = null;
        this.dataChanged = false;
        this.init();
    }

    init() {
        this.setupCodeEditor();
        this.attachEventListeners();
        this.loadSampleData();
        this.updateStats();
        this.showToast('Welcome to Hugging Face Training Code Generator Pro!', 'info');
    }

    setupCodeEditor() {
        this.codeEditor = CodeMirror.fromTextArea(document.getElementById('pythonCodeOutput'), {
            mode: 'python',
            theme: 'monokai',
            lineNumbers: true,
            readOnly: false,
            lineWrapping: true,
            autofocus: true
        });
    }

    attachEventListeners() {
        document.addEventListener('click', (e) => this.handleTableActions(e));
        document.addEventListener('change', (e) => this.handleSelects(e));
        document.addEventListener('input', (e) => this.handleInputs(e));

        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.closest('.tab')));
        });

        document.getElementById('addRowBtn').addEventListener('click', () => this.addNewRow());
        document.getElementById('clearTableBtn').addEventListener('click', () => this.clearTable());
        document.getElementById('addSampleDataBtn').addEventListener('click', () => this.addSampleData());
        document.getElementById('normalizeLabelsBtn').addEventListener('click', () => this.normalizeLabels());
        document.getElementById('analyzeDataBtn').addEventListener('click', () => this.openAnalysisModal());

        document.getElementById('generateCodeBtn').addEventListener('click', () => this.generateFullCode());
        document.getElementById('copyCodeBtn').addEventListener('click', () => this.copyCode());
        document.getElementById('downloadCodeBtn').addEventListener('click', () => this.downloadCode());
        document.getElementById('openColabBtn').addEventListener('click', () => this.openInColab());

        document.getElementById('importCSVBtn').addEventListener('click', () => this.importFromCSV());
        document.getElementById('importJSONBtn').addEventListener('click', () => this.importFromJSON());
        document.getElementById('copyCSVBtn').addEventListener('click', () => this.copyCSV());
        document.getElementById('downloadCSVBtn').addEventListener('click', () => this.downloadCSV());
        document.getElementById('copyJSONBtn').addEventListener('click', () => this.copyJSON());
        document.getElementById('downloadJSONBtn').addEventListener('click', () => this.downloadJSON());

        document.getElementById('closeAnalysisBtn').addEventListener('click', () => this.closeAnalysisModal());
        document.getElementById('analysisModal').addEventListener('click', (e) => {
            if (e.target.id === 'analysisModal') this.closeAnalysisModal();
        });
    }

    handleTableActions(e) {
        if (e.target.closest('.action-btn.delete')) {
            const index = parseInt(e.target.closest('.action-btn').dataset.index);
            this.deleteRow(index);
        }
        if (e.target.closest('.action-btn.edit')) {
            const index = parseInt(e.target.closest('.action-btn').dataset.index);
            const textArea = document.querySelector(`textarea[data-row-index="${index}"][data-field="text"]`);
            if (textArea) textArea.focus();
        }
        if (e.target.closest('[data-action="move-up"]')) {
            const index = parseInt(e.target.closest('[data-action]').dataset.index);
            this.moveRow(index, 'up');
        }
        if (e.target.closest('[data-action="move-down"]')) {
            const index = parseInt(e.target.closest('[data-action]').dataset.index);
            this.moveRow(index, 'down');
        }
    }

    handleSelects(e) {
        if (e.target.id === 'languageSelect') {
            this.updateModelOptions(e.target.value);
        }
        if (e.target.id === 'modelSelect') {
            const customGroup = document.getElementById('customModelGroup');
            customGroup.style.display = e.target.value === 'custom' ? 'block' : 'none';
        }
    }

    handleInputs(e) {
        if (e.target.dataset.field && e.target.dataset.rowIndex !== undefined) {
            const index = parseInt(e.target.dataset.rowIndex);
            const field = e.target.dataset.field;
            this.dataset[index][field] = e.target.value.trim();
            this.updateStats();
        }
    }

    updateModelOptions(language) {
        const modelSelect = document.getElementById('modelSelect');
        const models = {
            'en': `
                <option value="distilbert-base-uncased">DistilBERT (English)</option>
                <option value="bert-base-uncased">BERT (English)</option>
                <option value="roberta-base">RoBERTa (English)</option>
                <option value="custom">Custom Model</option>
            `,
            'multilingual': `
                <option value="xlm-roberta-base">XLM-RoBERTa (Multilingual)</option>
                <option value="bert-base-multilingual-cased">BERT (Multilingual)</option>
                <option value="custom">Custom Model</option>
            `,
            'other': `
                <option value="custom">Custom Model</option>
            `
        };
        modelSelect.innerHTML = models[language] || models['en'];
    }

    switchTab(tabElement) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));

        tabElement.classList.add('active');
        const tabId = tabElement.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');

        if (tabId === 'import-export') {
            this.updateExportData();
        }
    }

    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    addNewRow(text = '', label = '') {
        this.dataset.push({ text: String(text).trim(), label: String(label).trim() });
        this.renderTable();
        this.updateStats();
    }

    renderTable() {
        const tbody = document.getElementById('tableBody');
        const fragment = document.createDocumentFragment();

        this.dataset.forEach((item, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>
                    <textarea class="text-input" data-row-index="${index}" data-field="text" placeholder="Enter text here">${this.escapeHtml(item.text)}</textarea>
                </td>
                <td>
                    <input type="text" class="label-input" data-row-index="${index}" data-field="label" value="${this.escapeHtml(item.label)}" placeholder="Label">
                </td>
                <td>
                    <div class="row-actions">
                        <button class="action-btn edit" data-index="${index}" title="Edit" aria-label="Edit row ${index + 1}">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="action-btn delete" data-index="${index}" title="Delete" aria-label="Delete row ${index + 1}">
                            <i class="fas fa-trash"></i>
                        </button>
                        <button class="action-btn" data-action="move-up" data-index="${index}" ${index === 0 ? 'disabled' : ''} title="Move up" aria-label="Move row up">
                            <i class="fas fa-arrow-up"></i>
                        </button>
                        <button class="action-btn" data-action="move-down" data-index="${index}" ${index === this.dataset.length - 1 ? 'disabled' : ''} title="Move down" aria-label="Move row down">
                            <i class="fas fa-arrow-down"></i>
                        </button>
                    </div>
                </td>
            `;
            fragment.appendChild(row);
        });

        tbody.innerHTML = '';
        tbody.appendChild(fragment);
        document.getElementById('dataCount').textContent = `${this.dataset.length} records`;
        this.dataChanged = true;
    }

    deleteRow(index) {
        if (confirm('Are you sure? This row will be permanently deleted.')) {
            this.dataset.splice(index, 1);
            this.renderTable();
            this.updateStats();
            this.showToast('Row deleted successfully', 'warning');
        }
    }

    moveRow(index, direction) {
        if (direction === 'up' && index > 0) {
            [this.dataset[index], this.dataset[index - 1]] = [this.dataset[index - 1], this.dataset[index]];
            this.renderTable();
        } else if (direction === 'down' && index < this.dataset.length - 1) {
            [this.dataset[index], this.dataset[index + 1]] = [this.dataset[index + 1], this.dataset[index]];
            this.renderTable();
        }
    }

    clearTable() {
        if (this.dataset.length > 0 && confirm('Are you sure? All data will be deleted permanently.')) {
            this.dataset = [];
            this.renderTable();
            this.updateStats();
            this.showToast('All data cleared', 'warning');
        }
    }

    normalizeLabels() {
        let changes = 0;
        this.dataset.forEach(item => {
            const original = item.label;
            const normalized = original.trim().toLowerCase();
            if (original !== normalized) {
                changes++;
                item.label = normalized;
            }
        });

        this.renderTable();
        this.updateStats();

        if (changes > 0) {
            this.showToast(`${changes} labels normalized`, 'success');
        } else {
            this.showToast('All labels are already normalized', 'info');
        }
    }

    addSampleData() {
        const samples = [
            { text: "This movie was absolutely amazing with excellent acting", label: "positive" },
            { text: "I am completely satisfied with this product and recommend it", label: "positive" },
            { text: "The book was very engaging and educational", label: "positive" },
            { text: "Their service was terrible and I will never buy from them again", label: "negative" },
            { text: "The customer service was awful", label: "negative" },
            { text: "Great experience, will definitely do it again", label: "positive" },
            { text: "I cannot believe how good this was", label: "positive" },
            { text: "Worst experience of my life", label: "negative" },
            { text: "I was impressed by the product quality", label: "positive" },
            { text: "The support team was very unhelpful", label: "negative" }
        ];

        this.dataset = [...this.dataset, ...samples];
        this.renderTable();
        this.updateStats();
        this.showToast('Sample data added successfully', 'success');
    }

    updateStats() {
        const validSamples = this.dataset.filter(d => d.text !== '' && d.label !== '');
        const labels = validSamples.map(d => d.label.toLowerCase());
        const uniqueLabels = [...new Set(labels)];
        const avgLength = validSamples.length > 0 ?
            Math.round(validSamples.reduce((sum, d) => sum + d.text.length, 0) / validSamples.length) : 0;

        let balance = '100%';
        if (uniqueLabels.length > 1) {
            const labelCounts = {};
            labels.forEach(label => {
                labelCounts[label] = (labelCounts[label] || 0) + 1;
            });

            const counts = Object.values(labelCounts);
            const maxCount = Math.max(...counts);
            const minCount = Math.min(...counts);
            balance = `${Math.round((minCount / maxCount) * 100)}%`;
        }

        document.getElementById('totalSamples').textContent = validSamples.length;
        document.getElementById('uniqueLabels').textContent = uniqueLabels.length;
        document.getElementById('avgTextLength').textContent = avgLength;
        document.getElementById('dataBalance').textContent = balance;

        this.updateLabelDistribution(labels, uniqueLabels);
        this.updateDataPreview();
        this.checkDataQuality();
        this.updateCodeStatus();
    }

    updateLabelDistribution(labels, uniqueLabels) {
        const container = document.getElementById('distributionBars');
        container.innerHTML = '';

        if (labels.length === 0) return;

        const labelCounts = {};
        labels.forEach(label => {
            labelCounts[label] = (labelCounts[label] || 0) + 1;
        });

        const fragment = document.createDocumentFragment();

        uniqueLabels.forEach(label => {
            const count = labelCounts[label] || 0;
            const percentage = Math.round((count / labels.length) * 100);

            const item = document.createElement('div');
            item.className = 'label-item';
            item.innerHTML = `
                <span>${this.escapeHtml(label)}</span>
                <span>${count} samples (${percentage}%)</span>
            `;
            fragment.appendChild(item);

            const barWrapper = document.createElement('div');
            barWrapper.className = 'distribution-bar';
            const fill = document.createElement('div');
            fill.className = 'distribution-fill';
            fill.style.width = `${percentage}%`;
            barWrapper.appendChild(fill);
            fragment.appendChild(barWrapper);
        });

        container.appendChild(fragment);
        this.labelStats = labelCounts;
    }

    updateDataPreview() {
        const container = document.getElementById('dataPreview');

        if (this.dataset.length === 0) {
            container.innerHTML = '<p style="color: #888; text-align: center; padding: 20px;">No data to display yet</p>';
            return;
        }

        const fragment = document.createDocumentFragment();
        const wrapper = document.createElement('div');
        wrapper.style.fontFamily = 'monospace';
        wrapper.style.fontSize = '12px';

        this.dataset.slice(0, 5).forEach((item, index) => {
            const div = document.createElement('div');
            div.style.marginBottom = '15px';
            div.style.padding = '12px';
            div.style.background = 'rgba(0,0,0,0.2)';
            div.style.borderRadius = '8px';

            const title = document.createElement('div');
            title.style.color = 'var(--accent)';
            title.style.marginBottom = '8px';
            title.style.fontWeight = 'bold';
            title.textContent = `Sample ${index + 1}:`;
            div.appendChild(title);

            const text = document.createElement('div');
            text.style.marginBottom = '8px';
            text.style.color = '#ddd';
            text.textContent = item.text.substring(0, 80) + (item.text.length > 80 ? '...' : '');
            div.appendChild(text);

            const label = document.createElement('div');
            label.style.color = 'var(--success)';
            label.style.fontWeight = 'bold';
            label.innerHTML = `Label: <span>${this.escapeHtml(item.label || 'No label')}</span>`;
            div.appendChild(label);

            wrapper.appendChild(div);
        });

        if (this.dataset.length > 5) {
            const more = document.createElement('p');
            more.style.color = '#888';
            more.style.textAlign = 'center';
            more.style.marginTop = '10px';
            more.textContent = `+ ${this.dataset.length - 5} more samples`;
            wrapper.appendChild(more);
        }

        container.innerHTML = '';
        container.appendChild(wrapper);
    }

    checkDataQuality() {
        const warnings = [];
        const validSamples = this.dataset.filter(d => d.text !== '' && d.label !== '');

        if (validSamples.length === 0) {
            warnings.push({ icon: 'fa-database', message: 'No valid data to analyze' });
        }

        const emptyLabels = this.dataset.filter(d => d.text !== '' && d.label === '').length;
        if (emptyLabels > 0) {
            warnings.push({ icon: 'fa-exclamation-triangle', message: `${emptyLabels} samples without labels` });
        }

        const labels = validSamples.map(d => d.label.toLowerCase());
        const uniqueLabels = [...new Set(labels)];
        const originalLabels = validSamples.map(d => d.label).filter(l => l !== '');
        const originalUniqueLabels = [...new Set(originalLabels)];

        if (originalUniqueLabels.length > uniqueLabels.length) {
            warnings.push({ icon: 'fa-font', message: 'Labels use different cases (e.g., "Positive" and "positive")' });
        }

        if (uniqueLabels.length > 1) {
            const labelCounts = {};
            labels.forEach(label => {
                if (label !== '') labelCounts[label] = (labelCounts[label] || 0) + 1;
            });

            const counts = Object.values(labelCounts);
            if (counts.length > 1) {
                const maxCount = Math.max(...counts);
                const minCount = Math.min(...counts);
                const ratio = maxCount / minCount;

                if (ratio > 5) {
                    warnings.push({
                        icon: 'fa-balance-scale',
                        message: `Data is imbalanced (ratio: ${Math.round(ratio)}:1)`
                    });
                }
            }
        }

        const container = document.getElementById('qualityWarnings');
        if (warnings.length === 0) {
            container.innerHTML = `
                <div style="background: rgba(42, 157, 143, 0.1); border: 1px solid var(--success);
                            border-radius: 10px; padding: 15px; text-align: center;">
                    <i class="fas fa-check-circle" style="color: var(--success);"></i>
                    <span style="color: var(--success); margin-left: 10px;">Data quality is good</span>
                </div>
            `;
        } else {
            const fragment = document.createDocumentFragment();
            warnings.forEach(warning => {
                const div = document.createElement('div');
                div.className = 'warning-item';
                div.innerHTML = `
                    <i class="fas ${warning.icon}"></i>
                    <span>${this.escapeHtml(warning.message)}</span>
                `;
                fragment.appendChild(div);
            });
            container.innerHTML = '';
            container.appendChild(fragment);
        }
    }

    updateCodeStatus() {
        const validSamples = this.dataset.filter(d => d.text !== '' && d.label !== '').length;
        const statusEl = document.getElementById('codeReadyStatus');
        const statsEl = document.getElementById('codeStats');
        const progressEl = document.getElementById('codeProgress');

        if (validSamples >= 10) {
            statusEl.textContent = 'Ready';
            statusEl.className = 'badge badge-success';
            statsEl.textContent = `${validSamples} valid samples`;
            progressEl.style.width = '100%';
        } else if (validSamples > 0) {
            statusEl.textContent = 'Incomplete';
            statusEl.className = 'badge badge-warning';
            statsEl.textContent = `${validSamples} samples (need at least 10)`;
            progressEl.style.width = `${(validSamples / 10) * 100}%`;
        } else {
            statusEl.textContent = 'Not Ready';
            statusEl.className = 'badge badge-danger';
            statsEl.textContent = 'No data';
            progressEl.style.width = '0%';
        }
    }

    generateFullCode() {
        const validData = this.dataset.filter(d => d.text !== '' && d.label !== '');

        if (validData.length < 10) {
            this.showToast('At least 10 valid samples are required', 'error');
            return;
        }

        const language = document.getElementById('languageSelect').value;
        const modelSelect = document.getElementById('modelSelect').value;
        const customModel = document.getElementById('customModelName').value;
        const modelName = modelSelect === 'custom' ? customModel : modelSelect;
        const numEpochs = document.getElementById('numEpochs').value;
        const batchSize = document.getElementById('batchSize').value;
        const learningRate = document.getElementById('learningRate').value;
        const maxLength = document.getElementById('maxLength').value;
        const evalMetric = document.getElementById('evalMetric').value;
        const testSplit = document.getElementById('testSplit').value;

        const code = this.generatePythonCode(validData, {
            language, modelName, numEpochs, batchSize, learningRate, maxLength, evalMetric, testSplit
        });

        this.codeEditor.setValue(code);
        this.showToast('Code generated successfully!', 'success');
        document.querySelector('.tab[data-tab="code-output"]').click();
    }

    generatePythonCode(data, settings) {
        const labelMapping = {};
        const normalizedData = data.map(item => {
            const normalizedLabel = item.label.trim().toLowerCase();
            return { text: item.text, label: normalizedLabel };
        });

        const uniqueLabels = [...new Set(normalizedData.map(d => d.label))];
        uniqueLabels.forEach((label, index) => {
            labelMapping[label] = index;
        });

        const reverseMapping = {};
        Object.entries(labelMapping).forEach(([label, id]) => {
            reverseMapping[id] = label;
        });

        const labelCounts = {};
        normalizedData.forEach(item => {
            labelCounts[item.label] = (labelCounts[item.label] || 0) + 1;
        });

        const isImbalanced = () => {
            const counts = Object.values(labelCounts);
            const maxCount = Math.max(...counts);
            const minCount = Math.min(...counts);
            return maxCount / minCount > 5;
        };

        return `import torch
import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

print("🖥️ Checking hardware resources:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   ⚠️ No GPU found! Training on CPU will be slow.")

!pip install transformers datasets accelerate scikit-learn pandas numpy matplotlib -q

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

data = ${JSON.stringify(normalizedData, null, 2)}

label_mapping = ${JSON.stringify(labelMapping, null, 2)}
reverse_mapping = ${JSON.stringify(reverseMapping, null, 2)}

print("\\n🏷️ Label mapping:")
for label, idx in label_mapping.items():
    print(f"   '{label}' → {idx}")

texts = [item['text'] for item in data]
labels = [label_mapping[item['label']] for item in data]

df = pd.DataFrame({'text': texts, 'label': labels})

print(f"\\n📊 Dataset statistics:")
print(f"   Total samples: {len(df)}")
print(f"   Number of classes: {len(label_mapping)}")
print(f"   Class distribution:")

for label, idx in label_mapping.items():
    count = list(labels).count(idx)
    percentage = (count / len(labels)) * 100
    print(f"      Class {idx} ('{label}'): {count} samples ({percentage:.1f}%)")

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(
    test_size=${settings.testSplit},
    seed=42,
    stratify_by_column='label'
)

print(f"\\n🔀 Data split:")
print(f"   Training: {len(dataset['train'])} samples")
print(f"   Testing: {len(dataset['test'])} samples")

print(f"\\n🤖 Loading model ${settings.modelName}...")

try:
    tokenizer = AutoTokenizer.from_pretrained("${settings.modelName}")
    model = AutoModelForSequenceClassification.from_pretrained(
        "${settings.modelName}",
        num_labels=len(label_mapping),
        ignore_mismatched_sizes=True
    )
    print("   ✅ Model loaded successfully")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    print("   Loading alternative model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_mapping)
    )

def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=${settings.maxLength},
        return_tensors='pt'
    )

print("\\n⚙️ Tokenizing data...")

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['text'])
tokenized_dataset.set_format('torch')

print("   ✅ Preprocessing complete")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted')
    }

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=${settings.learningRate},
    per_device_train_batch_size=${settings.batchSize},
    per_device_eval_batch_size=${settings.batchSize},
    num_train_epochs=${settings.numEpochs},
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir='./logs',
    logging_steps=10,
    report_to='none',
    seed=42
)

print(f"\\n🎯 Training parameters:")
print(f"   Epochs: ${settings.numEpochs}")
print(f"   Batch size: ${settings.batchSize}")
print(f"   Learning rate: ${settings.learningRate}")
print(f"   Evaluation metric: ${settings.evalMetric}")

print("\\n🚀 Starting model training...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

train_results = trainer.train()

print("\\n✅ Training complete!")

print("\\n📈 Final evaluation:")
eval_results = trainer.evaluate()

for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"   {key}: {value:.4f}")

predictions = trainer.predict(tokenized_dataset['test'])
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

print(f"\\n📊 Classification Report:")
class_names = [reverse_mapping[str(i)] for i in range(len(reverse_mapping))]
print(classification_report(true_labels, pred_labels, target_names=class_names, digits=4))

print("\\n💾 Saving model...")
trainer.save_model('./trained_model')
tokenizer.save_pretrained('./trained_model')

with open('./trained_model/label_mapping.json', 'w') as f:
    json.dump({'label_to_id': label_mapping, 'id_to_label': reverse_mapping}, f, indent=2)

print("✨ Training complete! Model saved to ./trained_model/")`;
    }

    copyCode() {
        const code = this.codeEditor.getValue();
        navigator.clipboard.writeText(code).then(() => {
            this.showToast('Code copied to clipboard', 'success');
        }).catch(() => {
            this.showToast('Error copying code', 'error');
        });
    }

    downloadCode() {
        const code = this.codeEditor.getValue();
        if (!code.trim()) {
            this.showToast('No code to download', 'error');
            return;
        }

        const blob = new Blob([code], { type: 'text/x-python' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `hf_training_${Date.now()}.py`;
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('Code downloaded successfully', 'success');
    }

    openInColab() {
        const code = this.codeEditor.getValue();
        if (!code.trim()) {
            this.showToast('No code to send to Colab', 'error');
            return;
        }

        const encodedCode = encodeURIComponent(code);
        const colabUrl = `https://colab.research.google.com/`;
        window.open(colabUrl, '_blank');
        this.showToast('Opening Google Colab...', 'info');
    }

    updateExportData() {
        const validData = this.dataset.filter(d => d.text !== '' && d.label !== '');

        const csvLines = validData.map(d => `"${d.text.replace(/"/g, '""')}","${d.label.replace(/"/g, '"')}"`);
        const csvContent = "text,label\n" + csvLines.join("\n");
        document.getElementById('csvExport').value = csvContent;

        const jsonContent = JSON.stringify(validData, null, 2);
        document.getElementById('jsonExport').value = jsonContent;
    }

    importFromCSV() {
        const csvText = document.getElementById('csvImport').value.trim();
        if (!csvText) {
            this.showToast('Please enter CSV data', 'error');
            return;
        }

        try {
            Papa.parse(csvText, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                    const newData = results.data.filter(row =>
                        row.text && row.text.trim() && row.label && row.label.trim()
                    ).map(row => ({
                        text: String(row.text).trim(),
                        label: String(row.label).trim()
                    }));

                    if (newData.length === 0) {
                        this.showToast('No valid data found in CSV', 'error');
                        return;
                    }

                    this.dataset = [...this.dataset, ...newData];
                    this.renderTable();
                    this.updateStats();
                    this.showToast(`${newData.length} samples imported successfully`, 'success');
                    document.getElementById('csvImport').value = '';
                },
                error: (error) => {
                    this.showToast(`CSV parsing error: ${error.message}`, 'error');
                }
            });
        } catch (error) {
            this.showToast(`Import error: ${error.message}`, 'error');
        }
    }

    importFromJSON() {
        const jsonText = document.getElementById('jsonImport').value.trim();
        if (!jsonText) {
            this.showToast('Please enter JSON data', 'error');
            return;
        }

        try {
            const newData = JSON.parse(jsonText);

            if (!Array.isArray(newData)) {
                this.showToast('JSON must be an array', 'error');
                return;
            }

            const validItems = newData.filter(item =>
                item && typeof item === 'object' && item.text && item.label
            ).map(item => ({
                text: String(item.text).trim(),
                label: String(item.label).trim()
            }));

            if (validItems.length === 0) {
                this.showToast('No valid data in JSON', 'error');
                return;
            }

            this.dataset = [...this.dataset, ...validItems];
            this.renderTable();
            this.updateStats();
            this.showToast(`${validItems.length} samples imported successfully`, 'success');
            document.getElementById('jsonImport').value = '';
        } catch (error) {
            this.showToast(`JSON parsing error: ${error.message}`, 'error');
        }
    }

    copyCSV() {
        const csv = document.getElementById('csvExport').value;
        navigator.clipboard.writeText(csv).then(() => {
            this.showToast('CSV copied to clipboard', 'success');
        }).catch(() => {
            this.showToast('Error copying CSV', 'error');
        });
    }

    downloadCSV() {
        const csv = document.getElementById('csvExport').value;
        if (!csv.trim()) {
            this.showToast('No data to download', 'error');
            return;
        }

        const blob = new Blob(['\ufeff' + csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dataset_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('CSV downloaded successfully', 'success');
    }

    copyJSON() {
        const json = document.getElementById('jsonExport').value;
        navigator.clipboard.writeText(json).then(() => {
            this.showToast('JSON copied to clipboard', 'success');
        }).catch(() => {
            this.showToast('Error copying JSON', 'error');
        });
    }

    downloadJSON() {
        const json = document.getElementById('jsonExport').value;
        if (!json.trim()) {
            this.showToast('No data to download', 'error');
            return;
        }

        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dataset_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        this.showToast('JSON downloaded successfully', 'success');
    }

    openAnalysisModal() {
        const validData = this.dataset.filter(d => d.text !== '' && d.label !== '');

        if (validData.length === 0) {
            this.showToast('No data to analyze', 'error');
            return;
        }

        const labels = validData.map(d => d.label.toLowerCase());
        const uniqueLabels = [...new Set(labels)];

        const labelCounts = {};
        labels.forEach(label => {
            labelCounts[label] = (labelCounts[label] || 0) + 1;
        });

        const textLengths = validData.map(d => d.text.length);
        const avgLength = Math.round(textLengths.reduce((a, b) => a + b, 0) / textLengths.length);
        const maxLength = Math.max(...textLengths);
        const minLength = Math.min(...textLengths);

        const content = document.getElementById('analysisContent');
        const fragment = document.createDocumentFragment();

        let html = `
            <div style="margin: 20px 0;">
                <h3 style="color: var(--accent); margin-bottom: 20px;">
                    <i class="fas fa-chart-bar"></i> Comprehensive Data Analysis
                </h3>
                
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 30px;">
                    <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px;">
                        <h4 style="color: var(--accent); margin-bottom: 10px;">General Statistics</h4>
                        <p>Total samples: <strong>${validData.length}</strong></p>
                        <p>Number of classes: <strong>${uniqueLabels.length}</strong></p>
                        <p>Average text length: <strong>${avgLength} characters</strong></p>
                        <p>Maximum length: <strong>${maxLength} characters</strong></p>
                        <p>Minimum length: <strong>${minLength} characters</strong></p>
                    </div>
                    
                    <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px;">
                        <h4 style="color: var(--accent); margin-bottom: 10px;">Class Distribution</h4>
        `;

        Object.entries(labelCounts).forEach(([label, count]) => {
            const percentage = ((count / validData.length) * 100).toFixed(1);
            html += `
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>${this.escapeHtml(label)}</span>
                        <span>${count} (${percentage}%)</span>
                    </div>
                    <div style="height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; margin-top: 5px;">
                        <div style="height: 100%; width: ${percentage}%; background: var(--accent); border-radius: 4px;"></div>
                    </div>
                </div>
            `;
        });

        html += `
                    </div>
                </div>
                
                <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px;">
                    <h4 style="color: var(--accent); margin-bottom: 15px;">Training Recommendations</h4>
        `;

        const recommendations = [];

        if (validData.length < 100) {
            recommendations.push('📊 <strong>Small dataset:</strong> Collect at least 100 samples for better results.');
        }

        if (uniqueLabels.length < 2) {
            recommendations.push('🏷️ <strong>Single label:</strong> Classification requires at least 2 classes.');
        }

        const maxCount = Math.max(...Object.values(labelCounts));
        const minCount = Math.min(...Object.values(labelCounts));
        const imbalanceRatio = maxCount / minCount;

        if (imbalanceRatio > 5) {
            recommendations.push(`⚖️ <strong>Imbalanced data:</strong> Ratio ${imbalanceRatio.toFixed(1)}:1. 
                Use F1-Score as the evaluation metric.`);
        }

        if (avgLength > 200) {
            recommendations.push('📏 <strong>Long texts:</strong> Consider increasing max_length to 512.');
        }

        if (recommendations.length === 0) {
            recommendations.push('✅ <strong>Excellent data quality!</strong> Ready to start training.');
        }

        recommendations.forEach(rec => {
            html += `<div style="margin-bottom: 15px; padding: 10px; background: rgba(76, 201, 240, 0.1); border-radius: 8px;">${rec}</div>`;
        });

        html += `</div></div>`;

        content.innerHTML = html;
        document.getElementById('analysisModal').classList.add('active');
    }

    closeAnalysisModal() {
        document.getElementById('analysisModal').classList.remove('active');
    }

    loadSampleData() {
        this.addNewRow('This product is excellent!', 'positive');
        this.addNewRow('Not satisfied at all', 'negative');
    }

    showToast(message, type = 'info') {
        const colors = {
            success: '#2a9d8f',
            error: '#f72585',
            warning: '#f8961e',
            info: '#4361ee'
        };

        Toastify({
            text: message,
            duration: 3000,
            gravity: "top",
            position: "left",
            backgroundColor: colors[type] || colors.info,
            stopOnFocus: true
        }).showToast();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new HFCodeGenerator();
});
