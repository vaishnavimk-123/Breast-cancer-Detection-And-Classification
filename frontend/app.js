
/**
 * OncoVision AI - Breast Cancer CDSS Frontend
 * Educational/research prototype only. Not for clinical diagnosis.
 */

let appState = {
    metadata: null,
    featureDefinitions: {},
    selectedFeatures: [],
    presetCases: [],
    featureStats: {},
    activeTab: 'single',
    inputMode: 'quick10',
    lastResult: null
};

document.addEventListener('DOMContentLoaded', async () => {
    setupTabNavigation();
    setupInputModeToggle();
    await fetchAppMetadata();
    setupEventListeners();
});

function setupTabNavigation() {
    const tabs = document.querySelectorAll('.nav-tab-btn');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const targetId = tab.getAttribute('data-target');

            document.querySelectorAll('.tab-content-section').forEach(section => {
                section.classList.add('hidden');
            });

            document.getElementById(targetId)?.classList.remove('hidden');
            appState.activeTab = targetId;
        });
    });
}

function setupInputModeToggle() {
    const btnQuick = document.getElementById('btn-mode-quick');
    const btnAll = document.getElementById('btn-mode-all');
    const quickContainer = document.getElementById('quick-features-container');
    const allContainer = document.getElementById('all-features-container');

    if (!btnQuick || !btnAll || !quickContainer || !allContainer) return;

    function setMode(mode) {
        appState.inputMode = mode;

        const quickActive = mode === 'quick10';

        btnQuick.className = quickActive
            ? 'px-4 py-2 text-sm font-semibold rounded-lg bg-sky-600 text-white shadow-sm'
            : 'px-4 py-2 text-sm font-medium rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200';

        btnAll.className = !quickActive
            ? 'px-4 py-2 text-sm font-semibold rounded-lg bg-sky-600 text-white shadow-sm'
            : 'px-4 py-2 text-sm font-medium rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200';

        quickContainer.classList.toggle('hidden', !quickActive);
        allContainer.classList.toggle('hidden', quickActive);

        // Only visible inputs participate in validation and submission.
        quickContainer.querySelectorAll('input').forEach(input => {
            input.disabled = !quickActive;
        });

        allContainer.querySelectorAll('input').forEach(input => {
            input.disabled = quickActive;
        });
    }

    btnQuick.addEventListener('click', () => setMode('quick10'));
    btnAll.addEventListener('click', () => setMode('all30'));

    setMode('quick10');
}

async function fetchAppMetadata() {
    try {
        const [metaRes, statsRes] = await Promise.all([
            fetch('/api/metadata'),
            fetch('/api/feature-stats')
        ]);

        if (!metaRes.ok) throw new Error('Failed to load model metadata');
        if (!statsRes.ok) throw new Error('Failed to load feature statistics');

        const metaData = await metaRes.json();
        const statsData = await statsRes.json();

        appState.metadata = metaData.metadata;
        appState.featureDefinitions = metaData.feature_definitions || {};
        appState.selectedFeatures = metaData.selected_features || [];
        appState.presetCases = metaData.preset_cases || [];
        appState.featureStats = statsData || {};

        renderPresetButtons();
        renderInputForms();
        renderAnalyticsTab();
        updateModelStatusHeader();

    } catch (err) {
        console.error('Initialization error:', err);
        showToast('Error loading application metadata: ' + err.message, 'error');
    }
}

function updateModelStatusHeader() {
    if (!appState.metadata?.metrics) return;

    const metrics = appState.metadata.metrics;
    const element = document.getElementById('model-status-badge');

    if (!element) return;

    element.textContent =
        `Research Model Loaded | Test Accuracy ${(metrics.accuracy * 100).toFixed(1)}% | ROC-AUC ${Number(metrics.roc_auc).toFixed(3)}`;
}

function renderPresetButtons() {
    const presetSelect = document.getElementById('preset-case-select');
    if (!presetSelect) return;

    presetSelect.innerHTML = '<option value="">-- Select an Example Case --</option>';

    appState.presetCases.forEach((preset, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${preset.name} (${preset.expected})`;
        presetSelect.appendChild(option);
    });

    presetSelect.onchange = event => {
        const index = event.target.value;
        if (index !== '') {
            loadPresetCase(appState.presetCases[Number(index)]);
        }
    };
}

function loadPresetCase(preset) {
    if (!preset?.data) return;

    const patientId = document.getElementById('patient-id-input');
    if (patientId) patientId.value = preset.id || '';

    Object.entries(preset.data).forEach(([feature, value]) => {
        document.querySelectorAll(`input[name="${feature}"]`).forEach(input => {
            input.value = value;
            input.classList.add('bg-sky-50');

            setTimeout(() => {
                input.classList.remove('bg-sky-50');
            }, 600);
        });
    });

    const banner = document.getElementById('preset-info-banner');

    if (banner) {
        banner.classList.remove('hidden');
        banner.textContent = `${preset.name}: ${preset.summary || 'Illustrative research example.'}`;
    }

    showToast(`Loaded ${preset.name}`, 'info');
}

function renderInputForms() {
    renderQuick10Form();
    renderAll30Form();
    setupInputModeToggle();
}

function renderQuick10Form() {
    const container = document.getElementById('quick-features-container');
    if (!container) return;

    let html = `
        <div class="p-4 bg-sky-50/70 border border-sky-200 rounded-lg mb-4 text-xs text-slate-700">
            <strong>Selected 10 Model Features</strong>
            <p class="mt-1">These features were selected during model training. This is a research demonstration, not a validated clinical assessment.</p>
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
    `;

    appState.selectedFeatures.forEach(feature => {
        const meta = appState.featureDefinitions[feature] || {};
        const stats = appState.featureStats[feature] || {};
        const placeholder = stats.median ?? '0.0';

        html += `
            <div class="form-input-group bg-slate-50/60 p-3 rounded-lg border border-slate-200">
                <label class="text-xs font-semibold text-slate-800 block mb-1" for="q_${feature}">
                    ${escapeHTML(meta.label || feature)}
                </label>
                <input
                    type="number"
                    step="any"
                    id="q_${feature}"
                    name="${feature}"
                    placeholder="e.g. ${escapeHTML(String(placeholder))}"
                    class="form-control-clinical text-sm font-mono"
                    required
                />
                <div class="text-[10px] text-slate-600 mt-1">
                    ${escapeHTML(meta.unit || '')} | ${escapeHTML(meta.normal_range || 'Dataset reference')}
                </div>
            </div>
        `;
    });

    container.innerHTML = html + '</div>';
}

function renderAll30Form() {
    const container = document.getElementById('all-features-container');
    if (!container) return;

    const groups = {
        mean: { title: 'Means', features: [] },
        se: { title: 'Standard Errors', features: [] },
        worst: { title: 'Worst Values', features: [] }
    };

    Object.entries(appState.featureDefinitions).forEach(([feature, definition]) => {
        const group = definition.group || 'mean';
        if (groups[group]) groups[group].features.push(feature);
    });

    let html = `
        <div class="mb-4 flex border-b border-slate-200">
            <button type="button" class="subtab-btn px-4 py-2 text-xs font-semibold border-b-2 border-sky-600 text-sky-600" data-subtab="subtab-mean">Means</button>
            <button type="button" class="subtab-btn px-4 py-2 text-xs font-medium text-slate-500 border-b-2 border-transparent" data-subtab="subtab-se">Standard Errors</button>
            <button type="button" class="subtab-btn px-4 py-2 text-xs font-medium text-slate-500 border-b-2 border-transparent" data-subtab="subtab-worst">Worst Values</button>
        </div>
    `;

    Object.entries(groups).forEach(([groupKey, group], index) => {
        html += `
            <div id="subtab-${groupKey}" class="subtab-content ${index !== 0 ? 'hidden' : ''} grid grid-cols-1 sm:grid-cols-2 gap-4">
        `;

        group.features.forEach(feature => {
            const meta = appState.featureDefinitions[feature] || {};
            const stats = appState.featureStats[feature] || {};
            const selected = appState.selectedFeatures.includes(feature);

            html += `
                <div class="form-input-group bg-slate-50/60 p-3 rounded-lg border ${selected ? 'border-sky-300' : 'border-slate-200'}">
                    <label class="text-xs font-semibold text-slate-800 block mb-1" for="a_${feature}">
                        ${escapeHTML(meta.label || feature)}
                        ${selected ? '<span class="text-sky-700 text-[9px]">Top 10</span>' : ''}
                    </label>
                    <input
                        type="number"
                        step="any"
                        id="a_${feature}"
                        name="${feature}"
                        placeholder="e.g. ${escapeHTML(String(stats.median ?? '0.0'))}"
                        class="form-control-clinical text-sm font-mono"
                    />
                    <div class="text-[10px] text-slate-600 mt-1">
                        ${escapeHTML(meta.unit || '')} | ${escapeHTML(meta.normal_range || 'Dataset reference')}
                    </div>
                </div>
            `;
        });

        html += '</div>';
    });

    container.innerHTML = html;

    const buttons = container.querySelectorAll('.subtab-btn');

    buttons.forEach(button => {
        button.addEventListener('click', () => {
            buttons.forEach(item => {
                item.className = 'subtab-btn px-4 py-2 text-xs font-medium text-slate-500 border-b-2 border-transparent';
            });

            button.className = 'subtab-btn px-4 py-2 text-xs font-semibold border-b-2 border-sky-600 text-sky-600';

            container.querySelectorAll('.subtab-content').forEach(section => {
                section.classList.add('hidden');
            });

            document.getElementById(button.dataset.subtab)?.classList.remove('hidden');
        });
    });
}

function setupEventListeners() {
    const form = document.getElementById('diagnosis-form');

    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }

    const resetButton = document.getElementById('btn-reset-form');

    if (resetButton && form) {
        resetButton.addEventListener('click', () => {
            form.reset();
            setInputModeAfterReset();

            const presetSelect = document.getElementById('preset-case-select');
            if (presetSelect) presetSelect.value = '';

            document.getElementById('preset-info-banner')?.classList.add('hidden');
            document.getElementById('empty-results-card')?.classList.remove('hidden');
            document.getElementById('diagnosis-result-card')?.classList.add('hidden');

            appState.lastResult = null;
            showToast('Form reset', 'info');
        });
    }

    setupBatchUploadListeners();

    // Retraining is disabled by the backend for deployment.
    const retrainButton = document.getElementById('btn-retrain-model');

    if (retrainButton) {
        retrainButton.disabled = true;
        retrainButton.title = 'Model retraining is disabled in the deployed application.';
        retrainButton.textContent = 'Retraining Disabled';
    }

    document.getElementById('btn-print-report')?.addEventListener('click', () => {
        window.print();
    });
}

function setInputModeAfterReset() {
    const quickButton = document.getElementById('btn-mode-quick');
    if (quickButton) quickButton.click();
}

async function handleFormSubmit(event) {
    event.preventDefault();

    const form = document.getElementById('diagnosis-form');
    const patientId = document.getElementById('patient-id-input')?.value.trim() || undefined;

    const featureValues = {};

    new FormData(form).forEach((value, key) => {
        if (key && String(value).trim() !== '') {
            const number = Number(value);

            if (!Number.isFinite(number)) {
                featureValues[key] = null;
            } else {
                featureValues[key] = number;
            }
        }
    });

    if (Object.keys(featureValues).length === 0) {
        showToast('Please enter measurements or load an example case.', 'error');
        return;
    }

    if (Object.values(featureValues).some(value => value === null)) {
        showToast('Please enter valid numeric measurements.', 'error');
        return;
    }

    const submitButton = document.getElementById('btn-submit-prediction');
    if (!submitButton) return;

    submitButton.disabled = true;
    submitButton.textContent = 'Processing research example...';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patient_id: patientId,
                features: featureValues
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Prediction request failed');
        }

        appState.lastResult = result;
        renderPredictionResults(result);
        showToast('Model result received', 'success');

    } catch (error) {
        console.error('Prediction failed:', error);
        showToast('Request error: ' + error.message, 'error');

    } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'Run Research Model';
    }
}

function renderPredictionResults(result) {
    const emptyCard = document.getElementById('empty-results-card');
    const resultCard = document.getElementById('diagnosis-result-card');

    emptyCard?.classList.add('hidden');
    resultCard?.classList.remove('hidden');

    const isMalignant = result.prediction_code === 1;
    const probabilities = result.probabilities || {};
    const risk = result.risk_assessment || {};

    const header = document.getElementById('result-header-box');
    const title = document.getElementById('result-diagnosis-title');
    const riskBadge = document.getElementById('result-risk-badge');

    const malignantPercent = Number(probabilities.malignant_percentage || 0);
    const benignPercent = Number(probabilities.benign_percentage || 0);

    document.getElementById('result-case-id').textContent = result.case_id || 'N/A';
    document.getElementById('result-timestamp').textContent = result.timestamp || 'N/A';

    if (header && title) {
        header.className = isMalignant
            ? 'p-5 rounded-t-xl bg-rose-700 text-white'
            : 'p-5 rounded-t-xl bg-emerald-700 text-white';

        title.textContent = isMalignant
            ? 'Model Prediction: Malignant Class'
            : 'Model Prediction: Benign Class';
    }

    if (riskBadge) {
        riskBadge.className = 'inline-flex items-center px-3 py-1 rounded-full text-xs font-bold border';
        riskBadge.textContent = risk.risk_category || 'Model output';
    }

    document.getElementById('result-prob-malignant').textContent = `${malignantPercent}%`;
    document.getElementById('result-prob-benign').textContent = `${benignPercent}%`;

    const meter = document.getElementById('result-meter-fill');

    if (meter) {
        meter.style.width = `${Math.max(0, Math.min(100, malignantPercent))}%`;
        meter.className = 'meter-fill ' + (isMalignant ? 'bg-rose-600' : 'bg-emerald-500');
    }

    const recommendation = document.getElementById('result-recommendation-box');

    if (recommendation) {
        recommendation.textContent =
            risk.recommendation || 'This is an educational model output and must not be used to make medical decisions.';
    }

    const warningBox = document.getElementById('result-warnings-box');
    const warnings = result.data_quality?.validation_warnings || [];

    if (warningBox) {
        warningBox.classList.toggle('hidden', warnings.length === 0);
        warningBox.textContent = warnings.join(' | ');
    }

    renderExplainabilityTable(result.explainability?.key_biomarkers || []);
}

function renderExplainabilityTable(biomarkers) {
    const tableBody = document.getElementById('result-xai-table-body');
    if (!tableBody) return;

    tableBody.innerHTML = '';

    biomarkers.forEach(item => {
        const row = document.createElement('tr');
        row.className = 'border-b border-slate-100 text-xs';

        const values = [
            `${item.label || ''} (${item.feature || ''})`,
            `${item.patient_value ?? ''} ${item.unit || ''}`,
            item.benign_median ?? '',
            item.malignant_median ?? '',
            `${item.norm_value ?? 0}%`,
            item.is_malignant_lean ? 'Closer to malignant-group median' : 'Closer to benign-group median'
        ];

        values.forEach(value => {
            const cell = document.createElement('td');
            cell.className = 'py-2 px-3';
            cell.textContent = String(value);
            row.appendChild(cell);
        });

        tableBody.appendChild(row);
    });
}

function setupBatchUploadListeners() {
    const dropArea = document.getElementById('batch-drop-area');
    const fileInput = document.getElementById('batch-file-input');

    if (dropArea && fileInput) {
        dropArea.addEventListener('click', () => fileInput.click());

        ['dragenter', 'dragover'].forEach(name => {
            dropArea.addEventListener(name, event => {
                event.preventDefault();
                dropArea.classList.add('border-sky-500', 'bg-sky-50');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            dropArea.addEventListener(name, event => {
                event.preventDefault();
                dropArea.classList.remove('border-sky-500', 'bg-sky-50');
            });
        });

        dropArea.addEventListener('drop', event => {
            const file = event.dataTransfer.files[0];
            if (file) processBatchFile(file);
        });

        fileInput.addEventListener('change', event => {
            const file = event.target.files[0];
            if (file) processBatchFile(file);
        });
    }

    document.getElementById('btn-load-demo-batch')?.addEventListener('click', loadDemoBatch);
}

async function processBatchFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showToast('Please upload a CSV file.', 'error');
        return;
    }

    const loader = document.getElementById('batch-loader');
    loader?.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/predict-batch', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Batch processing failed');
        }

        renderBatchResults(data);
        showToast(`Processed ${data.total_screened} example records`, 'success');

    } catch (error) {
        console.error('Batch processing error:', error);
        showToast(error.message, 'error');

    } finally {
        loader?.classList.add('hidden');
    }
}

async function loadDemoBatch() {
    try {
        const response = await fetch('/api/sample-csv');

        if (!response.ok) {
            throw new Error('Could not load sample CSV');
        }

        const blob = await response.blob();
        const file = new File([blob], 'demo_wisconsin_batch.csv', {
            type: 'text/csv'
        });

        await processBatchFile(file);

    } catch (error) {
        showToast(error.message, 'error');
    }
}

function renderBatchResults(data) {
    document.getElementById('batch-summary-section')?.classList.remove('hidden');
    document.getElementById('batch-table-section')?.classList.remove('hidden');

    const summary = data.summary || {};

    document.getElementById('batch-count-total').textContent = data.total_screened ?? 0;
    document.getElementById('batch-count-malignant').textContent = summary.malignant_count ?? 0;
    document.getElementById('batch-count-benign').textContent = summary.benign_count ?? 0;
    document.getElementById('batch-count-indeterminate').textContent = summary.indeterminate_count ?? 0;

    const tbody = document.getElementById('batch-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';

    (data.patients || []).forEach(patient => {
        const row = document.createElement('tr');
        row.className = 'border-b border-slate-100 text-xs';

        const values = [
            patient.patient_id || 'N/A',
            patient.diagnosis || 'N/A',
            `${patient.malignancy_percentage ?? 0}%`,
            patient.risk_category || 'N/A',
            patient.requires_urgent_review
                ? 'HIGH MODEL PROBABILITY'
                : patient.is_borderline
                    ? 'INTERMEDIATE MODEL PROBABILITY'
                    : 'MODEL OUTPUT'
        ];

        values.forEach(value => {
            const cell = document.createElement('td');
            cell.className = 'py-2.5 px-3';
            cell.textContent = String(value);
            row.appendChild(cell);
        });

        tbody.appendChild(row);
    });
}

function renderAnalyticsTab() {
    if (!appState.metadata?.metrics) return;

    const metrics = appState.metadata.metrics;
    const matrix = metrics.confusion_matrix || {};

    const setText = (id, value) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    };

    setText('metric-accuracy', `${(metrics.accuracy * 100).toFixed(2)}%`);
    setText('metric-auc', Number(metrics.roc_auc).toFixed(4));
    setText('metric-sensitivity', `${(metrics.sensitivity_recall * 100).toFixed(1)}%`);
    setText('metric-specificity', `${(metrics.specificity * 100).toFixed(1)}%`);
    setText('metric-precision', `${(metrics.precision * 100).toFixed(1)}%`);
    setText('metric-f1', `${(metrics.f1_score * 100).toFixed(1)}%`);

    setText('cm-tn', matrix.true_negative ?? 0);
    setText('cm-fp', matrix.false_positive ?? 0);
    setText('cm-fn', matrix.false_negative ?? 0);
    setText('cm-tp', matrix.true_positive ?? 0);

    const rankingBody = document.getElementById('feature-ranking-tbody');
    if (!rankingBody) return;

    rankingBody.innerHTML = '';

    (appState.metadata.feature_ranking || []).forEach((feature, index) => {
        const definition = appState.featureDefinitions[feature.feature] || {};
        const row = document.createElement('tr');

        [
            `#${index + 1}`,
            `${definition.label || feature.feature} (${feature.feature})`,
            Number(feature.f_score).toFixed(1),
            feature.selected ? 'Selected by model' : 'Not selected'
        ].forEach(value => {
            const cell = document.createElement('td');
            cell.className = 'py-2 px-3 text-xs';
            cell.textContent = String(value);
            row.appendChild(cell);
        });

        rankingBody.appendChild(row);
    });
}

function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, character => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    })[character]);
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');

    const classes = {
        success: 'bg-emerald-600 text-white',
        error: 'bg-rose-600 text-white',
        info: 'bg-sky-700 text-white'
    };

    toast.className =
        `fixed bottom-5 right-5 z-50 px-4 py-3 rounded-lg shadow-xl text-xs font-semibold ${classes[type] || classes.info}`;

    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}
