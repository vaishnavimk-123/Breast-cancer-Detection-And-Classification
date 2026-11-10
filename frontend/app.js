/**
 * OncoVision AI - Breast Cancer Clinical Decision Support Application Frontend
 */

let appState = {
    metadata: null,
    featureDefinitions: {},
    selectedFeatures: [],
    presetCases: [],
    featureStats: {},
    activeTab: 'single',
    inputMode: 'quick10', // 'quick10' or 'all30'
    lastResult: null
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
    setupTabNavigation();
    setupInputModeToggle();
    await fetchAppMetadata();
    setupEventListeners();
});

// Tab Navigation
function setupTabNavigation() {
    const tabs = document.querySelectorAll('.nav-tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const targetId = tab.getAttribute('data-target');
            document.querySelectorAll('.tab-content-section').forEach(sec => {
                sec.classList.add('hidden');
            });
            const targetSec = document.getElementById(targetId);
            if (targetSec) targetSec.classList.remove('hidden');
        });
    });
}

// Input Mode Toggle (Quick 10 vs All 30)
function setupInputModeToggle() {
    const btnQuick = document.getElementById('btn-mode-quick');
    const btnAll = document.getElementById('btn-mode-all');
    const quickContainer = document.getElementById('quick-features-container');
    const allContainer = document.getElementById('all-features-container');

    if (!btnQuick || !btnAll) return;

    btnQuick.addEventListener('click', () => {
        appState.inputMode = 'quick10';
        btnQuick.className = 'px-4 py-2 text-sm font-semibold rounded-lg bg-sky-600 text-white shadow-sm';
        btnAll.className = 'px-4 py-2 text-sm font-medium rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200';
        quickContainer.classList.remove('hidden');
        allContainer.classList.add('hidden');
    });

    btnAll.addEventListener('click', () => {
        appState.inputMode = 'all30';
        btnAll.className = 'px-4 py-2 text-sm font-semibold rounded-lg bg-sky-600 text-white shadow-sm';
        btnQuick.className = 'px-4 py-2 text-sm font-medium rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200';
        allContainer.classList.remove('hidden');
        quickContainer.classList.add('hidden');
    });
}

// Fetch Metadata & Definitions
async function fetchAppMetadata() {
    try {
        const [metaRes, statsRes] = await Promise.all([
            fetch('/api/metadata'),
            fetch('/api/feature-stats')
        ]);

        if (!metaRes.ok) throw new Error('Failed to load model metadata');

        const metaData = await metaRes.json();
        const statsData = await statsRes.json();

        appState.metadata = metaData.metadata;
        appState.featureDefinitions = metaData.feature_definitions;
        appState.selectedFeatures = metaData.selected_features;
        appState.presetCases = metaData.preset_cases;
        appState.featureStats = statsData;

        renderPresetButtons();
        renderInputForms();
        renderAnalyticsTab();
        updateModelStatusHeader();

    } catch (err) {
        console.error('Initialization error:', err);
        showToast('Error loading application metadata: ' + err.message, 'error');
    }
}

// Update Top Bar Model Status
function updateModelStatusHeader() {
    if (!appState.metadata || !appState.metadata.metrics) return;
    const m = appState.metadata.metrics;
    const el = document.getElementById('model-status-badge');
    if (el) {
        el.innerHTML = `
            <span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                <span class="w-2 h-2 mr-1.5 bg-emerald-500 rounded-full animate-pulse"></span>
                Model Active: Test Accuracy ${(m.accuracy * 100).toFixed(1)}% | ROC-AUC ${(m.roc_auc).toFixed(3)}
            </span>
        `;
    }
}

// Render Presets Dropdown / Buttons
function renderPresetButtons() {
    const presetSelect = document.getElementById('preset-case-select');
    if (!presetSelect) return;

    presetSelect.innerHTML = '<option value="">-- Select a Clinical Case Preset --</option>';
    appState.presetCases.forEach((c, idx) => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = `${c.name} (${c.expected})`;
        presetSelect.appendChild(opt);
    });

    presetSelect.addEventListener('change', (e) => {
        const idx = e.target.value;
        if (idx !== "") {
            loadPresetCase(appState.presetCases[parseInt(idx)]);
        }
    });
}

// Load Preset Case into Form
function loadPresetCase(preset) {
    if (!preset || !preset.data) return;

    // Fill patient ID
    const pidInput = document.getElementById('patient-id-input');
    if (pidInput) pidInput.value = preset.id;

    // Fill all inputs across both quick and all tabs
    Object.entries(preset.data).forEach(([feat, val]) => {
        const inputs = document.querySelectorAll(`input[name="${feat}"]`);
        inputs.forEach(input => {
            input.value = val;
            // flash highlight
            input.classList.add('bg-sky-50');
            setTimeout(() => input.classList.remove('bg-sky-50'), 600);
        });
    });

    // Show summary badge
    const banner = document.getElementById('preset-info-banner');
    if (banner) {
        banner.classList.remove('hidden');
        banner.innerHTML = `
            <div class="flex items-start justify-between bg-sky-50 border border-sky-200 text-sky-900 rounded-lg p-3 text-xs">
                <div>
                    <span class="font-bold text-sky-950">${preset.name}</span>
                    <p class="mt-0.5 text-slate-600">${preset.summary}</p>
                </div>
                <button type="button" onclick="document.getElementById('preset-info-banner').classList.add('hidden')" class="text-sky-500 hover:text-sky-800 ml-2">✕</button>
            </div>
        `;
    }

    showToast(`Loaded ${preset.name}`, 'info');
}

// Render Input Forms
function renderInputForms() {
    renderQuick10Form();
    renderAll30Form();
}

// Render Quick 10 Form
function renderQuick10Form() {
    const container = document.getElementById('quick-features-container');
    if (!container) return;

    let html = `
        <div class="p-4 bg-sky-50/70 border border-sky-200 rounded-lg mb-4 text-xs text-slate-700 flex items-start space-x-2">
            <span class="text-sky-600 text-base">ℹ️</span>
            <div>
                <strong>Top 10 High-Impact Diagnostic Biomarkers:</strong>
                These 10 cytology features were identified via ANOVA F-test feature selection as the most discriminating indicators of breast cancer malignancy. Missing secondary measurements are automatically imputed using population baselines.
            </div>
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
    `;

    appState.selectedFeatures.forEach(feat => {
        const meta = appState.featureDefinitions[feat] || {};
        const stats = appState.featureStats[feat] || {};
        const placeholder = stats.median !== undefined ? stats.median : '0.0';

        html += `
            <div class="form-input-group bg-slate-50/60 p-3 rounded-lg border border-slate-200 hover:border-sky-300 transition">
                <div class="flex justify-between items-baseline mb-1">
                    <label class="text-xs font-semibold text-slate-800 flex items-center" for="q_${feat}">
                        ${meta.label || feat}
                        <span class="ml-1 text-slate-400 text-[10px] clinical-tooltip" title="${meta.description || ''}">ⓘ</span>
                    </label>
                    <span class="text-[10px] text-slate-600 font-mono">${meta.unit || ''}</span>
                </div>
                <input 
                    type="number" 
                    step="any" 
                    id="q_${feat}" 
                    name="${feat}" 
                    placeholder="e.g. ${placeholder}" 
                    class="form-control-clinical text-sm font-mono"
                    required
                />
                <div class="flex justify-between items-center mt-1 text-[10px] text-slate-600">
                    <span>Normal Range: ${meta.normal_range || 'N/A'}</span>
                    <span class="text-slate-600">Benign med: ${stats.benign_median || ''}</span>
                </div>
            </div>
        `;
    });

    html += `</div>`;
    container.innerHTML = html;
}

// Render All 30 Form with Sub-tabs (Mean, SE, Worst)
function renderAll30Form() {
    const container = document.getElementById('all-features-container');
    if (!container) return;

    const groups = {
        mean: { title: "Cell Nucleus Means (10)", features: [] },
        se: { title: "Standard Error Variances (10)", features: [] },
        worst: { title: "Extreme / Worst Values (10)", features: [] }
    };

    Object.keys(appState.featureDefinitions).forEach(feat => {
        const def = appState.featureDefinitions[feat];
        const grp = def.group || 'mean';
        if (groups[grp]) groups[grp].features.push(feat);
    });

    let html = `
        <div class="mb-4 flex border-b border-slate-200">
            <button type="button" class="subtab-btn px-4 py-2 text-xs font-semibold border-b-2 border-sky-600 text-sky-600" data-subtab="subtab-mean">Means</button>
            <button type="button" class="subtab-btn px-4 py-2 text-xs font-medium text-slate-500 hover:text-slate-800 border-b-2 border-transparent" data-subtab="subtab-se">Standard Errors (SE)</button>
            <button type="button" class="subtab-btn px-4 py-2 text-xs font-medium text-slate-500 hover:text-slate-800 border-b-2 border-transparent" data-subtab="subtab-worst">Worst / Extreme</button>
        </div>
    `;

    Object.entries(groups).forEach(([grpKey, grpObj], idx) => {
        const isHidden = idx !== 0 ? 'hidden' : '';
        html += `<div id="subtab-${grpKey}" class="subtab-content ${isHidden} grid grid-cols-1 sm:grid-cols-2 gap-4">`;

        grpObj.features.forEach(feat => {
            const meta = appState.featureDefinitions[feat] || {};
            const stats = appState.featureStats[feat] || {};
            const isTop10 = appState.selectedFeatures.includes(feat);
            const placeholder = stats.median !== undefined ? stats.median : '0.0';

            html += `
                <div class="form-input-group bg-slate-50/60 p-3 rounded-lg border ${isTop10 ? 'border-sky-300 bg-sky-50/20' : 'border-slate-200'}">
                    <div class="flex justify-between items-baseline mb-1">
                        <label class="text-xs font-semibold text-slate-800 flex items-center" for="a_${feat}">
                            ${meta.label || feat}
                            ${isTop10 ? '<span class="ml-1 text-[9px] bg-sky-100 text-sky-700 px-1 rounded font-bold">Top 10</span>' : ''}
                            <span class="ml-1 text-slate-400 text-[10px] clinical-tooltip" title="${meta.description || ''}">ⓘ</span>
                        </label>
                        <span class="text-[10px] text-slate-600 font-mono">${meta.unit || ''}</span>
                    </div>
                    <input 
                        type="number" 
                        step="any" 
                        id="a_${feat}" 
                        name="${feat}" 
                        placeholder="e.g. ${placeholder}" 
                        class="form-control-clinical text-sm font-mono"
                    />
                    <div class="flex justify-between items-center mt-1 text-[10px] text-slate-600">
                        <span>Range: ${meta.normal_range || 'N/A'}</span>
                        <span class="text-slate-600">Med: ${stats.median || ''}</span>
                    </div>
                </div>
            `;
        });

        html += `</div>`;
    });

    container.innerHTML = html;

    // Subtab event listeners
    const subtabBtns = container.querySelectorAll('.subtab-btn');
    subtabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            subtabBtns.forEach(b => {
                b.className = 'subtab-btn px-4 py-2 text-xs font-medium text-slate-500 hover:text-slate-800 border-b-2 border-transparent';
            });
            btn.className = 'subtab-btn px-4 py-2 text-xs font-semibold border-b-2 border-sky-600 text-sky-600';

            const target = btn.getAttribute('data-subtab');
            container.querySelectorAll('.subtab-content').forEach(c => c.classList.add('hidden'));
            const showSec = document.getElementById(target);
            if (showSec) showSec.classList.remove('hidden');
        });
    });
}

// Setup Event Listeners
function setupEventListeners() {
    // Single Prediction Form Submit
    const form = document.getElementById('diagnosis-form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }

    // Reset Form Button
    const btnReset = document.getElementById('btn-reset-form');
    if (btnReset) {
        btnReset.addEventListener('click', () => {
            form.reset();
            const presetSelect = document.getElementById('preset-case-select');
            if (presetSelect) presetSelect.value = "";
            const banner = document.getElementById('preset-info-banner');
            if (banner) banner.classList.add('hidden');
            document.getElementById('empty-results-card').classList.remove('hidden');
            document.getElementById('diagnosis-result-card').classList.add('hidden');
            showToast('Form reset', 'info');
        });
    }

    // Batch Drag & Drop & Upload
    setupBatchUploadListeners();

    // Model Retrain Button
    const btnRetrain = document.getElementById('btn-retrain-model');
    if (btnRetrain) {
        btnRetrain.addEventListener('click', handleRetrainModel);
    }

    // Print Report Button
    const btnPrint = document.getElementById('btn-print-report');
    if (btnPrint) {
        btnPrint.addEventListener('click', () => window.print());
    }
}

// Handle Form Submission for Single Prediction
async function handleFormSubmit(e) {
    e.preventDefault();

    const pid = document.getElementById('patient-id-input').value.trim() || undefined;
    const form = document.getElementById('diagnosis-form');
    const formData = new FormData(form);

    const featureValues = {};
    for (let [key, val] of formData.entries()) {
        if (key && val !== "") {
            featureValues[key] = parseFloat(val);
        }
    }

    // Check that at least some features were entered
    if (Object.keys(featureValues).length === 0) {
        showToast('Please enter biomarker measurements or select a preset case.', 'error');
        return;
    }

    const submitBtn = document.getElementById('btn-submit-prediction');
    submitBtn.disabled = true;
    submitBtn.innerHTML = `
        <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
        </svg>
        Analyzing Cytology Data...
    `;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patient_id: pid,
                features: featureValues
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Server prediction error');
        }

        const result = await response.json();
        appState.lastResult = result;
        renderPredictionResults(result);
        showToast('Clinical analysis complete', 'success');

    } catch (err) {
        console.error('Prediction failed:', err);
        showToast('Analysis error: ' + err.message, 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `
            <svg class="w-4 h-4 mr-1.5 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Perform Diagnostic Assessment
        `;
    }
}

// Render Single Prediction Results
function renderPredictionResults(res) {
    document.getElementById('empty-results-card').classList.add('hidden');
    const resultCard = document.getElementById('diagnosis-result-card');
    resultCard.classList.remove('hidden');

    const isMalignant = res.prediction_code === 1;
    const probMalignant = res.probabilities.malignant_percentage;
    const probBenign = res.probabilities.benign_percentage;
    const risk = res.risk_assessment;

    // Header badge
    const headerBox = document.getElementById('result-header-box');
    const diagTitle = document.getElementById('result-diagnosis-title');
    const riskBadge = document.getElementById('result-risk-badge');
    const caseIdLabel = document.getElementById('result-case-id');
    const timestampLabel = document.getElementById('result-timestamp');

    caseIdLabel.textContent = res.case_id;
    timestampLabel.textContent = res.timestamp;

    if (isMalignant) {
        headerBox.className = 'p-5 rounded-t-xl bg-gradient-to-r from-rose-700 to-red-800 text-white';
        diagTitle.innerHTML = `
            <div class="flex items-center space-x-2">
                <span class="text-2xl">⚠️</span>
                <div>
                    <h3 class="text-xl font-bold tracking-tight">Malignant Neoplasm Detected</h3>
                    <p class="text-xs text-rose-200 mt-0.5">High cytological pleomorphism & aggressive nuclear atypia pattern</p>
                </div>
            </div>
        `;
    } else {
        headerBox.className = 'p-5 rounded-t-xl bg-gradient-to-r from-emerald-600 to-teal-700 text-white';
        diagTitle.innerHTML = `
            <div class="flex items-center space-x-2">
                <span class="text-2xl">🛡️</span>
                <div>
                    <h3 class="text-xl font-bold tracking-tight">Benign Lesion Profile</h3>
                    <p class="text-xs text-emerald-100 mt-0.5">Uniform cellular boundaries, low concave distortion</p>
                </div>
            </div>
        `;
    }

    // Risk Badge
    let badgeClass = 'bg-emerald-100 text-emerald-800 border-emerald-300';
    if (risk.badge === 'warning') badgeClass = 'bg-amber-100 text-amber-900 border-amber-300';
    if (risk.badge === 'danger') badgeClass = 'bg-rose-100 text-rose-900 border-rose-300';

    riskBadge.className = `inline-flex items-center px-3 py-1 rounded-full text-xs font-bold border ${badgeClass}`;
    riskBadge.textContent = risk.risk_category;

    // Probability Gauge Bar
    const meterFill = document.getElementById('result-meter-fill');
    const pctMalignantText = document.getElementById('result-prob-malignant');
    const pctBenignText = document.getElementById('result-prob-benign');

    pctMalignantText.textContent = `${probMalignant}%`;
    pctBenignText.textContent = `${probBenign}%`;

    meterFill.style.width = `${probMalignant}%`;
    if (isMalignant) {
        meterFill.className = 'meter-fill bg-rose-600';
    } else if (risk.is_borderline) {
        meterFill.className = 'meter-fill bg-amber-500';
    } else {
        meterFill.className = 'meter-fill bg-emerald-500';
    }

    // Recommendation Section
    const recBox = document.getElementById('result-recommendation-box');
    recBox.innerHTML = `
        <div class="text-xs leading-relaxed text-slate-800">
            <strong class="block text-slate-900 font-semibold mb-1">Clinical Protocol Recommendation:</strong>
            ${risk.recommendation}
        </div>
    `;

    // Warnings if any
    const warnBox = document.getElementById('result-warnings-box');
    if (res.data_quality.validation_warnings && res.data_quality.validation_warnings.length > 0) {
        warnBox.classList.remove('hidden');
        warnBox.innerHTML = `
            <div class="p-3 bg-amber-50 border border-amber-200 text-amber-900 rounded-lg text-xs">
                <strong>Data Range Anomaly Alerts:</strong>
                <ul class="list-disc list-inside mt-1 space-y-0.5">
                    ${res.data_quality.validation_warnings.map(w => `<li>${w}</li>`).join('')}
                </ul>
            </div>
        `;
    } else {
        warnBox.classList.add('hidden');
    }

    // Explainable AI (XAI) Biomarker Profile Table
    renderExplainabilityTable(res.explainability.key_biomarkers);
}

// Render XAI Table
function renderExplainabilityTable(biomarkers) {
    const tableBody = document.getElementById('result-xai-table-body');
    if (!tableBody) return;

    let html = '';
    biomarkers.forEach(b => {
        const leanBadge = b.is_malignant_lean
            ? `<span class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-rose-50 text-rose-700 border border-rose-200">Malignant Tendency</span>`
            : `<span class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">Benign Tendency</span>`;

        html += `
            <tr class="border-b border-slate-100 hover:bg-slate-50/70 text-xs transition">
                <td class="py-2 px-3 font-medium text-slate-900">
                    <div>${b.label}</div>
                    <div class="text-[10px] text-slate-400 font-normal">${b.feature}</div>
                </td>
                <td class="py-2 px-3 font-mono font-semibold text-slate-800">
                    ${b.patient_value} <span class="text-[10px] text-slate-400">${b.unit}</span>
                </td>
                <td class="py-2 px-3 font-mono text-slate-500">
                    ${b.benign_median}
                </td>
                <td class="py-2 px-3 font-mono text-slate-500">
                    ${b.malignant_median}
                </td>
                <td class="py-2 px-3">
                    <div class="w-24 bg-slate-200 rounded-full h-1.5 overflow-hidden">
                        <div class="h-1.5 rounded-full ${b.is_malignant_lean ? 'bg-rose-500' : 'bg-emerald-500'}" style="width: ${b.norm_value}%"></div>
                    </div>
                    <span class="text-[9px] text-slate-400">${b.norm_value}% of max</span>
                </td>
                <td class="py-2 px-3">
                    ${leanBadge}
                </td>
            </tr>
        `;
    });

    tableBody.innerHTML = html;
}

// Batch Upload and Screening Setup
function setupBatchUploadListeners() {
    const dropArea = document.getElementById('batch-drop-area');
    const fileInput = document.getElementById('batch-file-input');
    const btnSampleBatch = document.getElementById('btn-load-demo-batch');

    if (dropArea && fileInput) {
        dropArea.addEventListener('click', () => fileInput.click());

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropArea.classList.add('border-sky-500', 'bg-sky-50');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropArea.classList.remove('border-sky-500', 'bg-sky-50');
            }, false);
        });

        dropArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) processBatchFile(files[0]);
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) processBatchFile(e.target.files[0]);
        });
    }

    if (btnSampleBatch) {
        btnSampleBatch.addEventListener('click', loadDemoBatch);
    }
}

// Process Batch CSV Upload
async function processBatchFile(file) {
    if (!file.name.endsWith('.csv')) {
        showToast('Please upload a valid .csv file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const loader = document.getElementById('batch-loader');
    loader.classList.remove('hidden');

    try {
        const response = await fetch('/api/predict-batch', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Batch screening failed');
        }

        const data = await response.json();
        renderBatchResults(data);
        showToast(`Successfully screened ${data.total_screened} patient records`, 'success');

    } catch (err) {
        console.error('Batch screening error:', err);
        showToast('Batch processing error: ' + err.message, 'error');
    } finally {
        loader.classList.add('hidden');
    }
}

// Load Demo Batch (uses preset cases + additional synthetic records)
async function loadDemoBatch() {
    const loader = document.getElementById('batch-loader');
    loader.classList.remove('hidden');

    try {
        // Fetch sample CSV from server
        const res = await fetch('/api/sample-csv');
        if (!res.ok) throw new Error('Could not fetch sample CSV');
        const blob = await res.blob();
        const file = new File([blob], "demo_wisconsin_batch.csv", { type: "text/csv" });
        await processBatchFile(file);
    } catch (err) {
        console.error('Demo batch error:', err);
        showToast('Failed to load demo batch: ' + err.message, 'error');
        loader.classList.add('hidden');
    }
}

// Render Batch Results Table & Summary Cards
function renderBatchResults(data) {
    const summarySec = document.getElementById('batch-summary-section');
    const tableSec = document.getElementById('batch-table-section');
    summarySec.classList.remove('hidden');
    tableSec.classList.remove('hidden');

    // Update Counts
    document.getElementById('batch-count-total').textContent = data.total_screened;
    document.getElementById('batch-count-malignant').textContent = data.summary.malignant_count;
    document.getElementById('batch-count-benign').textContent = data.summary.benign_count;
    document.getElementById('batch-count-indeterminate').textContent = data.summary.indeterminate_count;

    const tbody = document.getElementById('batch-table-body');
    let html = '';

    data.patients.forEach(p => {
        const isMal = p.diagnosis === 'Malignant';
        const badge = isMal
            ? `<span class="px-2 py-0.5 rounded text-xs font-bold bg-rose-100 text-rose-800">MALIGNANT</span>`
            : `<span class="px-2 py-0.5 rounded text-xs font-bold bg-emerald-100 text-emerald-800">BENIGN</span>`;

        const urgentTag = p.requires_urgent_review
            ? `<span class="px-2 py-0.5 rounded-full text-[10px] font-extrabold bg-red-600 text-white tracking-wider animate-pulse">URGENT</span>`
            : (p.is_borderline ? `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-200 text-amber-900">REVIEW</span>` : `<span class="text-xs text-slate-600">Routine</span>`);

        html += `
            <tr class="border-b border-slate-100 hover:bg-slate-50 text-xs">
                <td class="py-2.5 px-3 font-mono font-bold text-slate-800">${p.patient_id}</td>
                <td class="py-2.5 px-3">${badge}</td>
                <td class="py-2.5 px-3 font-mono font-medium">
                    <div class="flex items-center space-x-2">
                        <span class="w-10 text-right">${p.malignancy_percentage}%</span>
                        <div class="w-16 bg-slate-200 rounded-full h-1.5 overflow-hidden">
                            <div class="h-1.5 ${isMal ? 'bg-rose-500' : 'bg-emerald-500'}" style="width: ${p.malignancy_percentage}%"></div>
                        </div>
                    </div>
                </td>
                <td class="py-2.5 px-3 text-slate-600">${p.risk_category}</td>
                <td class="py-2.5 px-3">${urgentTag}</td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
}

// Render Analytics Tab (Performance, Confusion Matrix, Feature Ranking)
function renderAnalyticsTab() {
    if (!appState.metadata || !appState.metadata.metrics) return;
    const m = appState.metadata.metrics;
    const cm = m.confusion_matrix;

    document.getElementById('metric-accuracy').textContent = `${(m.accuracy * 100).toFixed(2)}%`;
    document.getElementById('metric-auc').textContent = m.roc_auc.toFixed(4);
    document.getElementById('metric-sensitivity').textContent = `${(m.sensitivity_recall * 100).toFixed(1)}%`;
    document.getElementById('metric-specificity').textContent = `${(m.specificity * 100).toFixed(1)}%`;
    document.getElementById('metric-precision').textContent = `${(m.precision * 100).toFixed(1)}%`;
    document.getElementById('metric-f1').textContent = `${(m.f1_score * 100).toFixed(1)}%`;

    // Confusion Matrix cells
    document.getElementById('cm-tn').textContent = cm.true_negative;
    document.getElementById('cm-fp').textContent = cm.false_positive;
    document.getElementById('cm-fn').textContent = cm.false_negative;
    document.getElementById('cm-tp').textContent = cm.true_positive;

    // Feature ranking table
    const rankTbody = document.getElementById('feature-ranking-tbody');
    if (rankTbody && appState.metadata.feature_ranking) {
        let html = '';
        appState.metadata.feature_ranking.forEach((f, idx) => {
            const def = appState.featureDefinitions[f.feature] || {};
            html += `
                <tr class="border-b border-slate-100 hover:bg-slate-50 text-xs">
                    <td class="py-2 px-3 font-bold text-slate-500">#${idx + 1}</td>
                    <td class="py-2 px-3 font-semibold text-slate-800">
                        ${def.label || f.feature}
                        <span class="text-[10px] text-slate-400 font-normal">(${f.feature})</span>
                    </td>
                    <td class="py-2 px-3 font-mono font-bold text-sky-700">${f.f_score.toFixed(1)}</td>
                    <td class="py-2 px-3">
                        ${f.selected 
                            ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-100 text-sky-800">Active (SelectKBest)</span>` 
                            : `<span class="px-2 py-0.5 rounded text-[10px] text-slate-400">Available</span>`}
                    </td>
                </tr>
            `;
        });
        rankTbody.innerHTML = html;
    }
}

// Retrain Model Trigger
async function handleRetrainModel() {
    if (!confirm('Are you sure you want to retrain the diagnostic model using the Wisconsin dataset?')) return;

    const btn = document.getElementById('btn-retrain-model');
    btn.disabled = true;
    btn.innerHTML = 'Retraining Pipeline...';

    try {
        const res = await fetch('/api/retrain', { method: 'POST' });
        if (!res.ok) throw new Error('Retraining request failed');
        const data = await res.json();
        showToast('Model successfully retrained and updated!', 'success');
        await fetchAppMetadata();
    } catch (err) {
        console.error('Retrain error:', err);
        showToast('Retraining error: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Retrain & Refresh Model';
    }
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    let bg = 'bg-slate-800 text-white';
    if (type === 'success') bg = 'bg-emerald-600 text-white';
    if (type === 'error') bg = 'bg-rose-600 text-white';
    if (type === 'info') bg = 'bg-sky-700 text-white';

    toast.className = `fixed bottom-5 right-5 z-50 px-4 py-3 rounded-lg shadow-xl text-xs font-semibold flex items-center space-x-2 transition-all transform duration-300 ${bg}`;
    toast.innerHTML = `<span>${message}</span>`;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}

