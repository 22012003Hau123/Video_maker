/* ============================================================
   Video Subtitles Automation — App JS
   ============================================================ */

'use strict';

// ── I18N ─────────────────────────────────────────────────────
const I18N = {
    en: {
        'app.title':           'Video Subtitles Automation',
        'sidebar.sub':         'Sub',
        'sidebar.legal':       'Legal',
        'sidebar.master':      'Master',
        'preview.placeholder': 'Upload a video to preview',
        'sub.source_video':    'Source Video',
        'upload.drop_video':   'Drop video or click to browse',
        'sub.mode_label':      'Subtitle Mode',
        'sub.auto_ai':         'Auto AI',
        'sub.excel_import':    'Excel Import',
        'sub.ai_desc':         'AI automatically generates subtitles from your video audio via Whisper.',
        'sub.upload_excel':    'Upload Translation Excel / CSV',
        'sub.add_row':         'Add Row',
        'sub.cells_editable':  'Cells are editable',
        'sub.advanced':        'Advanced Settings',
        'sub.translate_to':    'Translate to',
        'sub.live_preview':    'Live preview subtitle',
        'sub.font':            'Font',
        'sub.font_size':       'Font size',
        'sub.position':        'Position',
        'sub.output_preset':   'Output Preset',
        'sub.fmt.standard':    '🌐 Standard MP4 (H.264)',
        'sub.fmt.prores':      '🎬 ProRes 422 (.MOV)',
        'sub.fmt.highbitrate': '🚀 High Bitrate 20Mbps',
        'sub.fmt.ae':          '🧩 AE Package (ProRes+WAV+SRT+JSON)',
        'sub.info.standard':   'Ideal for web sharing and general review.',
        'sub.info.prores':     'High quality for post-production and editing.',
        'sub.info.highbitrate':'High bitrate for broadcast or archive use.',
        'sub.info.ae':         'Full asset bundle for After Effects workflow.',
        'sub.font_default':    'Default (Helvetica)',
        'sub.generate':        'Generate Subtitles',
        'legal.options':       'Legal Options',
        'legal.country':       'Country',
        'legal.media_type':    'Media Type',
        'legal.social':        'Social Media',
        'legal.tv':            'TV / Broadcast',
        'legal.usage_type':    'Usage Type',
        'legal.shareable':     'Shareable',
        'legal.non_shareable': 'Non-Shareable',
        'legal.post_type':     'Post Type (9:16)',
        'legal.output_format': 'Output Format',
        'legal.legal_preview': 'Legal Preview Video',
        'legal.ae_package':    '🧩 Legal AE Package',
        'legal.submit':        'Add Legal Content',
        'master.gold_pos':     'Gold POS',
        'master.scene_ai':     'Scene AI',
        'master.version':      'Version (VO)',
        'master.matrix':       'Matrix',
        'master.loading':      'Loading…',
        'master.deliverable':  'Deliverable Lines',
        'master.formats':      'Formats',
        'master.lengths':      'Lengths (s)',
        'master.codecs':       'Codecs',
        'master.preview_plan': 'Preview Plan',
        'master.queue_render': 'Queue Render',
        'master.scene_analysis': 'Scene Analysis',
        'master.analyze_scenes': 'Analyze Scenes',
        'master.smart_cut':    'Smart Cut',
        'master.remove_intro': 'Remove Intro',
        'master.remove_outro': 'Remove Outro',
        'master.remove_logo':  'Remove Logo',
        'master.remove_product': 'Remove Product',
        'master.logo_removal': 'Logo Removal',
        'master.remove_logo_ai': 'Remove Logo (AI)',
        'master.logo_replace': 'Logo Replace',
        'master.upload_logo':  'Upload new logo',
        'master.replace_logo': 'Replace Logo',
        'master.add_packshot': 'Add Packshot',
        'master.upload_packshot': 'Upload packshot image',
        'master.object_removal': 'Object Removal (LaMa AI)',
        'master.open_inpaint': 'Open Inpaint Tool',
        'status.ready':        'Ready',
        'mode.subtitle':       'Subtitle Pipeline',
        'mode.legal':          'Legal Pipeline',
        'mode.master':         'Mastering Pipeline',
    },
    fr: {
        'app.title':           'Automatisation Sous-titres',
        'sidebar.sub':         'Sub',
        'sidebar.legal':       'Légal',
        'sidebar.master':      'Master',
        'preview.placeholder': 'Importer une vidéo pour prévisualiser',
        'sub.source_video':    'Vidéo Source',
        'upload.drop_video':   'Déposer une vidéo ou cliquer pour parcourir',
        'sub.mode_label':      'Mode Sous-titres',
        'sub.auto_ai':         'Auto IA',
        'sub.excel_import':    'Import Excel',
        'sub.ai_desc':         "L'IA génère automatiquement des sous-titres à partir de l'audio de votre vidéo via Whisper.",
        'sub.upload_excel':    'Importer un Excel / CSV de traduction',
        'sub.add_row':         'Ajouter une ligne',
        'sub.cells_editable':  'Les cellules sont modifiables',
        'sub.advanced':        'Paramètres avancés',
        'sub.translate_to':    'Traduire en',
        'sub.live_preview':    'Aperçu en direct',
        'sub.font':            'Police',
        'sub.font_size':       'Taille police',
        'sub.position':        'Position',
        'sub.output_preset':   'Format de sortie',
        'sub.fmt.standard':    '🌐 Standard MP4 (H.264)',
        'sub.fmt.prores':      '🎬 ProRes 422 (.MOV)',
        'sub.fmt.highbitrate': '🚀 Débit élevé 20Mbps',
        'sub.fmt.ae':          '🧩 Package AE (ProRes+WAV+SRT+JSON)',
        'sub.info.standard':   'Idéal pour le partage web et les révisions générales.',
        'sub.info.prores':     'Haute qualité pour la post-production et le montage.',
        'sub.info.highbitrate':'Débit élevé pour la diffusion ou l\'archivage.',
        'sub.info.ae':         'Bundle complet pour le workflow After Effects.',
        'sub.font_default':    'Défaut (Helvetica)',
        'sub.generate':        'Générer les sous-titres',
        'legal.options':       'Options Légales',
        'legal.country':       'Pays',
        'legal.media_type':    'Type de Média',
        'legal.social':        'Réseaux Sociaux',
        'legal.tv':            'TV / Diffusion',
        'legal.usage_type':    "Type d'Utilisation",
        'legal.shareable':     'Partageable',
        'legal.non_shareable': 'Non Partageable',
        'legal.post_type':     'Type de Post (9:16)',
        'legal.output_format': 'Format de Sortie',
        'legal.legal_preview': 'Vidéo de Prévisualisation Légale',
        'legal.ae_package':    '🧩 Package AE Légal',
        'legal.submit':        'Ajouter Contenu Légal',
        'master.gold_pos':     'Gold POS',
        'master.scene_ai':     'IA Scènes',
        'master.version':      'Version (VO)',
        'master.matrix':       'Matrice',
        'master.loading':      'Chargement…',
        'master.deliverable':  'Lignes Livrables',
        'master.formats':      'Formats',
        'master.lengths':      'Durées (s)',
        'master.codecs':       'Codecs',
        'master.preview_plan': 'Aperçu du Plan',
        'master.queue_render': 'Lancer le Rendu',
        'master.scene_analysis': 'Analyse de Scènes',
        'master.analyze_scenes': 'Analyser les Scènes',
        'master.smart_cut':    'Coupe Intelligente',
        'master.remove_intro': "Supprimer l'Intro",
        'master.remove_outro': "Supprimer l'Outro",
        'master.remove_logo':  'Supprimer le Logo',
        'master.remove_product': 'Supprimer le Produit',
        'master.logo_removal': 'Suppression du Logo',
        'master.remove_logo_ai': 'Supprimer Logo (IA)',
        'master.logo_replace': 'Remplacement de Logo',
        'master.upload_logo':  'Importer un nouveau logo',
        'master.replace_logo': 'Remplacer le Logo',
        'master.add_packshot': 'Ajouter Packshot',
        'master.upload_packshot': "Importer l'image packshot",
        'master.object_removal': "Suppression d'Objet (LaMa IA)",
        'master.open_inpaint': "Ouvrir l'Outil d'Inpainting",
        'status.ready':        'Prêt',
        'mode.subtitle':       'Pipeline Sous-titres',
        'mode.legal':          'Pipeline Légal',
        'mode.master':         'Pipeline Mastering',
    },
};

let currentLang = 'en';

function setLang(lang) {
    currentLang = lang;
    const dict = I18N[lang] || I18N['en'];

    // Update all data-i18n elements
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (dict[key] !== undefined) el.textContent = dict[key];
    });

    // Update navbar mode label separately (its key depends on current mode)
    const modeEl = document.getElementById('navbar-mode-label');
    if (modeEl) {
        const modeKey = modeEl.getAttribute('data-i18n-mode');
        const translated = dict['mode.' + modeKey];
        if (translated) modeEl.textContent = translated;
    }

    // Toggle button shows the OTHER language you can switch to
    const btn = document.getElementById('lang-toggle');
    if (btn) btn.textContent = lang === 'en' ? '🇫🇷 FR' : '🇬🇧 EN';
}

function toggleLang() {
    setLang(currentLang === 'en' ? 'fr' : 'en');
}

// ── MODE SWITCHING ────────────────────────────────────────────
const MODE_LABELS = {
    subtitle: 'Subtitle Pipeline',
    legal:    'Legal Pipeline',
    master:   'Mastering Pipeline',
};

function switchMode(mode) {
    // Sidebar buttons
    document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.remove('active'));
    const sbBtn = document.getElementById('sb-' + mode);
    if (sbBtn) sbBtn.classList.add('active');

    // Panels
    document.querySelectorAll('.mode-panel').forEach(p => p.classList.remove('active'));
    const panel = document.getElementById('panel-' + mode);
    if (panel) panel.classList.add('active');

    // Navbar label
    const lbl = document.getElementById('navbar-mode-label');
    if (lbl) {
        lbl.setAttribute('data-i18n-mode', mode);
        const dict = I18N[currentLang] || I18N['en'];
        lbl.textContent = dict['mode.' + mode] || MODE_LABELS[mode] || mode;
    }

    // Load Gold manifest when entering Master
    if (mode === 'master' && typeof loadGoldManifest === 'function') {
        loadGoldManifest();
    }
}

// Legacy compat (used by switchMasterSubtab → switchTab)
function switchTab(tabName) {
    switchMode(tabName);
}

// ── COLLAPSIBLE ───────────────────────────────────────────────
function toggleCollapse(headerId, bodyId) {
    const hdr  = document.getElementById(headerId);
    const body = document.getElementById(bodyId);
    if (!hdr || !body) return;
    hdr.classList.toggle('open');
    body.classList.toggle('open');
}

// Old toggle function compat
function toggleSubtitleSettings() {
    toggleCollapse('sub-settings-hdr', 'sub-settings-body');
}

// ── INPUT MODE (Subtitle panel: auto vs excel) ────────────────
let currentInputMode = 'auto';
let translationLangs = ['en', 'fr', 'es', 'nl', 'ja'];

function switchInputMode(mode) {
    currentInputMode = mode;

    // Toggle buttons
    document.querySelectorAll('.toggle-tab').forEach(t => t.classList.remove('active'));
    const btn = document.getElementById('tab-' + mode);
    if (btn) btn.classList.add('active');

    // Toggle content panels
    document.querySelectorAll('.input-content').forEach(c => c.classList.remove('active'));
    const target = document.getElementById(mode === 'text' ? 'text-input' : 'auto-input');
    if (target) target.classList.add('active');

    // Language group visibility
    const langGroup = document.getElementById('target-language-group');
    if (langGroup) langGroup.style.display = mode === 'text' ? 'none' : '';
}

// ── EXCEL TRANSLATION TABLE ───────────────────────────────────
async function handleTranslationExcel(input) {
    if (input.files.length === 0) return;

    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const uploadContainer = document.getElementById('excel-upload-container');
    const oldContent = uploadContainer.innerHTML;
    uploadContainer.innerHTML = '<i class="ph ph-circle-notch ph-spin" style="font-size:2rem;color:var(--primary);"></i><div style="margin-top:0.5rem">Parsing Excel...</div>';

    try {
        const res = await fetch('/api/parse-translation-excel', { method: 'POST', body: formData });

        if (res.ok) {
            const data = await res.json();
            renderTranslationTable(data.rows);
            uploadContainer.style.display = 'none';
            document.getElementById('translation-table-container').style.display = 'block';
        } else {
            alert('Failed to parse Excel file.');
            uploadContainer.innerHTML = oldContent;
        }
    } catch (e) {
        console.error(e);
        alert('Connection error.');
        uploadContainer.innerHTML = oldContent;
    }
}

function renderTranslationTable(rows) {
    const tbody = document.querySelector('#translation-table tbody');
    tbody.innerHTML = '';
    const theadRow = document.querySelector('#translation-table thead tr');

    if (theadRow) {
        const detectedLangs = rows && rows.length
            ? Object.keys(rows.reduce((acc, row) => Object.assign(acc, row || {}), {}))
            : [];
        translationLangs = (detectedLangs.length ? detectedLangs : ['en', 'fr', 'es', 'nl', 'ja'])
            .map(l => String(l || '').trim().toLowerCase())
            .filter(Boolean);

        let headerHtml = '<th>#</th>';
        translationLangs.forEach((lang, idx) => {
            const label = idx === 0 ? `${lang.toUpperCase()} (Source)` : lang.toUpperCase();
            headerHtml += `<th>${label}</th>`;
        });
        headerHtml += '<th style="text-align:center;width:60px;">Del</th>';
        theadRow.innerHTML = headerHtml;
    }

    if (!rows || rows.length === 0) {
        addTranslationRow();
        return;
    }
    rows.forEach((row, idx) => addTranslationRow(row, idx + 1));
}

function reindexTranslationRows() {
    document.querySelectorAll('#translation-table tbody tr').forEach((row, idx) => {
        const cell = row.querySelector('[data-row-index]');
        if (cell) cell.textContent = String(idx + 1);
    });
}

function removeTranslationRow(btn) {
    const row = btn.closest('tr');
    if (!row) return;
    row.remove();
    const tbody = document.querySelector('#translation-table tbody');
    if (tbody.children.length === 0) addTranslationRow();
    else reindexTranslationRows();
}

function addTranslationRow(data = {}, index = null) {
    const tbody = document.querySelector('#translation-table tbody');
    const rowCount = index || (tbody.children.length + 1);
    const tr = document.createElement('tr');

    const langs = translationLangs && translationLangs.length ? translationLangs : ['en', 'fr', 'es', 'nl', 'ja'];
    let html = `<td data-row-index style="padding:0.6rem;color:var(--text-muted);font-weight:500;">${rowCount}</td>`;

    langs.forEach(lang => {
        const text = data[lang] || '';
        html += `<td contenteditable="true" data-lang="${lang}" style="padding:0.6rem;border-left:1px solid var(--border);outline:none;min-width:120px;">${text}</td>`;
    });

    html += `<td style="padding:0.4rem;border-left:1px solid var(--border);text-align:center;">
        <button type="button" class="btn btn-sm" onclick="removeTranslationRow(this)"
                style="background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.3);">
            <i class="ph ph-trash"></i>
        </button>
    </td>`;

    tr.innerHTML = html;
    tbody.appendChild(tr);
    reindexTranslationRows();

    const firstLang = langs[0];
    if (firstLang && !data[firstLang]) {
        const firstCell = tr.querySelector(`[data-lang="${firstLang}"]`);
        if (firstCell) firstCell.focus();
    }
}

// ── FORMAT DETECTION ──────────────────────────────────────────
function detectFormat(width, height) {
    const ratio = width / height;
    if (ratio > 1.7)                   return '16x9';
    if (ratio < 0.6)                   return '9x16';
    if (ratio > 0.7 && ratio < 0.9)   return '4x5';
    if (ratio > 0.9 && ratio < 1.1)   return '1x1';
    return '16x9';
}

// ── BROWSER DETECTION ─────────────────────────────────────────
function isSafari() {
    return /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
}

// ── THUMBNAIL / PREVIEW ───────────────────────────────────────
async function fetchThumbnail(file, imgEl, videoEl, placeholderEl) {
    if (placeholderEl) {
        placeholderEl.innerHTML = '<i class="ph ph-spinner ph-spin"></i><p>Generating preview...</p>';
        placeholderEl.style.display = '';
    }
    if (imgEl) imgEl.className = '';
    if (videoEl) videoEl.className = '';

    const formData = new FormData();
    formData.append('video', file);
    try {
        const res = await fetch('/api/video-thumbnail', { method: 'POST', body: formData });
        if (res.ok) {
            const width  = parseInt(res.headers.get('X-Video-Width'));
            const height = parseInt(res.headers.get('X-Video-Height'));
            const blob = await res.blob();
            imgEl.src = URL.createObjectURL(blob);
            imgEl.className = 'active';
            if (videoEl) videoEl.className = '';
            if (placeholderEl) placeholderEl.style.display = 'none';
            return { width, height };
        } else {
            if (placeholderEl) placeholderEl.innerHTML = '<i class="ph ph-warning-circle"></i><p>No preview</p>';
        }
    } catch (e) {
        if (placeholderEl) placeholderEl.innerHTML = '<i class="ph ph-warning-circle"></i><p>Preview error</p>';
    }
    return null;
}

async function updatePreview(input) {
    if (!input.files.length) return;
    const file = input.files[0];

    const player      = document.getElementById('main-preview-player');
    const img         = document.getElementById('main-preview-img');
    const placeholder = document.getElementById('main-preview-placeholder');
    const formatInput = document.getElementById('sub-video-format');

    const ext   = file.name.split('.').pop().toLowerCase();
    const isMov = ext === 'mov' || ext === 'qt';

    function showPlayer(src) {
        if (player) { player.src = src; player.className = 'active'; }
        if (img)    img.className = '';
        if (placeholder) placeholder.style.display = 'none';
    }
    function showPlaceholder(html) {
        if (player) player.className = '';
        if (img)    img.className = '';
        if (placeholder) { placeholder.style.display = ''; placeholder.innerHTML = html; }
    }

    if (isMov && isSafari()) {
        const url = URL.createObjectURL(file);
        showPlayer(url);
        const tmp = document.createElement('video');
        tmp.src = url;
        tmp.onloadedmetadata = () => {
            const fmt = detectFormat(tmp.videoWidth, tmp.videoHeight);
            if (formatInput) formatInput.value = fmt;
            syncSlidersToFormat(fmt);
            updateSubPreview();
        };
    } else if (isMov) {
        showPlaceholder('<i class="ph ph-spinner ph-spin"></i><p>Generating preview...</p>');
        try {
            const fd = new FormData();
            fd.append('video', file);
            const res = await fetch('/api/upload-preview', { method: 'POST', body: fd });
            if (!res.ok) throw new Error('transcode failed');
            const { preview_url } = await res.json();
            showPlayer(preview_url);
            const tmp = document.createElement('video');
            tmp.src = preview_url;
            tmp.onloadedmetadata = () => {
                const fmt = detectFormat(tmp.videoWidth, tmp.videoHeight);
                if (formatInput) formatInput.value = fmt;
                syncSlidersToFormat(fmt);
                updateSubPreview();
            };
        } catch {
            showPlaceholder('<i class="ph ph-file-video"></i><p>MOV Selected (Ready)</p>');
        }
    } else {
        const url = URL.createObjectURL(file);
        showPlayer(url);
        const tmp = document.createElement('video');
        tmp.src = url;
        tmp.onloadedmetadata = () => {
            const fmt = detectFormat(tmp.videoWidth, tmp.videoHeight);
            if (formatInput) formatInput.value = fmt;
            syncSlidersToFormat(fmt);
            updateSubPreview();
        };
    }
    clearSubResults();
}

async function updateMasterPreview(input) {
    const player      = document.getElementById('main-preview-player');
    const img         = document.getElementById('main-preview-img');
    const placeholder = document.getElementById('main-preview-placeholder');

    if (input.files.length > 0) {
        const file = input.files[0];
        const ext   = file.name.split('.').pop().toLowerCase();
        const isMov = ext === 'mov' || ext === 'qt';

        if (isMov) {
            await fetchThumbnail(file, img, player, placeholder);
        } else {
            const url = URL.createObjectURL(file);
            if (player) { player.src = url; player.className = 'active'; }
            if (img) img.className = '';
            if (placeholder) placeholder.style.display = 'none';
        }
        clearMasterResults();
    } else {
        if (player) player.className = '';
        if (img) img.className = '';
        if (placeholder) {
            placeholder.style.display = '';
            placeholder.innerHTML = '<i class="ph ph-film-strip"></i><p>Upload a video to preview</p>';
        }
    }
}

// ── CLEAR RESULTS ─────────────────────────────────────────────
function clearSubResults() {
    ['subtitle-status', 'batch-status', 'export-status'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.style.display = 'none'; el.innerHTML = ''; }
    });
}

function clearLegalResults() {
    const el = document.getElementById('legal-status');
    if (el) { el.style.display = 'none'; el.innerHTML = ''; }
}

function clearMasterResults() {
    ['status-smartcut', 'status-logo', 'master-status', 'v-status-area', 'm-status-area'].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.className = el.id.includes('status-') ? 'card-status' : '';
        if (!el.className) el.style.display = 'none';
        el.innerHTML = '';
        el.style.flexDirection = '';
    });
    ['m-result-area', 'v-result-area'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });
}

// ── FILE SELECT (upload zone visual update) ───────────────────
function handleFileSelect(input, uploadId) {
    const upload = document.getElementById(uploadId);
    if (!upload || !input.files.length) return;

    const file     = input.files[0];
    const fileName = file.name;
    const fileSize = (file.size / (1024 * 1024)).toFixed(1);

    upload.classList.add('has-file');

    const icon      = upload.querySelector('.upload-icon');
    const titleEl   = upload.querySelector('.upload-title');
    const subEl     = upload.querySelector('.upload-sub');
    const filenameEl = upload.querySelector('.upload-filename');

    if (icon)       { icon.className = 'ph ph-check-circle upload-icon'; icon.style.color = 'var(--success)'; }
    if (titleEl)    titleEl.style.display = 'none';
    if (subEl)      subEl.textContent = `${fileSize} MB · Click to change`;
    if (filenameEl) { filenameEl.textContent = fileName; filenameEl.style.display = 'block'; }

    // Legal panel: update shared preview player
    if (uploadId === 'video-upload-2') {
        const player      = document.getElementById('main-preview-player');
        const img         = document.getElementById('main-preview-img');
        const placeholder = document.getElementById('main-preview-placeholder');

        const ext   = file.name.split('.').pop().toLowerCase();
        const isMov = ext === 'mov' || ext === 'qt';

        if (isMov) {
            // Just show a fallback placeholder for MOV in legal panel
            if (player) player.className = '';
            if (img)    img.className = '';
            if (placeholder) {
                placeholder.style.display = '';
                placeholder.innerHTML = '<i class="ph ph-file-video"></i><p>MOV Selected (Ready)</p>';
            }
        } else {
            const url = URL.createObjectURL(file);
            if (player) { player.src = url; player.className = 'active'; }
            if (img) img.className = '';
            if (placeholder) placeholder.style.display = 'none';

            const tmp = document.createElement('video');
            tmp.src = url;
            tmp.onloadedmetadata = () => {
                const format = detectFormat(tmp.videoWidth, tmp.videoHeight);
                const mediaSelect = document.querySelector('#legal-form select[name="media_type"]');
                if (mediaSelect && format === '9x16') mediaSelect.value = 'social';
                const subtypeGroup = document.getElementById('legal-subtype-group');
                if (subtypeGroup) subtypeGroup.style.display = format === '9x16' ? '' : 'none';
            };
        }
        clearLegalResults();
    }
}

// ── DROP ZONE SETUP ───────────────────────────────────────────
function setupVideoDropZone(uploadId, inputSelector, onSelected) {
    const upload = document.getElementById(uploadId);
    if (!upload) return;
    const input = upload.querySelector(inputSelector);
    if (!input) return;

    const prevent = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => upload.addEventListener(ev, prevent));
    ['dragenter', 'dragover'].forEach(ev => upload.addEventListener(ev, () => upload.classList.add('drag-over')));
    ['dragleave', 'drop'].forEach(ev => upload.addEventListener(ev, () => upload.classList.remove('drag-over')));

    upload.addEventListener('drop', (e) => {
        const files = e.dataTransfer?.files;
        if (!files || !files.length) return;
        const file = files[0];
        if (!file.type || !file.type.startsWith('video/')) {
            showToast('Please drop a video file.', 'error');
            return;
        }
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        handleFileSelect(input, uploadId);
        if (typeof onSelected === 'function') onSelected(input);
    });
}

function setupExcelDropZone(containerId, inputId) {
    const container = document.getElementById(containerId);
    const input = document.getElementById(inputId);
    if (!container || !input) return;

    const prevent = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => container.addEventListener(ev, prevent));
    ['dragenter', 'dragover'].forEach(ev => container.addEventListener(ev, () => container.classList.add('drag-over')));
    ['dragleave', 'drop'].forEach(ev => container.addEventListener(ev, () => container.classList.remove('drag-over')));

    container.addEventListener('drop', (e) => {
        const files = e.dataTransfer?.files;
        if (!files || !files.length) return;
        const name = (files[0].name || '').toLowerCase();
        if (!name.endsWith('.xlsx') && !name.endsWith('.xls') && !name.endsWith('.csv')) {
            showToast('Please drop an Excel/CSV file.', 'error');
            return;
        }
        const dt = new DataTransfer();
        dt.items.add(files[0]);
        input.files = dt.files;
        handleTranslationExcel(input);
    });
}

// ── FORM SUBMISSIONS ──────────────────────────────────────────
async function submitSubtitle(e) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);

    const videoInput = form.querySelector('input[name="video"]');
    const videoFile  = videoInput?.files?.[0];
    if (!videoFile || !videoFile.size) {
        alert('Please select a valid video file before generating subtitles.');
        return;
    }

    const selectedLangs = [];
    form.querySelectorAll('input[name^="lang_"]:checked').forEach(cb => selectedLangs.push(cb.value));
    formData.set('target_languages', selectedLangs.join(','));

    const exportFormat = form.querySelector('select[name="export_format"]')?.value || 'mp4_standard';
    formData.set('export_video', true);
    formData.set('export_srt', exportFormat === 'ae_package');

    let endpoint = '/api/subtitle';

    if (currentInputMode === 'text') {
        const rows = document.querySelectorAll('#translation-table tbody tr');
        const langs = translationLangs && translationLangs.length ? translationLangs : ['en'];
        const translations = {};
        const nonEmptyByLang = {};
        langs.forEach(lang => { translations[lang] = []; nonEmptyByLang[lang] = 0; });

        let hasData = false;
        rows.forEach(row => {
            row.querySelectorAll('[data-lang]').forEach(cell => {
                const lang = cell.getAttribute('data-lang');
                const text = cell.innerText.trim();
                translations[lang].push(text);
                if (text) { hasData = true; if (nonEmptyByLang[lang] !== undefined) nonEmptyByLang[lang]++; }
            });
        });

        if (!hasData) { alert('Please upload an Excel file or enter some translations!'); return; }

        const autoTargetLangs = Object.keys(translations).filter(l => (nonEmptyByLang[l] || 0) > 0);
        if (!autoTargetLangs.length) { alert('No language columns with content found.'); return; }

        formData.set('target_languages', autoTargetLangs.join(','));
        formData.set('manual_translations_json', JSON.stringify(translations));

        const sourceLang = autoTargetLangs.includes('en') ? 'en' : autoTargetLangs[0];
        formData.set('source_lang', sourceLang);
        formData.set('subtitle_text', (translations[sourceLang] || []).join('\n'));
        formData.set('auto_segment_rhythm', true);
        endpoint = '/api/subtitle-text';
    }

    const submitBtn = document.getElementById('subtitle-submit');
    const statusBox = document.getElementById('subtitle-status');
    submitBtn.disabled = true;
    statusBox.style.display = 'block';

    const updateUI = (pct, msg) => {
        statusBox.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                <span style="font-size:0.8rem;color:var(--text-muted);">Subtitle Pipeline</span>
                <span style="font-size:1rem;font-weight:700;color:var(--primary);">${pct}%</span>
            </div>
            <div class="progress-bar"><div class="fill" style="width:${pct}%"></div></div>
            <div style="font-size:0.8rem;color:var(--text-muted);margin-top:0.4rem;">
                <i class="ph ph-circle-notch ph-spin"></i> ${msg}
            </div>`;
        updateStatusbar(pct, msg);
    };

    updateUI(0, 'Preparing upload...');

    const xhr = new XMLHttpRequest();
    xhr.open('POST', endpoint);
    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            updateUI(Math.round(pct * 0.2), `Uploading video (${pct}%)...`);
        }
    };
    xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            const data = JSON.parse(xhr.responseText);
            if (data.job_id) pollJobStatus(data.job_id, 'subtitle-status');
        } else {
            let msg = 'Backend error';
            try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
            alert('Error: ' + msg);
            submitBtn.disabled = false;
        }
    };
    xhr.onerror = () => {
        alert('Network error during upload');
        submitBtn.disabled = false;
    };
    xhr.send(formData);
}

async function submitBatch(e) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);

    const selectedLangs = [];
    document.querySelectorAll('#batch-form input[type="checkbox"]:checked').forEach(cb => selectedLangs.push(cb.value));
    if (!selectedLangs.length) { alert('Please select at least one language!'); return; }

    document.querySelectorAll('#batch-form input[type="checkbox"]').forEach(cb => formData.delete(cb.name));
    formData.append('target_languages', selectedLangs.join(','));

    const btn = document.getElementById('batch-submit');
    const statusBox = document.getElementById('batch-status');
    if (btn) btn.disabled = true;
    if (statusBox) statusBox.style.display = 'block';

    try {
        const res = await fetch('/api/subtitle-batch', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.job_id) pollJobStatus(data.job_id, 'batch-status');
    } catch (error) {
        alert('Error: ' + error.message);
        if (btn) btn.disabled = false;
    }
}

async function submitExport(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    const btn = document.getElementById('export-submit');
    const statusBox = document.getElementById('export-status');
    if (btn) btn.disabled = true;
    if (statusBox) statusBox.style.display = 'block';

    try {
        const res = await fetch('/api/subtitle-export', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.job_id) pollJobStatus(data.job_id, 'export-status');
    } catch (error) {
        alert('Error: ' + error.message);
        if (btn) btn.disabled = false;
    }
}

async function submitLegal(e) {
    e.preventDefault();
    const formData = new FormData(e.target);

    const submitBtn = document.getElementById('legal-submit');
    const statusBox = document.getElementById('legal-status');
    submitBtn.disabled = true;
    statusBox.style.display = 'block';
    statusBox.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
            <span style="font-size:0.8rem;color:var(--text-muted);">Legal Pipeline</span>
            <span style="font-size:1rem;font-weight:700;color:var(--primary);">0%</span>
        </div>
        <div class="progress-bar"><div class="fill" style="width:0%"></div></div>
        <div style="font-size:0.8rem;color:var(--text-muted);margin-top:0.4rem;">
            <i class="ph ph-spinner ph-spin"></i> Initializing...
        </div>`;
    updateStatusbar(0, 'Legal Pipeline — Initializing...');

    try {
        const res = await fetch('/api/legal', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.job_id) {
            pollJobStatus(data.job_id, 'legal-status');
        } else {
            statusBox.innerHTML = '<i class="ph ph-x-circle"></i> Error: No job ID returned';
            submitBtn.disabled = false;
        }
    } catch (error) {
        statusBox.innerHTML = '<i class="ph ph-x-circle"></i> Connection Error: ' + error.message;
        submitBtn.disabled = false;
    }
}

// ── STATUSBAR UPDATE ──────────────────────────────────────────
function updateStatusbar(pct, label) {
    const fill    = document.getElementById('statusbar-fill');
    const lbl     = document.getElementById('statusbar-label');
    const percent = document.getElementById('statusbar-percent');
    if (fill)    fill.style.width = pct + '%';
    if (lbl)     lbl.textContent  = label || 'Ready';
    if (percent) percent.textContent = pct > 0 ? pct + '%' : '';
}

// ── GOLD MANIFEST ─────────────────────────────────────────────
let goldManifestLoaded = false;
let goldManifestData   = null;

function switchMasterSubtab(which) {
    const g = document.getElementById('master-subtab-gold');
    const a = document.getElementById('master-subtab-ai');
    if (g) g.classList.toggle('active', which === 'gold');
    if (a) a.classList.toggle('active', which === 'ai');

    const pg = document.getElementById('master-subpanel-gold');
    const pa = document.getElementById('master-subpanel-ai');
    if (pg) { pg.className = 'master-subpanel' + (which === 'gold' ? ' active' : ''); }
    if (pa) { pa.className = 'master-subpanel' + (which === 'ai' ? ' active' : ''); }

    if (which === 'gold') loadGoldManifest();
}

async function loadGoldManifest() {
    if (goldManifestLoaded) return;
    const hint = document.getElementById('gold-matrix-hint');
    try {
        const r = await fetch('/api/mastering/gold-manifest');
        if (!r.ok) throw new Error(await r.text());
        goldManifestData = await r.json();
        goldManifestLoaded = true;
        const m = goldManifestData.matrix || {};
        if (hint) {
            hint.innerHTML = `<strong>${m.content_variants_per_vo||'—'}</strong> nội dung/VO → <strong>${m.file_outputs_per_vo||m.outputs_per_vo||'—'}</strong> file/VO (×${m.codecs||2}) · cả hai VO: <strong>${m.outputs_total_both_vo||'—'}</strong>`;
        }
        renderGoldFilters();
        renderGoldAssetBanner();
        renderGoldLines();
    } catch (e) {
        if (hint) hint.textContent = 'Could not load manifest: ' + e.message;
    }
}

function _goldChipHtml(ok, innerLabel, inputName, inputValue, checked) {
    const st  = ok ? 'background:#fff;color:#0f172a;border:1px solid #e2e8f0;' : 'background:#b91c1c;color:#f8fafc;border:1px solid #7f1d1d;';
    const dis = ok ? '' : ' disabled';
    const chk = (ok && checked) ? ' checked' : '';
    const suf = ok ? '' : '<span style="font-weight:700;margin-left:0.25rem;">niet klaar</span>';
    return `<label style="display:inline-flex;align-items:center;flex-wrap:wrap;gap:0.2rem;padding:0.3rem 0.45rem;border-radius:8px;cursor:${ok ? 'pointer' : 'not-allowed'};${st}font-size:0.75rem;font-weight:600;">
        <input type="checkbox" name="${inputName}" value="${inputValue}"${chk}${dis}> ${innerLabel}${suf}
    </label>`;
}

function renderGoldAssetBanner() {
    const el = document.getElementById('gold-asset-banner');
    if (!el || !goldManifestData?.asset_flags) return;
    const af = goldManifestData.asset_flags;
    const bad = [];
    if (!af.opening_ok) bad.push('Opening');
    if (!af.full_video_ok) bad.push('Full video');
    if (!af.closing_duo_ok) bad.push('Closing duo 16:9');
    if (af.closing_solo_placeholder) bad.push('Closing solo (placeholder)');
    Object.keys(af.wav_by_duration || {}).forEach(d => { if (!af.wav_by_duration[d]) bad.push('WAV ' + d + 's'); });
    Object.keys(af.logo_by_format  || {}).forEach(f => { if (!af.logo_by_format[f])  bad.push('Logo ' + f); });
    Object.keys(af.surimp_vo2_solo || {}).forEach(f => { if (!af.surimp_vo2_solo[f]) bad.push('Surimp VO2 solo ' + f); });

    if (!bad.length) { el.style.display = 'none'; return; }
    el.style.display = 'block';
    el.style.cssText += 'background:rgba(185,28,28,0.22);border:1px solid #b91c1c;color:#fecaca;';
    el.innerHTML = '<strong>Missing:</strong> ' + bad.join(' · ');
}

function renderGoldFilters() {
    if (!goldManifestData) return;
    const af = goldManifestData.asset_flags || {};
    const fw = document.getElementById('gold-formats-wrap');
    const dw = document.getElementById('gold-durations-wrap');
    const cw = document.getElementById('gold-codecs-wrap');

    if (fw) fw.innerHTML = (goldManifestData.formats || []).map(f => {
        const ok = af.logo_by_format?.[f.id] === true;
        return _goldChipHtml(ok, `${f.id} (${f.width}×${f.height})`, 'gold-fmt', f.id, true);
    }).join('');

    if (dw) dw.innerHTML = (goldManifestData.durations_seconds || []).map(d => {
        const ok = af.wav_by_duration?.[String(d)] === true;
        return _goldChipHtml(ok, d + 's', 'gold-dur', d, true);
    }).join('');

    if (cw) cw.innerHTML = (goldManifestData.export_codecs || []).map(c =>
        _goldChipHtml(true, c.label || c.id, 'gold-codec', c.id, true)
    ).join('');
}

function goldOnVoChange() { renderGoldLines(); }

function renderGoldLines() {
    if (!goldManifestData) return;
    const vo   = document.getElementById('gold-vo').value;
    const wrap = document.getElementById('gold-lines-wrap');
    const lines = (goldManifestData.deliverable_lines || {})[vo] || [];
    const readinessById = {};
    (goldManifestData.readiness?.[vo] || []).forEach(r => { readinessById[r.id] = r; });
    if (!wrap) return;

    wrap.innerHTML = lines.map((L, i) => {
        const r = readinessById[L.id];
        let ok = r?.ready;
        if (['solo_branded_sub','solo_branded_nosub','solo_clean_sub','solo_clean_nosub'].includes(L.id)) ok = false;

        const isSolo = L.label.toLowerCase().includes('solo');
        const rowStyle = ok ? 'background:#fff;color:#0f172a;' : (isSolo ? '' : 'background:#b91c1c;color:#f8fafc;');
        const suffix   = ok ? '' : '<span class="gold-line-not-ready">niet klaar</span>';
        const tip      = ok ? '' : ` title="Missing: ${(r?.missing || []).join(', ')}"`;

        return `<label class="gold-line-row${isSolo ? ' solo-warning' : ''}"${tip}
            style="display:flex;align-items:flex-start;gap:0.5rem;${rowStyle}padding:0.4rem 0.55rem;border-radius:8px;cursor:${ok ? 'pointer' : 'not-allowed'};">
            <input type="checkbox" name="gold-line" value="${L.id}"${ok ? ' checked' : ''}${ok ? '' : ' disabled'}>
            <span style="flex:1;font-size:0.8rem;">${i + 1}. ${L.label}</span>
            ${suffix}
        </label>`;
    }).join('');
}

function _goldChecked(name) {
    return Array.from(document.querySelectorAll(`input[name="${name}"]:checked`)).map(el => el.value);
}

function _goldRequestBody() {
    const vo      = document.getElementById('gold-vo').value;
    const lineIds = _goldChecked('gold-line').map(String);
    const formats = _goldChecked('gold-fmt').map(String);
    const durs    = _goldChecked('gold-dur').map(s => parseInt(s, 10));
    const codecs  = _goldChecked('gold-codec').map(String);
    if (!lineIds.length) return { error: 'Select at least one deliverable line.' };
    return { body: { vo, line_ids: lineIds, formats, durations_seconds: durs, export_codec_ids: codecs } };
}

async function submitGoldPlan() {
    const status = document.getElementById('gold-plan-status');
    const btn    = document.getElementById('gold-plan-btn');
    const gr     = _goldRequestBody();
    if (gr.error) { alert(gr.error); return; }
    if (status) status.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Planning…';
    if (btn) btn.disabled = true;

    try {
        const r = await fetch('/api/mastering/gold-plan', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(gr.body)
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || JSON.stringify(data));

        if (status) {
            status.innerHTML = `
                <div style="padding:0.75rem;background:var(--panel-light);border-radius:var(--radius);border:1px solid var(--border);">
                    <div style="color:var(--success);font-weight:700;margin-bottom:0.3rem;">
                        <i class="ph ph-check-circle"></i> Batch Plan Ready
                    </div>
                    <div style="font-size:0.82rem;"><b>${data.count}</b> jobs for VO: <code>${data.vo}</code></div>
                    ${data.first_job_assets ? `
                    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.3rem;">
                        Check: WAV ${data.first_job_assets.wav?.exists ? '✅' : '❌'} |
                        Logo ${data.first_job_assets.logo?.exists ? '✅' : '❌'} |
                        Surimp ${data.first_job_assets.surimp?.exists ? '✅' : '❌'}
                    </div>` : ''}
                </div>`;
        }
    } catch (e) {
        if (status) status.innerHTML = `<span style="color:var(--error)">${e.message}</span>`;
    }
    if (btn) btn.disabled = false;
}

async function submitGoldRender() {
    const status = document.getElementById('gold-plan-status');
    const btn    = document.getElementById('gold-render-btn');
    const gr     = _goldRequestBody();
    if (gr.error) { alert(gr.error); return; }

    if (status) status.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Queuing render…';
    if (btn) btn.disabled = true;

    try {
        const r = await fetch('/api/mastering/gold-render', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(gr.body)
        });
        const data = await r.json();
        if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));

        if (status) status.innerHTML = `<i class="ph ph-spinner ph-spin"></i> Job <code>${data.job_id}</code> queued…`;
        pollJobStatus(data.job_id, 'gold-plan-status');
    } catch (e) {
        if (status) status.innerHTML = `<span style="color:var(--error)">${e.message}</span>`;
        if (btn) btn.disabled = false;
    }
}

// ── MASTER AI TOOLS ───────────────────────────────────────────
let currentSceneJobId = null;

function getMasterVideo() {
    const input = document.getElementById('master-video-input');
    if (!input || !input.files.length) { alert('Please upload a video first!'); return null; }
    return input.files[0];
}

async function analyzeScenes() {
    const video = getMasterVideo();
    if (!video) return;

    const btn = document.getElementById('btn-analyze');
    btn.disabled = true;
    btn.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Analyzing...';

    try {
        const formData = new FormData();
        formData.append('video', video);
        const res  = await fetch('/api/master/analyze-scenes', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.job_id) pollSceneResults(data.job_id);
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
        btn.disabled = false;
        btn.innerHTML = '<i class="ph ph-scan"></i> Analyze Scenes';
    }
}

async function pollSceneResults(jobId) {
    const poll = async () => {
        try {
            const res  = await fetch(`/api/master/scene-results/${jobId}`);
            const data = await res.json();
            if (data.status === 'completed' && data.scenes) {
                currentSceneJobId = jobId;
                displaySceneTimeline(data.scenes);
                document.getElementById('btn-smartcut').disabled   = false;
                document.getElementById('btn-logo-remove').disabled = false;
                const btn = document.getElementById('btn-analyze');
                btn.disabled = false;
                btn.innerHTML = '<i class="ph ph-check-circle"></i> Analysis Done';
                showToast(`Scene analysis done! ${data.scenes.length} scenes detected.`, 'success');
            } else if (data.status === 'failed') {
                const btn = document.getElementById('btn-analyze');
                btn.disabled = false;
                btn.innerHTML = '<i class="ph ph-scan"></i> Analyze Scenes';
                showToast('Scene analysis failed!', 'error');
            } else {
                setTimeout(poll, 3000);
            }
        } catch (e) { setTimeout(poll, 5000); }
    };
    setTimeout(poll, 3000);
}

function displaySceneTimeline(scenes) {
    const container  = document.getElementById('scene-timeline');
    const resultsDiv = document.getElementById('scene-results');
    resultsDiv.style.display = 'block';

    const typeColors = {
        content: '#43e97b', logo: '#f5576c', intro: '#f093fb',
        outro: '#667eea', product: '#fda085', black_screen: '#333', unknown: '#888',
    };

    container.innerHTML = scenes.map(s => {
        const color    = typeColors[s.scene_type] || '#888';
        const thumbHtml = s.keyframe_url
            ? `<img src="${s.keyframe_url}" alt="Scene" style="width:100%;height:56px;object-fit:cover;border-radius:4px;">`
            : `<div style="width:100%;height:56px;background:var(--surface);border-radius:4px;display:flex;align-items:center;justify-content:center;"><i class="ph ph-image" style="color:var(--text-muted);"></i></div>`;
        return `<div style="min-width:100px;max-width:100px;background:var(--panel-light);border-radius:8px;border-top:3px solid ${color};padding:0.35rem;flex-shrink:0;">
            ${thumbHtml}
            <div style="margin-top:0.25rem;text-align:center;">
                <div style="font-size:0.6rem;font-weight:700;color:${color};text-transform:uppercase;">${s.scene_type}</div>
                <div style="font-size:0.55rem;color:var(--text-muted);">${s.start_time.toFixed(1)}s–${s.end_time.toFixed(1)}s</div>
            </div>
        </div>`;
    }).join('');
}

async function smartCut() {
    if (!currentSceneJobId) { alert('Please run Scene Analysis first!'); return; }
    const btn      = document.getElementById('btn-smartcut');
    const statusEl = document.getElementById('status-smartcut');
    btn.disabled = true;
    btn.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Cutting...';
    clearMasterResults();

    try {
        const formData = new FormData();
        formData.append('scene_job_id', currentSceneJobId);
        formData.append('remove_intro',   document.getElementById('sc-intro').checked);
        formData.append('remove_outro',   document.getElementById('sc-outro').checked);
        formData.append('remove_logo',    document.getElementById('sc-logo').checked);
        formData.append('remove_product', document.getElementById('sc-product').checked);

        const res  = await fetch('/api/master/smart-cut', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.job_id) {
            pollMasterJob(data.job_id, {
                statusEl,
                onComplete: (job) => {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="ph ph-scissors"></i> Smart Cut';
                    if (job.result_path?.endsWith('.mp4')) {
                        const dlUrl = '/api/download/' + data.job_id;
                        statusEl.className = 'card-status active done';
                        statusEl.style.flexDirection = 'column';
                        statusEl.innerHTML = `
                            <video controls src="${dlUrl}" style="width:100%;border-radius:6px;margin-bottom:0.4rem;"></video>
                            <a href="${dlUrl}" download class="btn btn-primary" style="width:100%;justify-content:center;text-decoration:none;font-size:0.8rem;">
                                <i class="ph ph-download-simple"></i> Download Video
                            </a>`;
                        showToast('Smart Cut completed!', 'success');
                    } else {
                        showToast('Smart Cut completed!', 'success');
                    }
                },
                onFail: (job) => {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="ph ph-scissors"></i> Smart Cut';
                    if (statusEl) { statusEl.className = 'card-status active error'; statusEl.innerHTML = '<i class="ph ph-x-circle"></i> ' + (job.error || 'Failed'); }
                    showToast('Smart Cut failed!', 'error');
                }
            });
        }
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
        btn.disabled = false;
        btn.innerHTML = '<i class="ph ph-scissors"></i> Smart Cut';
    }
}

async function logoRemove() {
    if (!currentSceneJobId) { alert('Please run Scene Analysis first!'); return; }
    const btn      = document.getElementById('btn-logo-remove');
    const statusEl = document.getElementById('status-logo');
    btn.disabled = true;
    btn.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Removing...';
    clearMasterResults();

    try {
        const formData = new FormData();
        formData.append('scene_job_id', currentSceneJobId);
        const res  = await fetch('/api/master/logo-remove', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.job_id) {
            pollMasterJob(data.job_id, {
                statusEl,
                onComplete: (job) => {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="ph ph-magic-wand"></i> Auto Remove Logo';
                    if (job.result_path?.endsWith('.mp4')) {
                        const dlUrl = '/api/download/' + data.job_id;
                        statusEl.className = 'card-status active done';
                        statusEl.style.flexDirection = 'column';
                        statusEl.innerHTML = `
                            <video controls src="${dlUrl}" style="width:100%;border-radius:6px;margin-bottom:0.4rem;"></video>
                            <a href="${dlUrl}" download class="btn btn-primary" style="width:100%;justify-content:center;text-decoration:none;font-size:0.8rem;">
                                <i class="ph ph-download-simple"></i> Download Video
                            </a>`;
                        showToast('Logo removal completed!', 'success');
                    } else {
                        showToast('Logo removal completed!', 'success');
                    }
                },
                onFail: (job) => {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="ph ph-magic-wand"></i> Auto Remove Logo';
                    if (statusEl) { statusEl.className = 'card-status active error'; statusEl.innerHTML = '<i class="ph ph-x-circle"></i> ' + (job.error || 'Failed'); }
                    showToast('Logo removal failed!', 'error');
                }
            });
        }
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
        btn.disabled = false;
        btn.innerHTML = '<i class="ph ph-magic-wand"></i> Auto Remove Logo';
    }
}

async function replaceLogo() {
    const video    = getMasterVideo(); if (!video) return;
    const logoInput = document.getElementById('logo-input');
    if (!logoInput?.files.length) { alert('Please upload a new logo!'); return; }

    document.getElementById('btn-replace-logo').disabled = true;
    const statusBox = document.getElementById('master-status');
    statusBox.style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('video', video);
        formData.append('logo', logoInput.files[0]);
        const res  = await fetch('/api/master/replace-logo', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.job_id) pollJobStatus(data.job_id, 'master-status');
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        document.getElementById('btn-replace-logo').disabled = false;
    }
}

async function addPackshot() {
    const video   = getMasterVideo(); if (!video) return;
    const psInput = document.getElementById('packshot-input');
    if (!psInput?.files.length) { alert('Please upload a packshot image!'); return; }

    document.getElementById('btn-packshot').disabled = true;
    const statusBox = document.getElementById('master-status');
    statusBox.style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('video', video);
        formData.append('packshot', psInput.files[0]);
        const res  = await fetch('/api/master/add-packshot', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.job_id) pollJobStatus(data.job_id, 'master-status');
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        document.getElementById('btn-packshot').disabled = false;
    }
}

// ── LOAD INTO MAIN PREVIEW ───────────────────────────────────
function loadInMainPreview(url, fallbacks, label) {
    const player = document.getElementById('main-preview-player');
    const img    = document.getElementById('main-preview-img');
    const ph     = document.getElementById('main-preview-placeholder');
    if (!player) return;
    if (fallbacks) player.dataset.fallbacks = fallbacks;
    player.src = url;
    player.className = 'active';
    if (img) img.className = '';
    if (ph)  ph.style.display = 'none';
    player.play().catch(() => {});
    showToast('Loaded: ' + (label || 'video'), 'success');
}

// ── TOAST ─────────────────────────────────────────────────────
// ── SUBTITLE LIVE PREVIEW ────────────────────────────────────
function syncSlidersToFormat(format) {
    const def = FORMAT_DEFAULTS[format];
    if (!def) return;
    const sizeEl   = document.getElementById('sub-font-size');
    const marginEl = document.getElementById('sub-margin-v');
    const sizeVal  = document.getElementById('font-size-val');
    const marginVal= document.getElementById('margin-v-val');
    if (sizeEl)   { sizeEl.value   = def.size;    if (sizeVal)   sizeVal.textContent   = def.size; }
    if (marginEl) { marginEl.value = def.marginV; if (marginVal) marginVal.textContent = def.marginV; }
}

const FORMAT_DEFAULTS = {
    '16x9': { size: 48, marginV: 30, refW: 1920, refH: 1080 },
    '1x1':  { size: 48, marginV: 30, refW: 1080, refH: 1080 },
    '4x5':  { size: 48, marginV: 38, refW: 1080, refH: 1350 },
    '9x16': { size: 55, marginV: 100, refW: 1080, refH: 1920 },
};

function updateSubPreview() {
    const overlay = document.getElementById('sub-preview-overlay');
    const textEl  = document.getElementById('sub-preview-text');
    if (!overlay || !textEl) return;
    if (!document.getElementById('sub-preview-enabled')?.checked) return;

    const player     = document.getElementById('main-preview-player');
    const format     = document.getElementById('sub-video-format')?.value || '16x9';
    const fontFamily = document.getElementById('font-family-select')?.value || 'Helvetica';
    const fontSize   = parseInt(document.getElementById('sub-font-size')?.value  || '48');
    const marginV    = parseInt(document.getElementById('sub-margin-v')?.value   || '30');
    const def        = FORMAT_DEFAULTS[format] || FORMAT_DEFAULTS['16x9'];

    // Find where the video content actually renders inside the element
    // (object-fit: contain may leave black bars inside the <video> element)
    const wrapper     = player.parentElement;
    const wrapperRect = wrapper.getBoundingClientRect();
    const playerRect  = player.getBoundingClientRect();

    const vidW = player.videoWidth  || def.refW;
    const vidH = player.videoHeight || def.refH;
    const videoAspect = vidW / vidH;
    const elemAspect  = playerRect.width / (playerRect.height || 1);

    let contentW, contentH, contentLeft, contentBottom;

    if (videoAspect > elemAspect) {
        // Letterbox: black bars top & bottom inside the player element
        contentW = playerRect.width;
        contentH = playerRect.width / videoAspect;
        const barH = (playerRect.height - contentH) / 2;
        contentLeft   = playerRect.left  - wrapperRect.left;
        contentBottom = (wrapperRect.bottom - playerRect.bottom) + barH;
    } else {
        // Pillarbox: black bars left & right inside the player element
        contentH = playerRect.height;
        contentW = playerRect.height * videoAspect;
        const barW = (playerRect.width - contentW) / 2;
        contentLeft   = (playerRect.left - wrapperRect.left) + barW;
        contentBottom = wrapperRect.bottom - playerRect.bottom;
    }

    const scale        = contentW / def.refW;
    const scaledSize   = Math.round(fontSize * scale);
    const scaledMargin = Math.round(marginV  * scale);

    // Derive font-style / font-weight from font name since many files are shared
    const fn = (fontFamily || '').toLowerCase();
    const isOblique = fn.includes('obl') || fn.includes('ita');
    const isBlack   = fn.includes('bla');
    const isBold    = isBlack || fn.includes('bol');

    textEl.style.fontFamily = fontFamily ? `"${fontFamily}", Helvetica, Arial, sans-serif`
                                         : 'Helvetica, Arial, sans-serif';
    textEl.style.fontStyle  = isOblique ? 'oblique' : 'normal';
    textEl.style.fontWeight = isBlack ? '900' : isBold ? 'bold' : 'normal';
    textEl.style.fontSize   = scaledSize + 'px';
    textEl.style.textShadow = `0 ${Math.round(2*scale)}px ${Math.round(12*scale)}px rgba(0,0,0,0.55)`;

    // Position overlay to exactly cover the rendered video content
    overlay.style.left    = contentLeft + 'px';
    overlay.style.right   = 'auto';
    overlay.style.width   = contentW + 'px';
    overlay.style.bottom  = (contentBottom + scaledMargin) + 'px';
    overlay.style.display = 'block';
}

function toggleSubPreview(on) {
    const overlay = document.getElementById('sub-preview-overlay');
    if (!overlay) return;
    if (on) {
        // Sync sliders to current format defaults
        const format = document.getElementById('sub-video-format')?.value || '16x9';
        const def = FORMAT_DEFAULTS[format] || FORMAT_DEFAULTS['16x9'];
        const sizeEl   = document.getElementById('sub-font-size');
        const marginEl = document.getElementById('sub-margin-v');
        if (sizeEl && !sizeEl.dataset.userChanged)   { sizeEl.value   = def.size;    document.getElementById('font-size-val').textContent  = def.size; }
        if (marginEl && !marginEl.dataset.userChanged){ marginEl.value = def.marginV; document.getElementById('margin-v-val').textContent = def.marginV; }
        updateSubPreview();
    } else {
        overlay.style.display = 'none';
    }
}

// ── FONT MANAGEMENT ──────────────────────────────────────────
async function loadFontList() {
    try {
        // Inject @font-face CSS so browser can actually render the fonts
        const cssRes = await fetch('/api/fonts/css');
        if (cssRes.ok) {
            const css = await cssRes.text();
            if (css.trim()) {
                const style = document.createElement('style');
                style.id = 'dynamic-font-faces';
                style.textContent = css;
                document.head.appendChild(style);
            }
        }

        const res = await fetch('/api/fonts');
        const data = await res.json();
        const sel = document.getElementById('font-family-select');
        if (!sel) return;
        const existing = Array.from(sel.options).map(o => o.value);
        data.fonts.forEach(f => {
            if (!existing.includes(f.name)) {
                const opt = document.createElement('option');
                opt.value = f.name;
                opt.textContent = f.name;
                sel.appendChild(opt);
            }
        });
    } catch(e) { console.warn('loadFontList:', e); }
}

async function uploadFont(input) {
    if (!input.files.length) return;
    const file = input.files[0];
    const fd = new FormData();
    fd.append('file', file);
    try {
        showToast('Uploading font...', 'info');
        const res = await fetch('/api/fonts/upload', { method: 'POST', body: fd });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        // Add to selector and select it
        const sel = document.getElementById('font-family-select');
        let opt = Array.from(sel.options).find(o => o.value === data.name);
        if (!opt) {
            opt = document.createElement('option');
            opt.value = data.name;
            opt.textContent = data.name;
            sel.appendChild(opt);
        }
        sel.value = data.name;
        showToast(`Font "${data.name}" uploaded`, 'success');
    } catch(e) {
        showToast('Upload failed: ' + e.message, 'error');
    }
    input.value = '';
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'success' ? 'check-circle' : type === 'error' ? 'x-circle' : 'info';
    toast.innerHTML = `<i class="ph ph-${icon}"></i> ${message}`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4500);
}

// ── JOB POLLING ───────────────────────────────────────────────
function pollMasterJob(jobId, { onComplete, onFail, statusEl }) {
    const poll = async () => {
        try {
            const res = await fetch('/api/jobs/' + jobId);
            const job = await res.json();
            const msg = job.message || 'Processing...';

            if (statusEl) {
                statusEl.className = 'card-status active';
                statusEl.innerHTML = `<i class="ph ph-spinner ph-spin"></i> ${msg} ${job.progress}%`;
            }

            if (job.status === 'completed') {
                if (onComplete) onComplete(job);
            } else if (job.status === 'failed') {
                if (onFail) onFail(job);
            } else {
                setTimeout(poll, 2500);
            }
        } catch (e) { setTimeout(poll, 5000); }
    };
    setTimeout(poll, 2000);
}

// ── PATH UTILITIES ────────────────────────────────────────────
function _normalizeCandidatePath(rawPath) {
    if (!rawPath) return '';
    const normalized = String(rawPath).replace(/\\/g, '/').trim();
    const cleaned = normalized.split('#')[0].split('?')[0];
    if (!cleaned) return '';
    try {
        const maybeUrl = cleaned.startsWith('//') ? ('http:' + cleaned) : cleaned;
        if (/^https?:\/\//i.test(maybeUrl)) {
            return new URL(maybeUrl).pathname || '';
        }
    } catch (_) {}
    return cleaned;
}

function buildPreviewVideoUrl(rawPath) {
    if (!rawPath) return '';
    const cleaned = _normalizeCandidatePath(rawPath);
    const lowered = cleaned.toLowerCase();
    let relativePath = '';

    if (!cleaned) return '';
    if (lowered.startsWith('/api/video-preview/')) return cleaned;
    if (lowered.startsWith('/api/video/'))         return cleaned;
    if (lowered.startsWith('api/video-preview/'))  return `/${cleaned}`;
    if (lowered.startsWith('api/video/'))          return `/${cleaned}`;

    if (lowered.startsWith('/outputs/')) {
        relativePath = cleaned.slice('/outputs/'.length);
    } else if (lowered.startsWith('outputs/')) {
        relativePath = cleaned.slice('outputs/'.length);
    } else {
        const marker = '/outputs/';
        const mi = lowered.lastIndexOf(marker);
        if (mi !== -1) {
            relativePath = cleaned.slice(mi + marker.length);
        } else {
            if (cleaned.startsWith('/') || /^[a-zA-Z]:\//.test(cleaned)) return '';
            relativePath = cleaned.replace(/^\/+/, '');
        }
    }

    relativePath = relativePath.replace(/^\/+/, '');
    if (!relativePath) return '';

    const encodedPath = encodeURI(relativePath).replace(/%2F/g, '/');
    if (isSafari()) return `/api/video/${encodedPath}`;
    return `/api/video-preview/${encodedPath}`;
}

function extractOutputRelativePath(rawPath) {
    if (!rawPath) return '';
    const cleaned = _normalizeCandidatePath(rawPath);
    const lowered = cleaned.toLowerCase();
    if (!cleaned) return '';
    if (lowered.startsWith('/api/video/'))    return cleaned.slice('/api/video/'.length).replace(/^\/+/, '');
    if (lowered.startsWith('api/video/'))     return cleaned.slice('api/video/'.length).replace(/^\/+/, '');
    if (lowered.startsWith('/api/outputs/')) return cleaned.slice('/api/outputs/'.length).replace(/^\/+/, '');
    if (lowered.startsWith('api/outputs/'))  return cleaned.slice('api/outputs/'.length).replace(/^\/+/, '');
    if (lowered.startsWith('/outputs/'))     return cleaned.slice('/outputs/'.length).replace(/^\/+/, '');
    if (lowered.startsWith('outputs/'))      return cleaned.slice('outputs/'.length).replace(/^\/+/, '');
    const mi = lowered.lastIndexOf('/outputs/');
    if (mi !== -1) return cleaned.slice(mi + '/outputs/'.length).replace(/^\/+/, '');
    return '';
}

function buildPreviewVideoCandidates(rawPath, downloadUrl) {
    const candidates = [];
    const pushUnique = (url) => { if (url && !candidates.includes(url)) candidates.push(url); };
    const apiPreviewUrl = buildPreviewVideoUrl(rawPath);
    const relativePath  = extractOutputRelativePath(rawPath);
    pushUnique(apiPreviewUrl);
    if (relativePath) pushUnique(`/api/video/${encodeURI(relativePath).replace(/%2F/g, '/')}`);
    if (relativePath) pushUnique(`/api/outputs/${encodeURI(relativePath).replace(/%2F/g, '/')}`);
    pushUnique(downloadUrl);
    return candidates;
}

function handleVideoPreviewError(videoEl) {
    if (!videoEl) return;
    const fallbackRaw = videoEl.dataset.fallbacks || '';
    if (!fallbackRaw) return;
    const fallbacks = fallbackRaw.split('|').filter(Boolean);
    if (!fallbacks.length) return;
    const nextUrl = fallbacks.shift();
    videoEl.dataset.fallbacks = fallbacks.join('|');
    videoEl.src = nextUrl;
    videoEl.load();
}

// ── MAIN JOB POLL ─────────────────────────────────────────────
const PIPELINE_LABELS = {
    'subtitle-status':  'Subtitle Pipeline',
    'legal-status':     'Legal Pipeline',
    'master-status':    'Mastering Pipeline',
    'gold-plan-status': 'Gold POS Pipeline',
    'batch-status':     'Batch Pipeline',
    'export-status':    'Export Pipeline',
};

async function pollJobStatus(jobId, statusBoxId) {
    const statusBox = document.getElementById(statusBoxId);

    const poll = async () => {
        try {
            const response = await fetch('/api/jobs/' + jobId);
            const job = await response.json();

            const progress = job.progress || 0;
            const msg      = job.message  || 'Processing...';
            const stepMatch = msg.match(/Step (\d\/\d): (.*)/);
            let stepHtml = '';
            if (stepMatch) {
                stepHtml = `<div class="status-compact-step"><i class="ph ph-circle-dashed ph-spin"></i> <b>Step ${stepMatch[1]}</b>: ${stepMatch[2]}</div>`;
            } else if (job.status === 'processing') {
                stepHtml = `<div class="status-compact-step"><i class="ph ph-gear ph-spin"></i> ${msg}</div>`;
            }

            // Update bottom statusbar
            updateStatusbar(progress, (PIPELINE_LABELS[statusBoxId] || 'Processing') + ' — ' + msg);

            let html = `
                <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:0.6rem;">
                    <div>
                        <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.2rem;">${PIPELINE_LABELS[statusBoxId] || 'Processing'}</div>
                        ${stepHtml}
                    </div>
                    <span style="font-size:1.3rem;font-weight:800;color:var(--primary);line-height:1;">${progress}%</span>
                </div>
                <div class="progress-bar" style="height:8px;margin-bottom:0.75rem;">
                    <div class="fill" style="width:${progress}%;box-shadow:0 0 10px var(--primary-glow);"></div>
                </div>
                <div style="font-size:0.82rem;">
                    <span class="${job.status === 'completed' ? 'status-completed' : ''} ${job.status === 'failed' ? 'status-failed' : ''}">
                        ${job.status === 'completed' ? '<i class="ph ph-check-circle"></i> Completed'
                          : job.status === 'failed'   ? '<i class="ph ph-x-circle"></i> Failed: ' + job.error
                          : '<i class="ph ph-lightning"></i> Processing...'}
                    </span>
                </div>
            `;

            if (job.status === 'completed') {
                const downloadUrl = `/api/download/${jobId}?t=${Date.now()}`;
                const resultPath  = (job.result_path || '').toLowerCase();
                const isVideo     = /\.(mp4|mov|qt|mkv|avi|webm)$/.test(resultPath);
                const hasMulti    = job.result_files && job.result_files.length > 1;

                if (isVideo && !hasMulti) {
                    const isProRes = /\.(mov|qt)$/.test(resultPath);
                    const mainLabel = job.result_files?.[0]?.label || 'Result Video';
                    const candidates = buildPreviewVideoCandidates(job.result_path, downloadUrl);
                    const videoUrl   = candidates[0] || downloadUrl;
                    const fallbacks  = candidates.slice(1).join('|');

                    // Auto-load into shared preview player
                    const mainPlayer = document.getElementById('main-preview-player');
                    const mainImg    = document.getElementById('main-preview-img');
                    const mainPH     = document.getElementById('main-preview-placeholder');
                    if (mainPlayer && !isProRes) {
                        mainPlayer.src = videoUrl;
                        mainPlayer.className = 'active';
                        if (mainImg) mainImg.className = '';
                        if (mainPH) mainPH.style.display = 'none';
                    }

                    html += `
                        <div style="margin-top:0.75rem;background:var(--panel-light);border-radius:var(--radius);overflow:hidden;border:1px solid var(--border);">
                            <div style="padding:0.6rem 0.85rem;border-bottom:1px solid var(--border);font-weight:600;font-size:0.82rem;color:var(--primary);display:flex;justify-content:space-between;align-items:center;">
                                <span><i class="ph ph-video"></i> ${mainLabel}</span>
                                ${isProRes ? '<span style="background:var(--warning);color:#000;padding:2px 7px;border-radius:4px;font-size:0.65rem;">PRO FORMAT</span>' : ''}
                            </div>
                            ${isProRes ? `
                                <div style="height:160px;background:#000;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:0.5rem;padding:1rem;text-align:center;color:var(--warning);">
                                    <i class="ph ph-monitor-play" style="font-size:2.5rem;"></i>
                                    <div style="font-size:0.8rem;">ProRes — Download to view</div>
                                </div>` : `
                                <video controls preload="auto" playsinline src="${videoUrl}" data-fallbacks="${fallbacks}"
                                       onerror="handleVideoPreviewError(this)"
                                       style="width:100%;max-height:240px;background:#000;display:block;"></video>`}
                            <div style="padding:0.75rem;">
                                <a href="${downloadUrl}" download class="btn btn-primary"
                                   style="width:100%;justify-content:center;text-decoration:none;font-size:0.82rem;">
                                    <i class="ph ph-download-simple"></i> Download
                                </a>
                            </div>
                        </div>`;
                } else if (job.result_files?.length > 0) {
                    html += '<div style="margin-top:0.75rem;display:flex;flex-direction:column;gap:0.45rem;">';
                    job.result_files.forEach((file, index) => {
                        const fileUrl   = `/api/download/${jobId}/${index}`;
                        const cands     = buildPreviewVideoCandidates(file.path, fileUrl);
                        const prevUrl   = cands[0] || fileUrl;
                        const prevFalls = cands.slice(1).join('|');
                        const filePath  = (file.path || '').toLowerCase();
                        const isProRes  = /\.(mov|qt)$/.test(filePath);
                        const isVideo   = file.type === 'video';
                        const canPreview = isVideo && !isProRes;
                        const loadCall  = canPreview
                            ? `onclick="loadInMainPreview('${prevUrl}','${prevFalls}','${(file.label||'Result').replace(/'/g,"\\'")}')" style="cursor:pointer;"`
                            : '';

                        html += `
                            <div class="result-card" ${loadCall} title="${canPreview ? 'Click to load in main preview' : ''}">
                                ${canPreview
                                    ? `<i class="ph ph-play-circle" style="color:var(--primary);font-size:1.2rem;flex-shrink:0;"></i>`
                                    : isVideo
                                        ? `<i class="ph ph-video"></i>`
                                        : `<i class="ph ph-file-text"></i>`}
                                <div style="flex:1;min-width:0;">
                                    <div class="rc-label">${file.label || 'Result'}</div>
                                    ${canPreview
                                        ? `<div style="font-size:0.68rem;color:var(--text-muted);margin-top:1px;">Click to preview · <a href="${fileUrl}" download style="color:var(--primary);text-decoration:none;">Download</a></div>`
                                        : `<div style="font-size:0.68rem;color:var(--text-muted);margin-top:1px;">${isProRes ? 'ProRes — download only' : 'Non-video'}</div>`}
                                </div>
                                ${canPreview ? `
                                <button class="btn btn-ghost btn-sm btn-icon"
                                        onclick="event.stopPropagation(); window.location='/api/download/${jobId}/${index}'"
                                        title="Download" style="flex-shrink:0;">
                                    <i class="ph ph-download-simple"></i>
                                </button>` : `
                                <a href="${fileUrl}" download class="btn btn-ghost btn-sm btn-icon" style="text-decoration:none;flex-shrink:0;">
                                    <i class="ph ph-download-simple"></i>
                                </a>`}
                            </div>`;
                    });
                    html += '</div>';
                }

                // Re-enable buttons
                const goldRenderBtn = document.getElementById('gold-render-btn');
                const goldPlanBtn   = document.getElementById('gold-plan-btn');
                if (goldRenderBtn) goldRenderBtn.disabled = false;
                if (goldPlanBtn)   goldPlanBtn.disabled   = false;
                document.querySelectorAll('button[type="submit"]').forEach(b => b.disabled = false);
                updateStatusbar(100, (PIPELINE_LABELS[statusBoxId] || 'Pipeline') + ' — Completed');

            } else if (job.status === 'failed') {
                const goldRenderBtn = document.getElementById('gold-render-btn');
                const goldPlanBtn   = document.getElementById('gold-plan-btn');
                if (goldRenderBtn) goldRenderBtn.disabled = false;
                if (goldPlanBtn)   goldPlanBtn.disabled   = false;
                document.querySelectorAll('button[type="submit"]').forEach(b => b.disabled = false);
                updateStatusbar(0, 'Failed — ' + (job.error || 'Unknown error'));
            }

            if (statusBox) statusBox.innerHTML = html;

            if (job.status !== 'completed' && job.status !== 'failed') {
                setTimeout(poll, 2000);
            }
        } catch (e) {
            console.error('Polling error', e);
        }
    };

    poll();
}

// ── FORMAT INFO ───────────────────────────────────────────────
function updateFormatInfo(val) {
    const infoText = document.getElementById('format-info-text');
    const infoBox  = document.getElementById('format-info-box');
    if (!infoText || !infoBox) return;

    const dict = I18N[currentLang] || I18N['en'];
    const map = {
        mp4_standard: [dict['sub.info.standard']   || 'Ideal for web sharing and general review.', 'var(--text-muted)'],
        prores:       [dict['sub.info.prores']      || 'High quality for post-production.', 'var(--warning)'],
        mp4_20mbps:   [dict['sub.info.highbitrate'] || 'High bitrate for broadcast or archive.', 'var(--success)'],
        ae_package:   [dict['sub.info.ae']          || 'Full asset bundle for After Effects.', 'var(--primary)'],
    };
    const [text, color] = map[val] || [dict['sub.info.standard'] || 'Standard output.', 'var(--text-muted)'];
    infoText.textContent = text;
    infoBox.style.color  = color;
}

// ── MODAL ─────────────────────────────────────────────────────
function openModal(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('active');
}
function closeModal(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('active');
}
function switchModalTab(tab) {
    document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.modal-panel').forEach(p => p.classList.remove('active'));
    document.getElementById('m-tab-' + tab).classList.add('active');
    document.getElementById('m-panel-' + tab).classList.add('active');
}

// ── CANVAS (Inpainting) ───────────────────────────────────────
function initCanvas(canvasId, placeholderId, uploadId, btnId, clearBtnId, brushSizeId, brushValId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return {};
    const ctx        = canvas.getContext('2d');
    let isDrawing    = false;
    let brushSize    = 25;
    let originalImage = null;
    let maskCanvas   = document.createElement('canvas');
    let maskCtx      = maskCanvas.getContext('2d');
    let uploadedFile = null;

    const brushEl = document.getElementById(brushSizeId);
    if (brushEl) {
        brushEl.oninput = (e) => {
            brushSize = parseInt(e.target.value);
            const valEl = document.getElementById(brushValId);
            if (valEl) valEl.textContent = brushSize;
        };
    }

    const handleFile = (file) => {
        uploadedFile = file;
        const reader = new FileReader();
        reader.onload = (ev) => {
            const img = new Image();
            img.onload = () => {
                const maxW = 700;
                let w = img.width, h = img.height;
                if (w > maxW) { h = Math.round((maxW / w) * h); w = maxW; }
                canvas.width  = w; canvas.height = h;
                canvas.style.display = 'block';
                const phEl = document.getElementById(placeholderId);
                if (phEl) phEl.style.display = 'none';
                ctx.drawImage(img, 0, 0, w, h);
                originalImage = img;
                maskCanvas.width = w; maskCanvas.height = h;
                maskCtx.fillStyle = 'black';
                maskCtx.fillRect(0, 0, w, h);
                const btn = document.getElementById(btnId);
                if (btn) btn.disabled = false;
            };
            img.src = ev.target.result;
        };
        reader.readAsDataURL(file);
    };

    const uploadEl = document.getElementById(uploadId);
    if (uploadEl) uploadEl.onchange = (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); };

    const draw = (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvas.width  / rect.width);
        const y = (e.clientY - rect.top)  * (canvas.height / rect.height);
        ctx.globalAlpha = 0.5; ctx.fillStyle = '#ff3366';
        ctx.beginPath(); ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2); ctx.fill();
        maskCtx.fillStyle = 'white';
        maskCtx.beginPath(); maskCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2); maskCtx.fill();
    };

    canvas.onmousedown = (e) => { isDrawing = true; draw(e); };
    canvas.onmousemove = (e) => { if (isDrawing) draw(e); };
    window.addEventListener('mouseup', () => { isDrawing = false; });

    const clearBtn = document.getElementById(clearBtnId);
    if (clearBtn) {
        clearBtn.onclick = () => {
            if (!originalImage) return;
            ctx.drawImage(originalImage, 0, 0, canvas.width, canvas.height);
            maskCtx.fillStyle = 'black';
            maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
        };
    }

    return { getMask: () => maskCanvas, getFile: () => uploadedFile, setFile: handleFile };
}

const imgInpaint = initCanvas('inpaint-canvas', 'inpaint-placeholder', 'm-inpaint-upload', 'btn-m-inpaint', 'btn-m-clear', 'brush-size', 'brush-size-val');
const vidInpaint = initCanvas('video-inpaint-canvas', 'v-inpaint-placeholder', 'm-video-upload', 'btn-v-inpaint', 'btn-v-clear', 'v-brush-size', 'v-brush-size-val');

// Image inpaint submit
const btnMInpaint = document.getElementById('btn-m-inpaint');
if (btnMInpaint) {
    btnMInpaint.onclick = async () => {
        btnMInpaint.disabled = true;
        btnMInpaint.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Processing...';
        try {
            const maskBlob = await new Promise(r => imgInpaint.getMask().toBlob(r, 'image/png'));
            const fd = new FormData();
            fd.append('image', imgInpaint.getFile());
            fd.append('mask', maskBlob, 'mask.png');
            const res  = await fetch('/api/master/inpaint-image', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.result_url) {
                document.getElementById('m-result-area').style.display = 'block';
                document.getElementById('m-result-img').src  = data.result_url + '?t=' + Date.now();
                document.getElementById('m-download-link').href = data.result_url;
            }
        } catch (e) { alert(e.message); }
        btnMInpaint.disabled = false;
        btnMInpaint.innerHTML = '<i class="ph ph-paint-brush"></i> Remove Object';
    };
}

// Video inpaint
let currentVideoPath = null;
const mVideoUpload = document.getElementById('m-video-upload');
if (mVideoUpload) {
    mVideoUpload.addEventListener('change', async (e) => {
        if (!e.target.files[0]) return;
        const btn = document.getElementById('btn-v-inpaint');
        if (btn) btn.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Extracting frame...';
        const fd = new FormData();
        fd.append('video', e.target.files[0]);
        try {
            const res  = await fetch('/api/master/inpaint-video-preview', { method: 'POST', body: fd });
            const data = await res.json();
            currentVideoPath = data.video_path;
            const imgRes = await fetch(data.preview_url);
            const blob   = await imgRes.blob();
            if (vidInpaint.setFile) vidInpaint.setFile(new File([blob], 'preview.png'));
            if (btn) btn.innerHTML = '<i class="ph ph-film-strip"></i> Process Video';
        } catch (e) {
            alert(e.message);
            if (btn) btn.innerHTML = 'Process Video';
        }
    });
}

const btnVInpaint = document.getElementById('btn-v-inpaint');
if (btnVInpaint) {
    btnVInpaint.onclick = async () => {
        if (!currentVideoPath) return;
        btnVInpaint.disabled = true;
        const statusArea = document.getElementById('v-status-area');
        if (statusArea) statusArea.style.display = 'block';
        try {
            const maskBlob = await new Promise(r => vidInpaint.getMask().toBlob(r, 'image/png'));
            const fd = new FormData();
            fd.append('video_path', currentVideoPath);
            fd.append('mask', maskBlob, 'mask.png');
            const res  = await fetch('/api/master/inpaint-video', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.job_id) pollVideoInpaint(data.job_id);
        } catch (e) {
            alert(e.message);
            btnVInpaint.disabled = false;
        }
    };
}

function pollVideoInpaint(jobId) {
    const poll = async () => {
        const res = await fetch('/api/jobs/' + jobId);
        const job = await res.json();
        const fill = document.getElementById('v-progress-fill');
        if (fill) fill.style.width = job.progress + '%';
        if (job.status === 'completed') {
            const resultArea = document.getElementById('v-result-area');
            const resultPrev = document.getElementById('v-result-preview');
            const dlLink     = document.getElementById('v-download-link');
            if (resultArea) resultArea.style.display = 'block';
            if (resultPrev) resultPrev.src = job.result_url;
            if (dlLink)     dlLink.href    = job.result_url;
            if (btnVInpaint) btnVInpaint.disabled = false;
        } else if (job.status === 'failed') {
            alert('Error: ' + job.error);
            if (btnVInpaint) btnVInpaint.disabled = false;
        } else {
            setTimeout(poll, 2000);
        }
    };
    poll();
}

// ── AUTO-RESET STATUS ON FILTER CHANGE ───────────────────────
['gold-formats-wrap', 'gold-durations-wrap', 'gold-codecs-wrap', 'gold-lines-wrap'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
        el.addEventListener('change', () => {
            const status = document.getElementById('gold-plan-status');
            if (status && (status.innerHTML.includes('Ready') || status.innerHTML.includes('Batch Plan'))) {
                status.innerHTML = '';
            }
        });
    }
});

loadFontList();
setLang('en');

// ── DRAG & DROP SETUP ─────────────────────────────────────────
setupVideoDropZone('video-upload-1', 'input[type="file"]', updatePreview);
setupVideoDropZone('video-upload-2', 'input[type="file"]');
setupVideoDropZone('video-upload-3', 'input[type="file"]', updateMasterPreview);
setupExcelDropZone('excel-upload-container', 'excel-translation-input');
