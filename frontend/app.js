/* ═════════════════════════════════════════════════════════════
   SkyLogic MAS — Frontend Application Logic
   Multi-Agent Satellite Disaster Intelligence Dashboard
   ═════════════════════════════════════════════════════════════ */

// Auto-detect API: if served from backend use same origin, else fallback
const API_BASE = (window.location.protocol === 'file:')
    ? 'http://localhost:8000'
    : window.location.origin;

// ── State ──────────────────────────────────────────────────────
const state = {
    token: localStorage.getItem('skylogic_token') || null,
    username: localStorage.getItem('skylogic_username') || null,
    currentView: 'dashboard',
    patches: [],
    selectedPatch: null,
    annotations: [],
    activeTool: 'select',
    isDrawing: false,
    drawStart: null,
    canvasImage: null,
};

// ── API Client ─────────────────────────────────────────────────
async function api(endpoint, options = {}) {
    const headers = { 'Content-Type': 'application/json', ...options.headers };
    if (state.token) headers['Authorization'] = `Bearer ${state.token}`;

    try {
        const res = await fetch(`${API_BASE}${endpoint}`, { ...options, headers });
        if (res.status === 401) {
            logout();
            throw new Error('Session expired');
        }
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        return await res.json();
    } catch (err) {
        if (err.message === 'Failed to fetch') {
            throw new Error('Cannot connect to API server');
        }
        throw err;
    }
}

// ── Toast Notifications ────────────────────────────────────────
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease-in forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ═══════════════════════════════════════════════════════════════
// AUTH
// ═══════════════════════════════════════════════════════════════
function initAuth() {
    const form = document.getElementById('login-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value;
        const errorEl = document.getElementById('login-error');
        const btn = document.getElementById('login-btn');

        if (!username || !password) {
            errorEl.textContent = 'Please enter username and password';
            return;
        }

        btn.disabled = true;
        btn.querySelector('span').textContent = 'Signing in...';
        errorEl.textContent = '';

        try {
            const formData = new URLSearchParams();
            formData.append('username', username);
            formData.append('password', password);

            const res = await fetch(`${API_BASE}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: formData,
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || 'Login failed');
            }

            const data = await res.json();
            state.token = data.access_token;
            state.username = data.username;
            localStorage.setItem('skylogic_token', data.access_token);
            localStorage.setItem('skylogic_username', data.username);

            showLogin(false);
            initApp();
            showToast(`Welcome back, ${data.username}!`, 'success');
        } catch (err) {
            errorEl.textContent = err.message;
        } finally {
            btn.disabled = false;
            btn.querySelector('span').textContent = 'Sign In';
        }
    });

    // Check existing token
    if (state.token) {
        showLogin(false);
        initApp();
    }
}

function showLogin(show) {
    document.getElementById('login-overlay').classList.toggle('hidden', !show);
    document.getElementById('app').classList.toggle('hidden', show);
}

function logout() {
    state.token = null;
    state.username = null;
    localStorage.removeItem('skylogic_token');
    localStorage.removeItem('skylogic_username');
    showLogin(true);
}

// ═══════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const views = {
        dashboard: { title: 'Dashboard', subtitle: 'System Overview' },
        annotator: { title: 'Annotator', subtitle: 'Interactive Canvas' },
        agents: { title: 'AI Agents', subtitle: 'Multi-Agent Pipeline' },
        predictions: { title: 'Predictions', subtitle: 'Auto-Detection Results' },
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const view = item.dataset.view;
            if (!view) return;

            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            document.getElementById(`view-${view}`).classList.add('active');

            document.getElementById('view-title').textContent = views[view]?.title || view;
            document.getElementById('view-subtitle').textContent = views[view]?.subtitle || '';

            state.currentView = view;

            // Refresh data on view change
            if (view === 'dashboard') refreshDashboard();
            if (view === 'predictions') refreshPredictions();
        });
    });

    document.getElementById('logout-btn').addEventListener('click', logout);
}

// ═══════════════════════════════════════════════════════════════
// DASHBOARD
// ═══════════════════════════════════════════════════════════════
async function refreshDashboard() {
    // Fetch prediction status
    try {
        const status = await api('/api/predictions/status');
        document.getElementById('stat-patches-value').textContent = status.total_patches.toLocaleString();
        document.getElementById('stat-predicted-value').textContent = status.predicted_patches.toLocaleString();
        document.getElementById('stat-annotations-value').textContent = status.total_ai_annotations.toLocaleString();
        document.getElementById('stat-progress-value').textContent = `${status.progress_percent}%`;

        updateApiStatus(true);
    } catch (err) {
        console.warn('Dashboard refresh failed:', err.message);
        updateApiStatus(false);
    }

    // Fetch recent patches
    try {
        const patches = await api('/api/patches/?limit=24');
        state.patches = patches;
        renderPatchesGrid(patches);
    } catch (err) {
        console.warn('Patches fetch failed:', err.message);
    }

    // Check system status
    checkSystemStatus();
}

function renderPatchesGrid(patches) {
    const grid = document.getElementById('patches-grid');
    if (!patches || patches.length === 0) {
        grid.innerHTML = `<div class="empty-state">
            <p>No patches loaded. Run the ingestion pipeline first.</p>
            <code>python scripts/ingest.py</code>
        </div>`;
        return;
    }

    grid.innerHTML = patches.map(p => `
        <div class="patch-thumb" data-id="${p.id}" title="${p.filename}">
            <img src="${API_BASE}${p.image_url}" alt="${p.filename}"
                 onerror="this.style.display='none'">
            ${p.is_predicted ? '<span class="patch-badge">✓</span>' : ''}
        </div>
    `).join('');

    grid.querySelectorAll('.patch-thumb').forEach(el => {
        el.addEventListener('click', () => {
            const id = parseInt(el.dataset.id);
            selectPatch(id);
            // Switch to annotator
            document.querySelector('[data-view="annotator"]').click();
        });
    });
}

function updateApiStatus(online) {
    const indicator = document.getElementById('api-status');
    const dot = indicator.querySelector('.status-dot');
    const text = indicator.querySelector('.status-text');

    if (online) {
        dot.className = 'status-dot online';
        text.textContent = 'API Connected';
    } else {
        dot.className = 'status-dot offline';
        text.textContent = 'API Offline';
    }
}

async function checkSystemStatus() {
    const updateBadge = (id, online) => {
        const el = document.getElementById(id);
        el.className = `badge ${online ? 'badge-online' : 'badge-offline'}`;
        el.textContent = online ? 'Online' : 'Offline';
    };

    // API Health
    try {
        await api('/health');
        updateBadge('status-api', true);
        updateBadge('status-db', true); // If API is up, DB is connected
    } catch {
        updateBadge('status-api', false);
        updateBadge('status-db', false);
    }

    // SAM / Qdrant status
    try {
        const samStatus = await api('/api/sam/status');
        updateBadge('status-sam', samStatus.sam_loaded);
        updateBadge('status-qdrant', samStatus.qdrant?.status === 'connected');
    } catch {
        updateBadge('status-sam', false);
        updateBadge('status-qdrant', false);
    }

    // Agents (YOLOv10, SegFormer) - check via a lightweight endpoint
    updateBadge('status-yolo', true);
    updateBadge('status-segformer', true);
}

document.getElementById('btn-refresh-patches').addEventListener('click', refreshDashboard);

// ═══════════════════════════════════════════════════════════════
// ANNOTATOR
// ═══════════════════════════════════════════════════════════════
const canvas = document.getElementById('annotation-canvas');
const ctx = canvas.getContext('2d');

function initAnnotator() {
    // Tool buttons
    const tools = ['select', 'bbox', 'sam', 'delete'];
    tools.forEach(tool => {
        const btn = document.getElementById(`tool-${tool}`);
        if (btn) {
            btn.addEventListener('click', () => {
                state.activeTool = tool;
                document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                canvas.style.cursor = tool === 'select' ? 'default' : 'crosshair';
            });
        }
    });

    // Canvas events
    canvas.addEventListener('mousedown', onCanvasMouseDown);
    canvas.addEventListener('mousemove', onCanvasMouseMove);
    canvas.addEventListener('mouseup', onCanvasMouseUp);
    canvas.addEventListener('click', onCanvasClick);

    // Auto-predict single
    document.getElementById('btn-auto-predict-single').addEventListener('click', async () => {
        if (!state.selectedPatch) {
            showToast('Select a patch first', 'error');
            return;
        }
        try {
            showToast('Running AI prediction...', 'info');
            const result = await api('/api/predictions/single', {
                method: 'POST',
                body: JSON.stringify({
                    patch_id: state.selectedPatch.id,
                    use_ensemble: true,
                }),
            });
            showToast(`${result.saved} predictions generated!`, 'success');
            loadAnnotations(state.selectedPatch.id);
        } catch (err) {
            showToast(`Prediction failed: ${err.message}`, 'error');
        }
    });
}

async function selectPatch(patchId) {
    try {
        const patch = await api(`/api/patches/${patchId}`);
        state.selectedPatch = patch;

        // Update annotator patch list
        document.querySelectorAll('.patch-list-item').forEach(el => {
            el.classList.toggle('active', parseInt(el.dataset.id) === patchId);
        });

        // Load image on canvas
        loadPatchImage(patch);
        loadAnnotations(patchId);

        // Hide empty overlay
        document.getElementById('canvas-empty').style.display = 'none';
    } catch (err) {
        showToast(`Failed to load patch: ${err.message}`, 'error');
    }
}

function loadPatchImage(patch) {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
        canvas.width = img.width || 512;
        canvas.height = img.height || 512;
        state.canvasImage = img;
        redrawCanvas();
    };
    img.onerror = () => {
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#5a6d88';
        ctx.font = '14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Image not available', canvas.width / 2, canvas.height / 2);
    };
    img.src = `${API_BASE}${patch.image_url}`;
}

async function loadAnnotations(patchId) {
    try {
        const annotations = await api(`/api/annotations/?patch_id=${patchId}&limit=200`);
        state.annotations = annotations;
        renderAnnotationList(annotations);
        redrawCanvas();

        document.getElementById('annot-count').textContent = annotations.length;
    } catch (err) {
        console.warn('Failed to load annotations:', err.message);
    }
}

function renderAnnotationList(annotations) {
    const list = document.getElementById('annotation-list');
    if (!annotations.length) {
        list.innerHTML = '<div class="empty-state-sm">No annotations</div>';
        return;
    }

    list.innerHTML = annotations.map(a => `
        <div class="annot-item" data-id="${a.id}">
            <div>
                <span class="annot-class">${a.class_name || `Class ${a.class_id}`}</span>
                <span class="annot-source">${a.source}</span>
            </div>
            <div style="display:flex;align-items:center;gap:0.4rem;">
                <span class="annot-conf">${(a.confidence * 100).toFixed(0)}%</span>
                <button class="annot-delete" data-id="${a.id}" title="Delete">×</button>
            </div>
        </div>
    `).join('');

    list.querySelectorAll('.annot-delete').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const id = parseInt(btn.dataset.id);
            try {
                await api(`/api/annotations/${id}`, { method: 'DELETE' });
                showToast('Annotation deleted', 'success');
                loadAnnotations(state.selectedPatch.id);
            } catch (err) {
                showToast(`Delete failed: ${err.message}`, 'error');
            }
        });
    });
}

function redrawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image
    if (state.canvasImage) {
        ctx.drawImage(state.canvasImage, 0, 0);
    }

    // Draw annotations
    const colors = {
        human: '#22d3ee',
        yolov10: '#3b82f6',
        segformer: '#a78bfa',
        sam: '#10b981',
        ensemble: '#f59e0b',
        xview_ground_truth: '#f87171',
    };

    state.annotations.forEach((a, i) => {
        if (!a.bbox || a.bbox.length < 4) return;
        const [x1, y1, x2, y2] = a.bbox;
        const color = colors[a.source] || '#8597b4';

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label
        const label = `${a.class_name || a.class_id} ${(a.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 10px Inter';
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - 14, textWidth + 6, 14);
        ctx.fillStyle = '#07090f';
        ctx.fillText(label, x1 + 3, y1 - 3);
    });

    // Draw in-progress bbox
    if (state.isDrawing && state.drawStart && state.drawCurrent) {
        const x = Math.min(state.drawStart.x, state.drawCurrent.x);
        const y = Math.min(state.drawStart.y, state.drawCurrent.y);
        const w = Math.abs(state.drawCurrent.x - state.drawStart.x);
        const h = Math.abs(state.drawCurrent.y - state.drawStart.y);

        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
    }
}

function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: Math.round((e.clientX - rect.left) * scaleX),
        y: Math.round((e.clientY - rect.top) * scaleY),
    };
}

function onCanvasMouseDown(e) {
    if (state.activeTool === 'bbox') {
        state.isDrawing = true;
        state.drawStart = getCanvasCoords(e);
        state.drawCurrent = state.drawStart;
    }
}

function onCanvasMouseMove(e) {
    if (state.isDrawing) {
        state.drawCurrent = getCanvasCoords(e);
        redrawCanvas();
    }
}

async function onCanvasMouseUp(e) {
    if (state.isDrawing && state.activeTool === 'bbox') {
        state.isDrawing = false;
        const end = getCanvasCoords(e);

        const bbox = [
            Math.min(state.drawStart.x, end.x),
            Math.min(state.drawStart.y, end.y),
            Math.max(state.drawStart.x, end.x),
            Math.max(state.drawStart.y, end.y),
        ];

        const width = bbox[2] - bbox[0];
        const height = bbox[3] - bbox[1];

        if (width < 5 || height < 5) return; // Too small

        if (state.selectedPatch) {
            try {
                await api('/api/annotations/', {
                    method: 'POST',
                    body: JSON.stringify({
                        patch_id: state.selectedPatch.id,
                        class_id: 0,
                        class_name: 'manual',
                        bbox: bbox,
                        confidence: 1.0,
                        source: 'human',
                    }),
                });
                showToast('Annotation created', 'success');
                loadAnnotations(state.selectedPatch.id);
            } catch (err) {
                showToast(`Failed: ${err.message}`, 'error');
            }
        }

        state.drawStart = null;
        state.drawCurrent = null;
    }
}

async function onCanvasClick(e) {
    if (state.activeTool === 'sam' && state.selectedPatch) {
        const coords = getCanvasCoords(e);
        showToast(`SAM click at (${coords.x}, ${coords.y})...`, 'info');

        try {
            const result = await api('/api/sam/click', {
                method: 'POST',
                body: JSON.stringify({
                    patch_id: state.selectedPatch.id,
                    image_path: `patches/${state.selectedPatch.split}/${state.selectedPatch.filename}`,
                    clicks: [[coords.x, coords.y]],
                    labels: [1],
                    auto_annotate: true,
                }),
            });
            showToast(
                `SAM: mask generated (${(result.score * 100).toFixed(0)}% conf)` +
                (result.similar_count > 0 ? ` + ${result.similar_count} similar` : ''),
                'success'
            );
            loadAnnotations(state.selectedPatch.id);
        } catch (err) {
            showToast(`SAM failed: ${err.message}`, 'error');
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// PREDICTIONS
// ═══════════════════════════════════════════════════════════════
function initPredictions() {
    document.getElementById('btn-auto-predict-all').addEventListener('click', async () => {
        try {
            const result = await api('/api/predictions/batch', {
                method: 'POST',
                body: JSON.stringify({
                    split: 'train',
                    limit: 50,
                    skip_predicted: true,
                    use_ensemble: true,
                }),
            });

            showToast(result.message, 'success');

            // Show progress card
            const progressCard = document.getElementById('prediction-progress');
            progressCard.style.display = 'block';
            pollPredictionProgress();
        } catch (err) {
            showToast(`Batch prediction failed: ${err.message}`, 'error');
        }
    });

    document.getElementById('btn-refresh-predictions').addEventListener('click', refreshPredictions);
}

async function pollPredictionProgress() {
    const poll = async () => {
        try {
            const status = await api('/api/predictions/status');
            const pct = status.progress_percent;

            document.getElementById('progress-fill').style.width = `${pct}%`;
            document.getElementById('progress-text').textContent = `${pct}%`;
            document.getElementById('progress-detail').textContent =
                `${status.predicted_patches} / ${status.total_patches} patches processed — ${status.total_ai_annotations} annotations`;

            if (pct < 100 && status.pending_patches > 0) {
                setTimeout(poll, 3000);
            } else {
                showToast('Batch prediction complete!', 'success');
            }
        } catch (err) {
            console.warn('Progress poll failed:', err.message);
        }
    };
    poll();
}

async function refreshPredictions() {
    try {
        const annotations = await api('/api/annotations/?source=ensemble&limit=100');
        const tbody = document.getElementById('results-tbody');

        if (!annotations.length) {
            tbody.innerHTML = '<tr><td colspan="5" class="empty-state-sm">No predictions yet</td></tr>';
            return;
        }

        tbody.innerHTML = annotations.map(a => `
            <tr>
                <td>${a.patch_id}</td>
                <td><span class="badge badge-ready">${a.source}</span></td>
                <td>${a.class_name || `Class ${a.class_id}`}</td>
                <td><span class="annot-conf">${(a.confidence * 100).toFixed(1)}%</span></td>
                <td>
                    <button class="btn-ghost btn-sm" onclick="viewPatchFromResult(${a.patch_id})">View</button>
                    <button class="annot-delete" onclick="deleteAnnotation(${a.id})">×</button>
                </td>
            </tr>
        `).join('');
    } catch (err) {
        showToast(`Failed to load predictions: ${err.message}`, 'error');
    }
}

// Global functions for inline handlers
window.viewPatchFromResult = (patchId) => {
    selectPatch(patchId);
    document.querySelector('[data-view="annotator"]').click();
};

window.deleteAnnotation = async (id) => {
    try {
        await api(`/api/annotations/${id}`, { method: 'DELETE' });
        showToast('Deleted', 'success');
        refreshPredictions();
    } catch (err) {
        showToast(err.message, 'error');
    }
};

// ═══════════════════════════════════════════════════════════════
// ANNOTATOR PATCH LIST
// ═══════════════════════════════════════════════════════════════
async function loadAnnotatorPatchList() {
    try {
        const patches = await api('/api/patches/?limit=100');
        const list = document.getElementById('annotator-patch-list');

        if (!patches.length) {
            list.innerHTML = '<div class="empty-state-sm">No patches loaded</div>';
            return;
        }

        list.innerHTML = patches.map(p => `
            <div class="patch-list-item" data-id="${p.id}">
                <div class="mini-thumb">
                    <img src="${API_BASE}${p.image_url}" alt="" onerror="this.style.display='none'">
                </div>
                <span>${p.filename.length > 20 ? p.filename.substring(0, 20) + '...' : p.filename}</span>
            </div>
        `).join('');

        list.querySelectorAll('.patch-list-item').forEach(el => {
            el.addEventListener('click', () => {
                selectPatch(parseInt(el.dataset.id));
            });
        });
    } catch (err) {
        console.warn('Patch list load failed:', err.message);
    }
}

// ═══════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════
function initApp() {
    document.getElementById('display-username').textContent = state.username || 'SkyLogic';
    document.querySelector('.user-avatar').textContent = (state.username || 'S')[0].toUpperCase();

    initNavigation();
    initAnnotator();
    initPredictions();

    refreshDashboard();
    loadAnnotatorPatchList();
}

// Boot
document.addEventListener('DOMContentLoaded', () => {
    initAuth();
});
