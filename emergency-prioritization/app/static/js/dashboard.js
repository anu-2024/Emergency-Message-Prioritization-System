// Dashboard client logic. Vanilla JS + fetch() against the FastAPI backend.
// Auth token is kept in localStorage (standard for a browser-rendered SPA-lite
// dashboard) and attached as a Bearer token on every API call.

function getToken() { return window.localStorage.getItem('access_token'); }
function getRole() { return window.localStorage.getItem('role'); }

function authHeaders(extra = {}) {
    return { 'Authorization': `Bearer ${getToken()}`, ...extra };
}

function logout() {
    window.localStorage.removeItem('access_token');
    window.localStorage.removeItem('role');
    window.location.href = '/login';
}

function requireAuth() {
    if (!getToken()) {
        window.location.href = '/login';
        return false;
    }
    document.getElementById('userLabel').textContent = `Role: ${getRole()}`;
    return true;
}

async function apiGet(path) {
    const resp = await fetch(path, { headers: authHeaders() });
    if (resp.status === 401) { logout(); return null; }
    return resp.json();
}

async function apiPost(path, body) {
    const resp = await fetch(path, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(body),
    });
    if (resp.status === 401) { logout(); return null; }
    return resp.json();
}

const URGENCY_ORDER = { Critical: 0, High: 1, Medium: 2, Low: 3 };

async function loadStats() {
    const stats = await apiGet('/api/messages/stats/summary');
    if (!stats) return;
    const row = document.getElementById('statsRow');
    row.innerHTML = `
        ${statCard('Total Messages', stats.total_messages)}
        ${statCard('Duplicates Flagged', stats.duplicate_count)}
        ${statCard('Critical', stats.by_urgency.Critical || 0)}
        ${statCard('High', stats.by_urgency.High || 0)}
    `;
}

function statCard(label, value) {
    return `<div class="col-md-3 mb-2">
        <div class="card stat-card p-3">
            <div class="text-muted small">${label}</div>
            <div class="value">${value}</div>
        </div>
    </div>`;
}

async function loadQueue() {
    const messages = await apiGet('/api/messages/queue');
    if (!messages) return;

    const urgencyFilter = document.getElementById('filterUrgency').value;
    const categoryFilter = document.getElementById('filterCategory').value;
    const sourceFilter = document.getElementById('filterSource').value;

    const filtered = messages.filter(m =>
        (!urgencyFilter || m.urgency === urgencyFilter) &&
        (!categoryFilter || m.category === categoryFilter) &&
        (!sourceFilter || m.final_priority_source === sourceFilter)
    );

    const tbody = document.getElementById('queueBody');
    tbody.innerHTML = filtered.map(m => {
        const priority = m.human_override_priority ?? m.rl_priority ?? m.rule_based_priority ?? 0;
        return `<tr>
            <td><strong>${priority.toFixed(3)}</strong></td>
            <td><span class="badge badge-${m.urgency}">${m.urgency || '-'}</span></td>
            <td>${m.category || '-'}</td>
            <td class="msg-text-cell">${escapeHtml(m.raw_text)}</td>
            <td>${m.is_duplicate ? '<span class="dup-flag">DUP</span>' : ''}</td>
            <td><span class="badge bg-secondary">${m.final_priority_source || '-'}</span></td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="openDetail('${m.message_id}')">View</button>
            </td>
        </tr>`;
    }).join('');
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

async function submitMessage() {
    const text = document.getElementById('newMessageText').value.trim();
    const resultBox = document.getElementById('submitResult');
    if (!text) return;
    resultBox.textContent = 'Analyzing...';
    const result = await apiPost('/api/messages/submit', { text });
    if (!result) return;
    if (result.detail) {
        resultBox.innerHTML = `<span class="text-danger">${result.detail}</span>`;
        return;
    }
    resultBox.innerHTML = `<span class="text-success">Added as ${result.message_id} — 
        ${result.category} / ${result.urgency}</span>`;
    document.getElementById('newMessageText').value = '';
    loadQueue();
    loadStats();
}

async function openDetail(messageId) {
    const m = await apiGet(`/api/messages/${messageId}`);
    if (!m) return;
    const body = document.getElementById('detailBody');
    body.innerHTML = `
        <p><strong>Message:</strong> ${escapeHtml(m.raw_text)}</p>
        <div class="row">
            <div class="col-6">
                <p><strong>Category:</strong> ${m.category} (${(m.category_confidence*100).toFixed(1)}% confidence)</p>
                <p><strong>Urgency:</strong> ${m.urgency}</p>
                <p><strong>Locations:</strong> ${(m.locations || []).join(', ') || 'none detected'}</p>
                <p><strong>Assistance types:</strong> ${(m.assistance_types || []).join(', ') || 'none detected'}</p>
            </div>
            <div class="col-6">
                <p><strong>Rule-based priority:</strong> ${m.rule_based_priority?.toFixed(3) ?? '-'}</p>
                <p><strong>RL priority:</strong> ${m.rl_priority?.toFixed(3) ?? 'not available'}</p>
                <p><strong>Human override:</strong> ${m.human_override_priority?.toFixed(3) ?? 'none'}</p>
                <p><strong>Duplicate of:</strong> ${m.duplicate_of_message_id || 'n/a'}</p>
            </div>
        </div>
        <hr>
        <div class="row g-2 align-items-end">
            <div class="col-auto">
                <label class="form-label small">Override priority (0-1)</label>
                <input type="number" min="0" max="1" step="0.01" class="form-control form-control-sm" id="overrideInput" style="width:100px">
            </div>
            <div class="col-auto">
                <button class="btn btn-sm btn-warning" onclick="overridePriority('${m.message_id}')">Apply Override</button>
            </div>
            <div class="col-auto">
                <button class="btn btn-sm btn-outline-success" onclick="recordAction('${m.message_id}', 'assign')">Assign</button>
                <button class="btn btn-sm btn-outline-danger" onclick="recordAction('${m.message_id}', 'escalate')">Escalate</button>
                <button class="btn btn-sm btn-outline-secondary" onclick="recordAction('${m.message_id}', 'resolve')">Resolve</button>
            </div>
        </div>
        <p class="text-muted small mt-2 mb-0">
            ${m.locations && m.locations.length ? 'Note: location is text-extracted (NER) and indicative only — verify before dispatch.' : ''}
        </p>
    `;
    const modal = new bootstrap.Modal(document.getElementById('detailModal'));
    modal.show();
}

async function overridePriority(messageId) {
    const val = parseFloat(document.getElementById('overrideInput').value);
    if (isNaN(val)) return;
    await apiPost(`/api/messages/${messageId}/override`, { new_priority: val });
    bootstrap.Modal.getInstance(document.getElementById('detailModal')).hide();
    loadQueue();
}

async function recordAction(messageId, actionType) {
    await apiPost(`/api/messages/${messageId}/action`, { action_type: actionType });
    bootstrap.Modal.getInstance(document.getElementById('detailModal')).hide();
    loadQueue();
    loadStats();
}

// --- init ---
if (requireAuth()) {
    loadStats();
    loadQueue();
    setInterval(() => { loadStats(); loadQueue(); }, 15000);
}
