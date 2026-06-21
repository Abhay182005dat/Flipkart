/**
 * ECIP Dashboard — Application Logic
 * ====================================
 * Handles: API communication, map rendering, EII gauge animation,
 * panel navigation, prediction flow, scenario planning, and resource optimization.
 */

const API_BASE = "http://localhost:8000/api/v1";
let map = null;
let markersLayer = null;

// ── Initialization ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initMap();
    initPredictForm();
    initOptimizer();
    initScenarioForm();
    setDefaultDatetime();
    checkApiHealth();
});

// ── Navigation ───────────────────────────────────────────────────────
function initNavigation() {
    document.querySelectorAll(".nav-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const panel = btn.dataset.panel;
            document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(`panel-${panel}`).classList.add("active");
            document.getElementById("page-title").textContent =
                btn.textContent.trim();
        });
    });
}

// ── Map ──────────────────────────────────────────────────────────────
function initMap() {
    map = L.map("map", {
        zoomControl: false,
    }).setView([12.9716, 77.5946], 12);

    L.control.zoom({ position: "bottomright" }).addTo(map);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        maxZoom: 19,
    }).addTo(map);

    markersLayer = L.layerGroup().addTo(map);
}

function plotEventsOnMap(events) {
    markersLayer.clearLayers();
    events.forEach(ev => {
        if (!ev.latitude || !ev.longitude) return;
        const color = ev.requires_road_closure ? "#ef4444" : "#3b82f6";
        const marker = L.circleMarker([ev.latitude, ev.longitude], {
            radius: 6,
            fillColor: color,
            fillOpacity: 0.7,
            color: color,
            weight: 1,
            opacity: 0.9,
        });
        marker.bindPopup(`
            <div style="font-family:Inter,sans-serif; font-size:12px; min-width:180px;">
                <strong style="text-transform:capitalize;">${ev.event_cause}</strong><br>
                <span style="color:#94a3b8;">${ev.corridor}</span><br>
                <span>Duration: ${ev.duration_min ? ev.duration_min + " min" : "N/A"}</span><br>
                <span>Closure: ${ev.requires_road_closure ? "Yes" : "No"}</span><br>
                <span>Priority: ${ev.priority}</span>
            </div>
        `);
        markersLayer.addLayer(marker);
    });
}

// ── API Health ───────────────────────────────────────────────────────
async function checkApiHealth() {
    try {
        const res = await fetch("http://localhost:8000/health");
        if (res.ok) {
            setApiStatus(true);
            loadDashboardData();
        } else {
            setApiStatus(false);
        }
    } catch {
        setApiStatus(false);
        // Retry in 5 seconds
        setTimeout(checkApiHealth, 5000);
    }
}

function setApiStatus(online) {
    const el = document.getElementById("api-status");
    const dot = el.querySelector(".status-dot");
    const text = el.querySelector(".status-text");
    dot.className = `status-dot ${online ? "online" : "offline"}`;
    text.textContent = online ? "API Connected" : "API Offline — retry in 5s";
}

// ── Dashboard Data ───────────────────────────────────────────────────
async function loadDashboardData() {
    try {
        // Stats
        const stats = await apiGet("/stats");
        document.querySelector("#stat-total-events .pill-value").textContent =
            stats.total_events.toLocaleString();
        document.querySelector("#stat-closure-rate .pill-value").textContent =
            (stats.closure_rate * 100).toFixed(1) + "%";
        document.querySelector("#stat-avg-duration .pill-value").textContent =
            stats.avg_duration_min + " min";

        // Cause chart
        renderBarChart("cause-chart", stats.event_cause_distribution);

        // Corridor chart
        renderBarChart("corridor-chart", stats.corridor_distribution);

        // Populate filter
        const filterEl = document.getElementById("filter-cause");
        Object.keys(stats.event_cause_distribution).forEach(cause => {
            const opt = document.createElement("option");
            opt.value = cause;
            opt.textContent = cause.replace(/_/g, " ");
            filterEl.appendChild(opt);
        });
        filterEl.addEventListener("change", () => loadEvents(filterEl.value));

        // Load events
        loadEvents();
    } catch (e) {
        console.error("Failed to load dashboard data:", e);
    }
}

async function loadEvents(cause = "") {
    const listEl = document.getElementById("events-list");
    listEl.innerHTML = '<div class="loading-spinner">Loading…</div>';

    try {
        let url = "/events/list?limit=30";
        if (cause) url += `&event_cause=${cause}`;
        const data = await apiGet(url);

        listEl.innerHTML = "";
        plotEventsOnMap(data.events);

        data.events.forEach(ev => {
            const item = document.createElement("div");
            item.className = "event-item";
            const badgeColor = ev.requires_road_closure ? "var(--eii-critical)" : "var(--eii-low)";
            item.innerHTML = `
                <div class="event-badge" style="background:${badgeColor}"></div>
                <div class="event-info">
                    <div class="event-cause">${ev.event_cause.replace(/_/g, " ")}</div>
                    <div class="event-meta">${ev.corridor} · ${ev.priority} · ${formatDate(ev.start_datetime)}</div>
                </div>
                <div class="event-duration">${ev.duration_min ? ev.duration_min + "m" : "—"}</div>
            `;
            listEl.appendChild(item);
        });
    } catch (e) {
        listEl.innerHTML = '<div class="loading-spinner">Failed to load events</div>';
    }
}

// ── Bar Chart Renderer ───────────────────────────────────────────────
function renderBarChart(containerId, data) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    const entries = Object.entries(data).slice(0, 8);
    const max = Math.max(...entries.map(e => e[1]));

    entries.forEach(([label, value]) => {
        const pct = (value / max * 100).toFixed(1);
        const row = document.createElement("div");
        row.className = "chart-bar-row";
        row.innerHTML = `
            <div class="chart-bar-label" title="${label}">${label.replace(/_/g, " ")}</div>
            <div class="chart-bar-track">
                <div class="chart-bar-fill" style="width:${pct}%"></div>
            </div>
            <div class="chart-bar-value">${value}</div>
        `;
        container.appendChild(row);
    });
}

// ── Prediction Form ──────────────────────────────────────────────────
function initPredictForm() {
    document.getElementById("predict-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("btn-predict");
        btn.disabled = true;
        btn.textContent = "⏳ Predicting…";

        const payload = {
            event_type: document.getElementById("inp-event-type").value,
            event_cause: document.getElementById("inp-event-cause").value,
            corridor: document.getElementById("inp-corridor").value,
            priority: document.getElementById("inp-priority").value,
            veh_type: document.getElementById("inp-veh-type").value,
            start_datetime: new Date(document.getElementById("inp-datetime").value).toISOString(),
            latitude: parseFloat(document.getElementById("inp-lat").value),
            longitude: parseFloat(document.getElementById("inp-lon").value),
            requires_road_closure: document.getElementById("inp-closure").checked,
        };

        try {
            const result = await apiPost("/events/predict", payload);
            renderPredictionResults(result, payload);
        } catch (e) {
            alert("Prediction failed: " + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = "⚡ Predict Impact";
        }
    });
}

function renderPredictionResults(result, payload) {
    const container = document.getElementById("predict-results");
    container.style.display = "block";

    // EII Gauge
    const eii = result.eii;
    const gaugeArc = document.getElementById("gauge-arc");
    const maxArc = 251; // approx circumference of the arc path
    const arcLength = (eii.eii_score / 100) * maxArc;
    gaugeArc.setAttribute("stroke-dasharray", `${arcLength} ${maxArc}`);

    document.getElementById("gauge-value").textContent = eii.eii_score;
    const labelEl = document.getElementById("gauge-label");
    labelEl.textContent = eii.eii_level;
    labelEl.style.fill = `var(--eii-${eii.eii_level.toLowerCase()})`;

    // EII Components
    const compEl = document.getElementById("eii-components");
    compEl.innerHTML = "";
    const colorMap = {
        duration_risk: "#3b82f6",
        closure_risk: "#ef4444",
        priority_risk: "#eab308",
        location_risk: "#8b5cf6",
    };
    Object.entries(eii.components).forEach(([key, val]) => {
        const pct = (val * 100).toFixed(0);
        compEl.innerHTML += `
            <div class="eii-comp-row">
                <div class="eii-comp-label">${key.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}</div>
                <div class="eii-comp-bar">
                    <div class="eii-comp-fill" style="width:${pct}%; background:${colorMap[key] || '#3b82f6'}"></div>
                </div>
                <div class="eii-comp-value">${pct}%</div>
            </div>
        `;
    });

    // Priority
    const prio = result.response_priority;
    const prioEl = document.getElementById("priority-display");
    prioEl.innerHTML = `
        <div class="priority-badge p${prio.priority}">
            Priority P${prio.priority} — ${prio.label}
        </div>
        <div class="priority-reason">${prio.reason}</div>
    `;

    // Predictions
    const predsEl = document.getElementById("prediction-metrics");
    predsEl.innerHTML = `
        <div class="metric-card">
            <div class="metric-value">${result.predictions.duration_min}</div>
            <div class="metric-label">Predicted Duration (min)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${(result.predictions.closure_probability * 100).toFixed(1)}%</div>
            <div class="metric-label">Closure Probability</div>
        </div>
    `;

    // Similar Events
    renderSimilarEvents(result.similar_events);

    // Scenarios
    renderScenarios(result.scenarios);

    // SHAP button
    document.getElementById("btn-explain").onclick = async () => {
        document.getElementById("btn-explain").textContent = "Loading…";
        try {
            const shap = await apiPost("/events/explain", payload);
            renderShap(shap);
        } catch (e) {
            document.getElementById("shap-explanation").innerHTML =
                `<p style="color:var(--eii-critical)">SHAP failed: ${e.message}</p>`;
        }
    };

    // Scroll to results
    container.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSimilarEvents(data) {
    // Aggregate stats
    const statsEl = document.getElementById("similar-stats");
    const agg = data.aggregate_stats;
    statsEl.innerHTML = `
        <div class="similar-stat"><div class="stat-val">${agg.avg_duration_min || "—"}</div><div class="stat-lbl">Avg Duration (min)</div></div>
        <div class="similar-stat"><div class="stat-val">${agg.closure_rate !== undefined ? (agg.closure_rate * 100).toFixed(0) + "%" : "—"}</div><div class="stat-lbl">Closure Rate</div></div>
        <div class="similar-stat"><div class="stat-val">${agg.sample_size}</div><div class="stat-lbl">Sample Size</div></div>
        <div class="similar-stat"><div class="stat-val">${agg.avg_similarity}</div><div class="stat-lbl">Avg Similarity</div></div>
    `;

    // Table
    const tbody = document.getElementById("similar-tbody");
    tbody.innerHTML = "";
    (data.similar_events || []).forEach(ev => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td style="text-transform:capitalize">${(ev.event_cause || "").replace(/_/g, " ")}</td>
            <td>${ev.corridor || "—"}</td>
            <td>${ev.duration_min ? ev.duration_min.toFixed(0) + " min" : "—"}</td>
            <td><span class="closure-badge ${ev.requires_road_closure ? 'yes' : 'no'}">${ev.requires_road_closure ? 'Yes' : 'No'}</span></td>
            <td style="font-family:'JetBrains Mono',monospace">${ev.similarity_score}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderScenarios(scenarios) {
    const grid = document.getElementById("scenarios-grid");
    grid.innerHTML = "";
    (scenarios || []).forEach(s => {
        const card = document.createElement("div");
        card.className = "scenario-card";
        const eiiDelta = s.delta.eii_change;
        const durDelta = s.delta.duration_change_pct;
        const eiiClass = eiiDelta <= 0 ? "positive" : "negative";
        const durClass = durDelta <= 0 ? "positive" : "negative";
        card.innerHTML = `
            <div class="scenario-label">${s.label || "Custom Scenario"}</div>
            <div class="scenario-metrics">
                <div class="scenario-metric">
                    <span class="sm-label">EII Change</span>
                    <span class="sm-value ${eiiClass}">${eiiDelta > 0 ? "+" : ""}${eiiDelta}</span>
                </div>
                <div class="scenario-metric">
                    <span class="sm-label">New EII</span>
                    <span class="sm-value">${s.projected.eii_score} (${s.projected.eii_level})</span>
                </div>
                <div class="scenario-metric">
                    <span class="sm-label">Duration Δ</span>
                    <span class="sm-value ${durClass}">${durDelta > 0 ? "+" : ""}${durDelta}%</span>
                </div>
                <div class="scenario-metric">
                    <span class="sm-label">Closure Prob</span>
                    <span class="sm-value">${(s.projected.closure_prob * 100).toFixed(0)}%</span>
                </div>
                <div class="scenario-metric">
                    <span class="sm-label">Level Change</span>
                    <span class="sm-value">${s.delta.eii_level_change}</span>
                </div>
            </div>
        `;
        grid.appendChild(card);
    });
}

function renderShap(data) {
    const container = document.getElementById("shap-explanation");
    container.innerHTML = "";

    ["duration", "closure"].forEach(modelName => {
        const model = data[modelName];
        if (!model) return;

        const div = document.createElement("div");
        div.className = "shap-model";
        div.innerHTML = `<div class="shap-model-title">${modelName.charAt(0).toUpperCase() + modelName.slice(1)} Model (prediction: ${model.prediction})</div>`;

        const maxImpact = Math.max(...model.top_features.map(f => Math.abs(f.impact)), 0.01);

        model.top_features.forEach(f => {
            const pct = (Math.abs(f.impact) / maxImpact * 45).toFixed(1);
            const isPos = f.impact > 0;
            const row = document.createElement("div");
            row.className = "shap-bar-row";
            row.innerHTML = `
                <div class="shap-feature">${f.feature}</div>
                <div class="shap-bar-container">
                    <div class="shap-bar-center"></div>
                    <div class="shap-bar ${isPos ? "positive" : "negative"}" style="width:${pct}%"></div>
                </div>
                <div class="shap-impact" style="color:${isPos ? "var(--eii-critical)" : "var(--accent-blue)"}">${f.impact > 0 ? "+" : ""}${f.impact.toFixed(3)}</div>
            `;
            div.appendChild(row);
        });

        container.appendChild(div);
    });
}

// ── Optimizer ────────────────────────────────────────────────────────
let optEventCounter = 0;

function initOptimizer() {
    document.getElementById("btn-add-opt-event").addEventListener("click", addOptEvent);
    document.getElementById("btn-run-optimize").addEventListener("click", runOptimize);
    // Add 3 sample events
    addOptEvent(); addOptEvent(); addOptEvent();
}

function addOptEvent() {
    optEventCounter++;
    const list = document.getElementById("opt-events-list");
    const causes = ["accident", "vehicle_breakdown", "construction", "public_event"];
    const levels = ["Low", "Medium", "High", "Critical"];
    const row = document.createElement("div");
    row.className = "opt-event-row";
    row.id = `opt-event-${optEventCounter}`;
    row.innerHTML = `
        <span style="width:20px; font-weight:600; color:var(--text-muted);">#${optEventCounter}</span>
        <input type="text" value="event_${optEventCounter}" placeholder="Event ID" style="width:100px" data-field="event_id">
        <select data-field="eii_level">
            ${levels.map(l => `<option value="${l}" ${l === levels[Math.min(optEventCounter-1, 3)] ? "selected" : ""}>${l}</option>`).join("")}
        </select>
        <input type="number" value="${30 + optEventCounter * 15}" min="0" max="100" step="5" data-field="eii_score" style="width:70px" placeholder="EII">
        <input type="number" value="${(0.3 + optEventCounter * 0.15).toFixed(2)}" min="0" max="1" step="0.1" data-field="closure_prob" style="width:70px" placeholder="Closure">
        <input type="number" value="${1 + optEventCounter * 0.5}" min="0.5" step="0.5" data-field="duration_hours" style="width:70px" placeholder="Hours">
        <select data-field="response_priority">
            <option value="1" ${optEventCounter === 3 ? "selected" : ""}>P1</option>
            <option value="2" ${optEventCounter === 2 ? "selected" : ""}>P2</option>
            <option value="3" ${optEventCounter === 1 ? "selected" : ""}>P3</option>
            <option value="4">P4</option>
        </select>
        <button class="btn-secondary" onclick="this.parentElement.remove()" style="padding:4px 8px; font-size:11px;">&times;</button>
    `;
    list.appendChild(row);
}

async function runOptimize() {
    const events = [];
    document.querySelectorAll(".opt-event-row").forEach(row => {
        const ev = {};
        row.querySelectorAll("[data-field]").forEach(input => {
            const field = input.dataset.field;
            const val = input.value;
            if (["eii_score", "closure_prob", "duration_hours"].includes(field)) {
                ev[field] = parseFloat(val);
            } else if (field === "response_priority") {
                ev[field] = parseInt(val);
            } else {
                ev[field] = val;
            }
        });
        events.push(ev);
    });

    const payload = {
        events,
        total_personnel: parseInt(document.getElementById("opt-total-personnel").value),
        total_barricades: parseInt(document.getElementById("opt-total-barricades").value),
    };

    try {
        const result = await apiPost("/optimize/resources", payload);
        renderOptResults(result);
    } catch (e) {
        alert("Optimization failed: " + e.message);
    }
}

function renderOptResults(result) {
    const container = document.getElementById("opt-results");
    container.style.display = "block";
    container.innerHTML = "";

    // Summary
    const s = result.summary;
    container.innerHTML += `
        <div class="opt-summary">
            <div class="similar-stat"><div class="stat-val">${s.total_events}</div><div class="stat-lbl">Events</div></div>
            <div class="similar-stat"><div class="stat-val">${s.total_personnel_used}/${s.total_personnel_used + s.personnel_remaining}</div><div class="stat-lbl">Personnel Used</div></div>
            <div class="similar-stat"><div class="stat-val">${s.total_barricades_used}/${s.total_barricades_used + s.barricades_remaining}</div><div class="stat-lbl">Barricades Used</div></div>
            <div class="similar-stat"><div class="stat-val">${s.personnel_utilization_pct}%</div><div class="stat-lbl">Utilization</div></div>
            <div class="similar-stat"><div class="stat-val">${s.solver}</div><div class="stat-lbl">Solver</div></div>
        </div>
    `;

    // Allocations
    result.allocations.forEach(a => {
        const borderColor = `var(--p${a.response_priority})`;
        container.innerHTML += `
            <div class="opt-alloc-card" style="border-left-color:${borderColor}">
                <div class="opt-alloc-header">
                    <div class="opt-alloc-title">
                        P${a.response_priority} — ${a.event_id} (EII: ${a.eii_score} ${a.eii_level})
                    </div>
                </div>
                <div class="opt-alloc-detail">
                    ${a.personnel} officers · ${a.barricades} barricades
                    ${a.escalated ? " · Escalated" : ""}
                </div>
                <div class="opt-alloc-detail" style="margin-top:4px; font-style:italic;">
                    ${a.explanation}
                </div>
            </div>
        `;
    });
}

// ── Scenario Form ────────────────────────────────────────────────────
function initScenarioForm() {
    document.getElementById("scenario-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            duration_min: parseFloat(document.getElementById("sc-duration").value),
            closure_prob: parseFloat(document.getElementById("sc-closure").value),
            priority_is_high: document.getElementById("sc-priority").value === "true",
            location_risk: parseFloat(document.getElementById("sc-location-risk").value),
            current_personnel: 0,
            current_barricades: 0,
            delta_personnel: parseInt(document.getElementById("sc-delta-p").value),
            delta_barricades: parseInt(document.getElementById("sc-delta-b").value),
            close_road: document.getElementById("sc-close-road").checked,
        };

        try {
            const result = await apiPost("/events/scenario", payload);
            renderStandaloneScenario(result);
        } catch (e) {
            alert("Scenario failed: " + e.message);
        }
    });
}

function renderStandaloneScenario(result) {
    const container = document.getElementById("scenario-result");
    container.style.display = "block";

    const d = result.delta;
    const eiiClass = d.eii_change <= 0 ? "positive" : "negative";

    container.innerHTML = `
        <div class="card" style="margin-top:16px;">
            <div class="card-header"><h3>📊 Scenario Result</h3></div>
            <div class="prediction-metrics" style="grid-template-columns: repeat(4, 1fr);">
                <div class="metric-card">
                    <div class="metric-value" style="${d.eii_change <= 0 ? "" : "-webkit-text-fill-color:var(--eii-critical)"}">${d.eii_change > 0 ? "+" : ""}${d.eii_change}</div>
                    <div class="metric-label">EII Change</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${result.projected.eii_score}</div>
                    <div class="metric-label">New EII (${result.projected.eii_level})</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${d.duration_change_pct}%</div>
                    <div class="metric-label">Duration Change</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(result.projected.closure_prob * 100).toFixed(0)}%</div>
                    <div class="metric-label">New Closure Prob</div>
                </div>
            </div>
            <p style="margin-top:12px; font-size:13px; color:var(--text-secondary);">
                Level: ${d.eii_level_change} · Personnel: ${result.total_personnel} · Barricades: ${result.total_barricades}
            </p>
        </div>
    `;
}

// ── Utility ──────────────────────────────────────────────────────────
function setDefaultDatetime() {
    const now = new Date();
    const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000);
    document.getElementById("inp-datetime").value = local.toISOString().slice(0, 16);
}

function formatDate(isoStr) {
    if (!isoStr || isoStr === "NaT") return "—";
    try {
        const d = new Date(isoStr);
        return d.toLocaleDateString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
    } catch { return "—"; }
}

async function apiGet(path) {
    const res = await fetch(API_BASE + path);
    if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
    return res.json();
}

async function apiPost(path, body) {
    const res = await fetch(API_BASE + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
    return res.json();
}
