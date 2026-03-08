/* ═══════════════════════════════════════════════════════════════
   ResistAI v2.0 — Main Application JavaScript
   ═══════════════════════════════════════════════════════════════ */

const API = '';
let importanceChart = null;
let currentResults = null;
let networkSimulation = null;

// ── Tab Switching ──────────────────────────────────────────────
function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach((b, i) => {
        b.classList.toggle('active', ['single', 'batch', 'history'][i] === tab);
    });
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-' + tab).classList.add('active');
    if (tab === 'history') renderHistory();
}

// ── File Upload ───────────────────────────────────────────────
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => { document.getElementById('fasta-input').value = ev.target.result; updateSeqInfo(); };
    reader.readAsText(file);
}

function updateSeqInfo() {
    const fasta = document.getElementById('fasta-input').value;
    const seqOnly = fasta.replace(/^>.*$/gm, '').replace(/\s/g, '');
    if (seqOnly.length > 50) {
        const gc = ((seqOnly.match(/[GC]/gi) || []).length / seqOnly.length * 100).toFixed(1);
        document.getElementById('seq-len-display').textContent = seqOnly.length.toLocaleString() + ' bp';
        document.getElementById('seq-gc-display').textContent = gc + '% GC';
        document.getElementById('seq-info').style.display = 'inline';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const ta = document.getElementById('fasta-input');
    if (ta) ta.addEventListener('input', updateSeqInfo);
    initDNACanvas();
});

// ── Load Sample ───────────────────────────────────────────────
async function loadSample() {
    try {
        const res = await fetch(API + '/api/sample_fasta');
        const data = await res.json();
        document.getElementById('fasta-input').value = data.fasta;
        updateSeqInfo();
    } catch (e) {
        const bases = 'ATGC'; let seq = '';
        for (let i = 0; i < 5000; i++) seq += bases[Math.floor(Math.random() * 4)];
        let fasta = '>Escherichia_coli_sample contig_1\n';
        for (let i = 0; i < seq.length; i += 70) fasta += seq.slice(i, i + 70) + '\n';
        document.getElementById('fasta-input').value = fasta;
        updateSeqInfo();
    }
}

function clearAll() {
    document.getElementById('fasta-input').value = '';
    document.getElementById('results').classList.remove('active');
    document.getElementById('seq-info').style.display = 'none';
}

// ── Main Prediction ───────────────────────────────────────────
async function runPrediction() {
    const fasta = document.getElementById('fasta-input').value.trim();
    if (!fasta) { alert('Please enter a FASTA sequence'); return; }

    const btn = document.getElementById('btn-predict');
    btn.disabled = true;
    document.getElementById('loader').classList.add('active');
    document.getElementById('results').classList.remove('active');

    const msgs = ['Parsing FASTA assembly...', 'Scanning for AMR genes...', 'Running XGBoost models...', 'Computing SHAP values...', 'Building gene network...', 'Generating clinical report...'];
    let mi = 0;
    const msgInterval = setInterval(() => {
        document.getElementById('loading-text').textContent = msgs[mi % msgs.length];
        mi++;
    }, 700);

    try {
        const res = await fetch(API + '/api/predict', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fasta })
        });
        const data = await res.json();
        if (data.error) { alert('Error: ' + data.error); return; }
        currentResults = data;
        renderResults(data);
        saveToHistory(data);
        triggerParticles();
    } catch (e) {
        alert('Connection error. Is the server running?');
    } finally {
        clearInterval(msgInterval);
        btn.disabled = false;
        document.getElementById('loader').classList.remove('active');
    }
}

// ── Render All Results ────────────────────────────────────────
function renderResults(data) {
    document.getElementById('results').classList.add('active');
    renderSeqSummary(data.seq_stats);
    renderPredictions(data.predictions);
    renderWHOPanel(data);
    renderClinical(data.clinical);
    renderMetrics(data.model_metrics);
    renderGenes(data.detected_genes);
    renderNetwork(data.network);
    renderHeatmap(data.heatmap);
    renderTimeline(data.timeline);
    renderImportanceChart(data.model_metrics);
    setTimeout(() => {
        window.scrollTo({ top: document.getElementById('results').offsetTop - 20, behavior: 'smooth' });
    }, 100);
}

// ── Sequence Summary ──────────────────────────────────────────
function renderSeqSummary(stats) {
    if (!stats) return;
    const g = document.getElementById('seq-summary-grid');
    g.innerHTML = `
        <div class="seq-stat-item"><div class="val">${stats.total_length.toLocaleString()}</div><div class="lbl">Total bp</div></div>
        <div class="seq-stat-item"><div class="val">${stats.n_contigs}</div><div class="lbl">Contigs</div></div>
        <div class="seq-stat-item"><div class="val">${stats.gc_content}%</div><div class="lbl">GC Content</div></div>
        <div class="seq-stat-item"><div class="val">${stats.n50.toLocaleString()}</div><div class="lbl">N50</div></div>
        <div class="seq-stat-item"><div class="val" style="color:var(--accent-purple)">E. coli</div><div class="lbl">Organism</div></div>`;
}

// ── Predictions ───────────────────────────────────────────────
function renderPredictions(predictions) {
    const grid = document.getElementById('pred-grid');
    const drugClasses = { ampicillin: 'Beta-lactam (Penicillin)', ciprofloxacin: 'Fluoroquinolone', gentamicin: 'Aminoglycoside' };
    grid.innerHTML = '';

    for (const [ab, pred] of Object.entries(predictions)) {
        const cls = pred.phenotype.toLowerCase();
        const card = document.createElement('div');
        card.className = `pred-card ${cls}`;

        let probBars = '';
        for (const [label, prob] of Object.entries(pred.probabilities)) {
            const color = label === 'Resistant' ? 'var(--accent-red)' : label === 'Susceptible' ? 'var(--accent-green)' : 'var(--accent-amber)';
            probBars += `<div class="prob-bar"><span class="label">${label}</span><div class="track"><div class="fill" style="width:${prob * 100}%;background:${color}"></div></div><span class="value">${(prob * 100).toFixed(1)}%</span></div>`;
        }

        card.innerHTML = `
            <div class="ab-name">${ab}</div>
            <div class="drug-class">${drugClasses[ab] || ''}</div>
            <div class="phenotype">${pred.phenotype}</div>
            <div class="confidence">${(pred.confidence * 100).toFixed(1)}%</div>
            <div class="conf-label">Confidence</div>
            <div class="prob-bars">${probBars}</div>`;
        grid.appendChild(card);
    }
}

// ── WHO AWaRe Panel ───────────────────────────────────────────
function renderWHOPanel(data) {
    const panel = document.getElementById('who-panel');
    const catColors = { 'Access': '#22c55e', 'Watch': '#f59e0b', 'Reserve': '#ef4444' };
    const catDescs = { 'Access': 'First-line — widely available, low resistance potential', 'Watch': 'Higher resistance potential — targeted stewardship', 'Reserve': 'Last resort — reserved for MDR infections' };

    let whoHtml = '<div class="who-grid">';
    for (const ab of ['ampicillin', 'ciprofloxacin', 'gentamicin']) {
        const pred = data.predictions[ab] || {};
        const aware = data.who_aware?.[ab] || {};
        const cat = aware.category || 'Unknown';
        const color = catColors[cat] || '#6b7280';
        whoHtml += `<div class="who-card" style="border-color:${color}30">
            <div class="who-icon">${aware.icon || '⚪'}</div>
            <div class="who-cat" style="color:${color}">${cat.toUpperCase()}</div>
            <div class="who-drug">${ab}</div>
            <div class="who-desc">${catDescs[cat] || ''}</div>
        </div>`;
    }
    whoHtml += '</div>';

    // Stewardship score
    const score = data.clinical?.stewardship_score || 50;
    const scoreColor = score >= 70 ? '#22c55e' : score >= 40 ? '#f59e0b' : '#ef4444';
    whoHtml += `<div class="stewardship-section">
        <div class="stewardship-gauge">
            <canvas id="stewardship-canvas" width="80" height="80"></canvas>
            <div class="stewardship-score-text" style="color:${scoreColor}">${score}</div>
        </div>
        <div class="stewardship-info">
            <h3>🏥 Antibiotic Stewardship Score</h3>
            <p>${score >= 70 ? 'Good stewardship outlook — first-line options available. Follow WHO AWaRe guidelines for optimal prescribing.' : score >= 40 ? 'Moderate concern — limited options. Consider antimicrobial stewardship team consultation.' : '⚠️ Critical — multi-drug resistance detected. Consult infectious disease specialist immediately. Reserve-category drugs may be needed.'}</p>
        </div>
    </div>`;

    panel.innerHTML = whoHtml;

    // Draw stewardship gauge
    setTimeout(() => drawStewardshipGauge(score, scoreColor), 50);
}

function drawStewardshipGauge(score, color) {
    const canvas = document.getElementById('stewardship-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cx = 40, cy = 40, r = 32;
    ctx.clearRect(0, 0, 80, 80);
    ctx.beginPath(); ctx.arc(cx, cy, r, 0.75 * Math.PI, 2.25 * Math.PI);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 6; ctx.lineCap = 'round'; ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, r, 0.75 * Math.PI, (0.75 + 1.5 * score / 100) * Math.PI);
    ctx.strokeStyle = color; ctx.lineWidth = 6; ctx.lineCap = 'round'; ctx.stroke();
}

// ── Clinical Recommendations ──────────────────────────────────
function renderClinical(clinical) {
    const panel = document.getElementById('clinical-panel');
    let html = '';
    for (const rec of clinical.recommendations) {
        const whoTag = rec.who_category ? `<span style="font-size:10px;color:${rec.who_color};margin-left:8px">${rec.who_icon} ${rec.who_category}</span>` : '';
        html += `<div class="clinical-card" style="border-left-color:${rec.status_color}">
            <div class="status" style="color:${rec.status_color}">${rec.status} — ${rec.antibiotic}${whoTag}</div>
            <div class="message">${rec.message}</div></div>`;
    }
    const boxCls = clinical.mdr_risk ? 'overall-box mdr-warning' : 'overall-box';
    html += `<div class="${boxCls}">${clinical.mdr_risk ? '⚠️ ' : '✅ '}${clinical.overall}</div>`;
    panel.innerHTML = html;
}

// ── Model Metrics ─────────────────────────────────────────────
function renderMetrics(metrics) {
    const panel = document.getElementById('metrics-panel');
    let html = '';
    for (const [ab, m] of Object.entries(metrics)) {
        html += `<div style="margin-bottom:16px"><div style="font-weight:700;text-transform:capitalize;margin-bottom:8px;font-size:14px">${ab}</div>
        <div class="metric-row">
            <div class="metric-box green"><div class="val">${(m.auc * 100).toFixed(1)}%</div><div class="lbl">AUC</div></div>
            <div class="metric-box purple"><div class="val">${(m.mcc * 100).toFixed(0)}</div><div class="lbl">MCC×100</div></div>
            <div class="metric-box cyan"><div class="val">${(m.accuracy * 100).toFixed(1)}%</div><div class="lbl">Accuracy</div></div>
        </div></div>`;
    }
    panel.innerHTML = html;
}

// ── Detected Genes (with mechanism btn) ───────────────────────
function renderGenes(genes) {
    const tbody = document.getElementById('gene-tbody');
    tbody.innerHTML = '';
    const catMap = { primary_resistance: 'primary', secondary_mechanism: 'secondary', snp_marker: 'snp', primary: 'primary', secondary: 'secondary', snps: 'snp' };
    const catLabel = { primary: 'Primary Gene', secondary: 'Secondary', snp: 'SNP Marker' };

    for (const g of genes.slice(0, 25)) {
        const cat = catMap[g.category] || 'snp';
        const mechType = g.mechanism_info?.mechanism || '';
        const tr = document.createElement('tr');
        tr.innerHTML = `<td style="font-family:'JetBrains Mono',monospace;font-weight:600;font-size:12px">${g.gene}</td>
            <td><span class="gene-tag ${cat}">${catLabel[cat] || g.category}</span></td>
            <td style="font-size:11px;color:var(--text-secondary)">${mechType}</td>
            <td><button class="gene-mech-btn" onclick="showMechanism('${g.gene}')">🔍 Details</button></td>`;
        tbody.appendChild(tr);
    }
}

// ── Mechanism Modal ───────────────────────────────────────────
function showMechanism(geneName) {
    const gene = currentResults?.detected_genes?.find(g => g.gene === geneName);
    const info = gene?.mechanism_info || {};
    const mechColors = { 'Enzymatic hydrolysis': '#8b5cf6', 'Target site modification': '#f97316', 'Active efflux': '#06b6d4', 'Target protection': '#22c55e', 'Enzymatic modification': '#ec4899', '16S rRNA methylation': '#ef4444', 'Reduced permeability': '#f59e0b' };
    const color = mechColors[info.mechanism] || '#8b5cf6';

    document.getElementById('mechanism-content').innerHTML = `
        <div class="mech-title">${info.full_name || geneName}</div>
        <div class="mech-subtitle">${geneName}</div>
        <div style="margin-bottom:16px"><span class="mech-tag" style="background:${color}20;color:${color};border:1px solid ${color}40">${info.mechanism || 'Unknown'}</span></div>
        <div class="mech-section"><div class="mech-section-title">📖 Description</div><div class="mech-section-body">${info.description || 'No detailed information available.'}</div></div>
        <div class="mech-section"><div class="mech-section-title">📍 Genetic Location</div><div class="mech-section-body">${info.location || 'Unknown'}</div></div>
        <div class="mech-section"><div class="mech-section-title">📊 Prevalence</div><div class="mech-section-body">${info.prevalence || 'Unknown'}</div></div>
        <div class="mech-section"><div class="mech-section-title">🎯 Molecular Target</div><div class="mech-section-body">${info.molecular_target || 'Unknown'}</div></div>`;
    document.getElementById('mechanism-modal').classList.add('active');
}

function closeMechanismModal() {
    document.getElementById('mechanism-modal').classList.remove('active');
}

// ── Gene Network ──────────────────────────────────────────────
function renderNetwork(network) {
    const svg = d3.select('#network-svg');
    svg.selectAll('*').remove();
    const width = svg.node().getBoundingClientRect().width;
    const height = 480;
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    if (!network || !network.nodes || network.nodes.length === 0) return;

    networkSimulation = d3.forceSimulation(network.nodes)
        .force('link', d3.forceLink(network.links).id(d => d.id).distance(110).strength(d => d.strength || 0.3))
        .force('charge', d3.forceManyBody().strength(-250))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => d.size + 8));

    const defs = svg.append('defs');
    const glow = defs.append('filter').attr('id', 'glow');
    glow.append('feGaussianBlur').attr('stdDeviation', 4).attr('result', 'blur');
    glow.append('feMerge').selectAll('feMergeNode').data(['blur', 'SourceGraphic']).enter().append('feMergeNode').attr('in', d => d);

    // Gradient links
    const link = svg.append('g').selectAll('line').data(network.links).enter().append('line')
        .style('stroke', 'rgba(255,255,255,0.08)').style('stroke-width', d => Math.max(1.5, d.strength * 6));

    const node = svg.append('g').selectAll('g').data(network.nodes).enter().append('g')
        .call(d3.drag()
            .on('start', (e, d) => { if (!e.active) networkSimulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
            .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y })
            .on('end', (e, d) => { if (!e.active) networkSimulation.alphaTarget(0); d.fx = null; d.fy = null }));

    // Outer glow ring for antibiotics
    node.filter(d => d.type === 'antibiotic').append('circle').attr('r', d => d.size + 8)
        .attr('fill', 'none').attr('stroke', d => d.color).attr('stroke-width', 1).attr('opacity', 0.3)
        .style('filter', 'url(#glow)');

    node.append('circle').attr('r', d => d.size || 10).attr('fill', d => d.color || '#8b5cf6')
        .attr('stroke', 'rgba(255,255,255,0.15)').attr('stroke-width', 1.5).style('filter', 'url(#glow)')
        .style('cursor', 'pointer');

    node.append('text').text(d => d.name).attr('dx', d => (d.size || 10) + 8).attr('dy', 4)
        .style('fill', 'rgba(255,255,255,0.6)').style('font-size', d => d.type === 'antibiotic' ? '13px' : '10px')
        .style('font-family', 'Inter,sans-serif').style('font-weight', d => d.type === 'antibiotic' ? '700' : '500');

    node.append('title').text(d => d.type === 'antibiotic' ? `${d.name}: ${d.phenotype}` : `Gene: ${d.name} (${d.category || ''})`);

    networkSimulation.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${Math.max(20, Math.min(width - 20, d.x))},${Math.max(20, Math.min(height - 20, d.y))})`);
    });
}

function resetNetwork() {
    if (currentResults?.network) renderNetwork(currentResults.network);
}

// ── Heatmap (D3.js SVG) ──────────────────────────────────────
function renderHeatmap(heatmapData) {
    const container = document.getElementById('heatmap-container');
    const tooltip = document.getElementById('heatmap-tooltip');
    if (!heatmapData || heatmapData.length === 0) { container.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:30px">No gene data to display</p>'; return; }

    const svg = d3.select('#heatmap-svg');
    svg.selectAll('*').remove();

    const antibiotics = ['ampicillin', 'ciprofloxacin', 'gentamicin'];
    const abLabels = ['Ampicillin', 'Ciprofloxacin', 'Gentamicin'];
    const genes = heatmapData.map(d => d.gene);

    // Dimensions
    const margin = { top: 55, right: 30, bottom: 20, left: 170 };
    const cellH = 32, cellPad = 3, cellRad = 6;
    const totalW = container.clientWidth || 700;
    const cellW = Math.min(160, (totalW - margin.left - margin.right) / antibiotics.length);
    const width = margin.left + cellW * antibiotics.length + margin.right;
    const height = margin.top + cellH * genes.length + margin.bottom;

    svg.attr('width', width).attr('height', height).attr('viewBox', `0 0 ${width} ${height}`);

    const g = svg.append('g');

    // Color scale: green → yellow → red
    const colorScale = d3.scaleLinear()
        .domain([0, 0.3, 0.6, 1])
        .range(['#1a3a2a', '#2d6a4f', '#e9c46a', '#e63946'])
        .clamp(true);

    // Antibiotic column headers
    antibiotics.forEach((ab, i) => {
        const x = margin.left + i * cellW + cellW / 2;
        g.append('text').text(abLabels[i])
            .attr('x', x).attr('y', margin.top - 18)
            .attr('text-anchor', 'middle')
            .attr('fill', '#e2e8f0').attr('font-size', '13px').attr('font-weight', '700')
            .attr('font-family', 'Inter, sans-serif');
        // Drug class subtitle
        const classes = ['β-lactam', 'Fluoroquinolone', 'Aminoglycoside'];
        g.append('text').text(classes[i])
            .attr('x', x).attr('y', margin.top - 4)
            .attr('text-anchor', 'middle')
            .attr('fill', '#64748b').attr('font-size', '10px')
            .attr('font-family', 'Inter, sans-serif');
    });

    // Gene row labels
    genes.forEach((gene, gi) => {
        const y = margin.top + gi * cellH + cellH / 2 + 4;
        const displayName = gene.length > 22 ? gene.slice(0, 20) + '…' : gene;
        g.append('text').text(displayName)
            .attr('x', margin.left - 10).attr('y', y)
            .attr('text-anchor', 'end')
            .attr('fill', '#94a3b8').attr('font-size', '11px').attr('font-weight', '500')
            .attr('font-family', "'JetBrains Mono', monospace");
    });

    // Heatmap cells
    genes.forEach((gene, gi) => {
        antibiotics.forEach((ab, ai) => {
            const val = heatmapData[gi][ab] || 0;
            const x = margin.left + ai * cellW + cellPad;
            const y = margin.top + gi * cellH + cellPad;
            const w = cellW - cellPad * 2;
            const h = cellH - cellPad * 2;

            const cell = g.append('rect')
                .attr('x', x).attr('y', y)
                .attr('width', w).attr('height', h)
                .attr('rx', cellRad).attr('ry', cellRad)
                .attr('fill', colorScale(val))
                .attr('stroke', 'rgba(255,255,255,0.04)')
                .attr('stroke-width', 1)
                .style('cursor', 'pointer')
                .style('opacity', 0)
                .transition().duration(400).delay(gi * 30 + ai * 80)
                .style('opacity', 1);

            // Value text inside cell
            g.append('text')
                .text(Math.round(val * 100) + '%')
                .attr('x', x + w / 2).attr('y', y + h / 2 + 4)
                .attr('text-anchor', 'middle')
                .attr('fill', val > 0.5 ? '#fff' : 'rgba(255,255,255,0.6)')
                .attr('font-size', '11px').attr('font-weight', '700')
                .attr('font-family', "'JetBrains Mono', monospace")
                .style('pointer-events', 'none')
                .style('opacity', 0)
                .transition().duration(400).delay(gi * 30 + ai * 80)
                .style('opacity', 1);

            // Invisible overlay rect for hover (non-transitioning)
            g.append('rect')
                .attr('x', x).attr('y', y)
                .attr('width', w).attr('height', h)
                .attr('rx', cellRad).attr('ry', cellRad)
                .attr('fill', 'transparent')
                .style('cursor', 'pointer')
                .on('mouseover', (event) => {
                    const strength = val > 0.7 ? 'Strong' : val > 0.4 ? 'Moderate' : 'Weak';
                    const barColor = val > 0.7 ? '#e63946' : val > 0.4 ? '#e9c46a' : '#2d6a4f';
                    tooltip.innerHTML = `<div class="tt-gene">${gene}</div>
                        <div>→ <span class="tt-ab">${abLabels[ai]}</span></div>
                        <div class="tt-val" style="color:${barColor}">${Math.round(val * 100)}% — ${strength}</div>
                        <div class="tt-bar"><div class="tt-bar-fill" style="width:${val * 100}%;background:${barColor}"></div></div>`;
                    tooltip.style.display = 'block';
                })
                .on('mousemove', (event) => {
                    tooltip.style.left = (event.clientX + 16) + 'px';
                    tooltip.style.top = (event.clientY - 10) + 'px';
                })
                .on('mouseout', () => { tooltip.style.display = 'none'; });
        });
    });

    // Color legend
    const legendW = 14, legendH = genes.length * cellH - 10;
    const legendX = margin.left + antibiotics.length * cellW + 12;
    const legendY = margin.top + 5;

    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient').attr('id', 'hm-gradient').attr('x1', '0%').attr('y1', '0%').attr('x2', '0%').attr('y2', '100%');
    gradient.append('stop').attr('offset', '0%').attr('stop-color', '#e63946');
    gradient.append('stop').attr('offset', '40%').attr('stop-color', '#e9c46a');
    gradient.append('stop').attr('offset', '100%').attr('stop-color', '#1a3a2a');

    g.append('rect').attr('x', legendX).attr('y', legendY)
        .attr('width', legendW).attr('height', legendH)
        .attr('rx', 4).attr('fill', 'url(#hm-gradient)');
    g.append('text').text('High').attr('x', legendX + legendW + 6).attr('y', legendY + 10)
        .attr('fill', '#e63946').attr('font-size', '9px').attr('font-weight', '600').attr('font-family', 'Inter');
    g.append('text').text('Low').attr('x', legendX + legendW + 6).attr('y', legendY + legendH - 2)
        .attr('fill', '#2d6a4f').attr('font-size', '9px').attr('font-weight', '600').attr('font-family', 'Inter');
}

// ── Evolution Timeline ────────────────────────────────────────
function renderTimeline(timeline) {
    if (!timeline || timeline.length === 0) return;
    const container = document.getElementById('timeline-container');
    let html = '<div class="timeline">';
    timeline.forEach((step, idx) => {
        const isDanger = step.resistance_level >= 80;
        const barColor = step.resistance_level < 30 ? 'var(--accent-green)' : step.resistance_level < 70 ? 'var(--accent-amber)' : 'var(--accent-red)';
        html += `<div class="timeline-step ${isDanger ? 'danger' : ''}" style="animation-delay:${idx * 0.12}s">
            <div class="timeline-dot"></div>
            <div class="step-title">${step.stage}</div>
            <div class="step-desc">${step.description}</div>
            <div class="step-bar"><div class="step-bar-fill" style="width:${step.resistance_level}%;background:${barColor}"></div></div>
            ${step.gene_acquired ? `<span class="step-gene">${step.gene_acquired}</span>` : ''}
        </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
}

// ── Feature Importance Chart ──────────────────────────────────
function renderImportanceChart(metrics) {
    if (importanceChart) importanceChart.destroy();
    const ctx = document.getElementById('importance-chart').getContext('2d');
    const allFeats = {};
    for (const [ab, m] of Object.entries(metrics)) {
        for (const f of (m.top_features || []).slice(0, 8)) {
            if (!allFeats[f.gene]) allFeats[f.gene] = {};
            allFeats[f.gene][ab] = f.importance;
        }
    }
    const labels = Object.keys(allFeats).slice(0, 12);
    const colors = ['rgba(139,92,246,0.7)', 'rgba(6,182,212,0.7)', 'rgba(34,197,94,0.7)'];
    const datasets = Object.keys(metrics).map((ab, i) => ({
        label: ab.charAt(0).toUpperCase() + ab.slice(1),
        data: labels.map(l => (allFeats[l] || {})[ab] || 0),
        backgroundColor: colors[i], borderColor: colors[i], borderWidth: 1, borderRadius: 3
    }));

    importanceChart = new Chart(ctx, {
        type: 'bar', data: { labels, datasets },
        options: {
            indexAxis: 'y', responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', font: { size: 10 } } },
                y: { grid: { display: false }, ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 9 } } }
            }
        }
    });
}

// ── Batch Analysis ────────────────────────────────────────────
async function loadBatchSample() {
    try {
        const res = await fetch(API + '/api/multi_sample_fasta');
        const data = await res.json();
        document.getElementById('batch-input').value = data.sequences.join('\n');
    } catch (e) { alert('Could not load batch sample'); }
}

async function runBatchAnalysis() {
    const raw = document.getElementById('batch-input').value.trim();
    if (!raw) { alert('Please paste FASTA sequences'); return; }

    // Split by headers
    const sequences = [];
    let current = '';
    for (const line of raw.split('\n')) {
        if (line.startsWith('>') && current) { sequences.push(current); current = ''; }
        current += line + '\n';
    }
    if (current.trim()) sequences.push(current);

    try {
        const res = await fetch(API + '/api/batch_predict', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sequences })
        });
        const data = await res.json();
        if (data.error) { alert(data.error); return; }
        renderBatchResults(data);
    } catch (e) { alert('Batch analysis failed. Is the server running?'); }
}

function renderBatchResults(data) {
    const container = document.getElementById('batch-results');
    let html = `<div class="batch-card">
        <div class="card-title"><span class="icon">📊</span> Batch Analysis Results</div>
        <div class="batch-summary">
            <div class="batch-stat"><div class="val" style="color:var(--accent-cyan)">${data.n_analyzed}</div><div class="lbl">Analyzed</div></div>
            <div class="batch-stat"><div class="val" style="color:var(--accent-red)">${data.n_mdr}</div><div class="lbl">MDR Detected</div></div>
            <div class="batch-stat"><div class="val" style="color:var(--accent-purple)">${data.clusters?.length || 0}</div><div class="lbl">Clusters</div></div>
            <div class="batch-stat"><div class="val" style="color:var(--accent-green)">${data.results.filter(r => !r.mdr).length}</div><div class="lbl">Non-MDR</div></div>
        </div>
        <table class="batch-table"><thead><tr><th>Isolate</th><th>Ampicillin</th><th>Ciprofloxacin</th><th>Gentamicin</th><th>MDR</th></tr></thead><tbody>`;

    for (const r of data.results) {
        if (r.error) { html += `<tr><td colspan="5" style="color:var(--accent-red)">Error: ${r.error}</td></tr>`; continue; }
        html += `<tr><td style="font-family:'JetBrains Mono',monospace;font-size:11px">${r.name}</td>`;
        for (const ab of ['ampicillin', 'ciprofloxacin', 'gentamicin']) {
            const p = r.predictions[ab]?.phenotype || '?';
            const cls = p === 'Resistant' ? 'r' : p === 'Susceptible' ? 's' : 'i';
            html += `<td><span class="batch-phenotype ${cls}">${p}</span></td>`;
        }
        html += `<td>${r.mdr ? '<span style="color:var(--accent-red);font-weight:700">⚠️ MDR</span>' : '<span style="color:var(--accent-green)">—</span>'}</td></tr>`;
    }
    html += '</tbody></table>';

    if (data.clusters && data.clusters.length > 0) {
        html += '<div class="cluster-section"><div class="cluster-title">🧬 Epidemiological Clusters</div>';
        data.clusters.forEach((c, i) => {
            const profile = Object.entries(c.profile).map(([ab, p]) => {
                const color = p === 'Resistant' ? 'var(--accent-red)' : p === 'Susceptible' ? 'var(--accent-green)' : 'var(--accent-amber)';
                return `<span style="color:${color};font-weight:700">${ab}: ${p}</span>`;
            }).join(' · ');
            html += `<div class="cluster-item"><div style="font-weight:700;font-size:12px;margin-bottom:4px">Cluster ${i + 1} (${c.isolates.length} isolate${c.isolates.length > 1 ? 's' : ''})</div><div style="font-size:11px">${profile}</div><div style="font-size:10px;color:var(--text-muted);margin-top:4px">${c.isolates.join(', ')}</div></div>`;
        });
        html += '</div>';
    }
    html += '</div>';
    container.innerHTML = html;
}

// ── History ───────────────────────────────────────────────────
function saveToHistory(data) {
    try {
        const history = JSON.parse(localStorage.getItem('resistai_history') || '[]');
        history.unshift({
            timestamp: new Date().toISOString(),
            predictions: data.predictions,
            n_genes: data.detected_genes?.length || 0,
            mdr: data.clinical?.mdr_risk || false,
            stewardship: data.clinical?.stewardship_score || 0
        });
        localStorage.setItem('resistai_history', JSON.stringify(history.slice(0, 20)));
    } catch (e) { }
}

function renderHistory() {
    const list = document.getElementById('history-list');
    try {
        const history = JSON.parse(localStorage.getItem('resistai_history') || '[]');
        if (history.length === 0) { list.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:40px">No analyses saved yet.</p>'; return; }
        let html = '';
        history.forEach(h => {
            const date = new Date(h.timestamp);
            const phenotypes = Object.values(h.predictions).map(p => p.phenotype);
            html += `<div class="history-item">
                <div class="history-time">${date.toLocaleDateString()} ${date.toLocaleTimeString()}</div>
                <div class="history-profile">${phenotypes.map(p => `<div class="history-dot" style="background:${p === 'Resistant' ? 'var(--accent-red)' : p === 'Susceptible' ? 'var(--accent-green)' : 'var(--accent-amber)'}"></div>`).join('')}</div>
                <div class="history-label">${h.n_genes} genes · ${h.mdr ? '⚠️ MDR' : 'Clean'} · Score: ${h.stewardship}</div>
            </div>`;
        });
        list.innerHTML = html;
    } catch (e) { list.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:40px">Could not load history.</p>'; }
}

// ── PDF Report ────────────────────────────────────────────────
function downloadReport() {
    if (!currentResults) { alert('Run a prediction first!'); return; }
    const d = currentResults;
    let text = `RESISTAI — CLINICAL RESISTANCE REPORT\n${'═'.repeat(50)}\nGenerated: ${new Date().toLocaleString()}\nOrganism: Escherichia coli\n\n`;
    text += `SEQUENCE SUMMARY\n${'─'.repeat(30)}\nTotal Length: ${d.seq_stats?.total_length?.toLocaleString()} bp\nContigs: ${d.seq_stats?.n_contigs}\nGC Content: ${d.seq_stats?.gc_content}%\n\n`;
    text += `RESISTANCE PREDICTIONS\n${'─'.repeat(30)}\n`;
    for (const [ab, p] of Object.entries(d.predictions)) {
        text += `${ab.toUpperCase()}: ${p.phenotype} (${(p.confidence * 100).toFixed(1)}% confidence)\n`;
        text += `  Susceptible: ${(p.probabilities.Susceptible * 100).toFixed(1)}% | Intermediate: ${(p.probabilities.Intermediate * 100).toFixed(1)}% | Resistant: ${(p.probabilities.Resistant * 100).toFixed(1)}%\n`;
    }
    text += `\nCLINICAL RECOMMENDATIONS\n${'─'.repeat(30)}\n`;
    for (const rec of d.clinical.recommendations) {
        text += `${rec.status} — ${rec.antibiotic} (WHO: ${rec.who_category})\n  ${rec.message}\n`;
    }
    text += `\nOverall: ${d.clinical.overall}\nStewardship Score: ${d.clinical.stewardship_score}/100\nMDR Risk: ${d.clinical.mdr_risk ? 'YES' : 'NO'}\n`;
    text += `\nDETECTED RESISTANCE GENES (${d.detected_genes.length})\n${'─'.repeat(30)}\n`;
    for (const g of d.detected_genes) {
        text += `• ${g.gene} [${g.category}] — ${g.mechanism_info?.mechanism || 'Unknown'}\n`;
    }
    text += `\n${'═'.repeat(50)}\nDISCLAIMER: This report is for RESEARCH and DECISION SUPPORT only.\nAlways confirm with phenotypic antimicrobial susceptibility testing (AST).\n`;

    const blob = new Blob([text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `ResistAI_Report_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
}

// ── Particle Celebration ──────────────────────────────────────
function triggerParticles() {
    const canvas = document.getElementById('particle-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.style.opacity = '1';

    const particles = [];
    const colors = ['#8b5cf6', '#06b6d4', '#22c55e', '#f59e0b', '#ec4899', '#ef4444'];
    for (let i = 0; i < 80; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: canvas.height + 10,
            vx: (Math.random() - 0.5) * 6,
            vy: -Math.random() * 12 - 6,
            r: Math.random() * 4 + 2,
            color: colors[Math.floor(Math.random() * colors.length)],
            alpha: 1,
            gravity: 0.12
        });
    }

    let frame = 0;
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        let alive = false;
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy; p.vy += p.gravity; p.alpha -= 0.008;
            if (p.alpha > 0) {
                alive = true;
                ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = p.color + Math.round(p.alpha * 255).toString(16).padStart(2, '0');
                ctx.fill();
            }
        });
        frame++;
        if (alive && frame < 200) requestAnimationFrame(animate);
        else { canvas.style.opacity = '0'; }
    }
    animate();
}

// ── DNA Canvas Background ─────────────────────────────────────
function initDNACanvas() {
    const canvas = document.getElementById('dna-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    let t = 0;
    function drawDNA() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const baseY = canvas.height / 2;
        const amp = 60, freq = 0.015, spacing = 30;

        for (let x = 0; x < canvas.width; x += spacing) {
            const y1 = baseY + Math.sin(x * freq + t) * amp;
            const y2 = baseY - Math.sin(x * freq + t) * amp;

            ctx.beginPath(); ctx.arc(x, y1, 2, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(139,92,246,0.15)'; ctx.fill();

            ctx.beginPath(); ctx.arc(x, y2, 2, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(6,182,212,0.15)'; ctx.fill();

            if (Math.floor(x / spacing) % 3 === 0) {
                ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y2);
                ctx.strokeStyle = 'rgba(255,255,255,0.03)'; ctx.lineWidth = 1; ctx.stroke();
            }
        }
        t += 0.02;
        requestAnimationFrame(drawDNA);
    }
    drawDNA();

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}
