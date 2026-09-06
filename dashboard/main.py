"""
FastAPI dashboard — REST API + simple HTML UI for viewing scan results.

Endpoints:
  GET  /                        → HTML dashboard
  GET  /api/packages            → paginated list of scanned packages
  GET  /api/packages/{id}       → single package detail + signals
  POST /api/scan                → scan a package on demand
  GET  /api/stats               → summary counts by risk level
  POST /api/allowlist           → add a package to the allowlist
  GET  /api/allowlist           → list allowlisted packages

Run:
  cd supply-chain-detector
  uvicorn dashboard.main:app --reload --port 8000
  open http://localhost:8000
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import func

from db.models import (
    Package, ScanResult, DetectorSignal, Allowlist,
    Registry, RiskLevel, init_db, get_session
)
from monitor.package_fetcher import PackageFetcher
from detectors.name_similarity import NameSimilarityDetector, POPULAR_PYPI, POPULAR_NPM
from detectors.metadata_heuristics import MetadataHeuristicsDetector
from detectors.static_analysis import StaticAnalysisDetector
from scoring.risk_scorer import RiskScorer
from integrations.alert_engine import AlertEngine

logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────

app    = FastAPI(title="Supply Chain Detector", version="1.0.0")
engine = init_db()


# ── Pydantic schemas ───────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    registry: str   # "npm" or "pypi"
    name: str

class AllowlistRequest(BaseModel):
    registry: str
    name: str
    reason: str = ""


# ── API routes ─────────────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats():
    """Return count of packages by risk level."""
    with get_session(engine) as session:
        total = session.query(Package).count()
        counts = (
            session.query(ScanResult.risk_level, func.count(ScanResult.id))
            .group_by(ScanResult.risk_level)
            .all()
        )
        by_level = {level.value: count for level, count in counts if level is not None}
        unscanned = total - sum(by_level.values())

    return {
        "total":     total,
        "high":      by_level.get("high", 0),
        "medium":    by_level.get("medium", 0),
        "low":       by_level.get("low", 0),
        "unscanned": unscanned,
    }


@app.get("/api/packages")
def list_packages(
    page:       int = Query(1, ge=1),
    per_page:   int = Query(20, ge=1, le=100),
    risk_level: str = Query(None),
    registry:   str = Query(None),
    search:     str = Query(None),
):
    """Paginated list of packages with their scan results."""
    with get_session(engine) as session:
        q = session.query(Package)

        if registry:
            try:
                q = q.filter(Package.registry == Registry(registry.lower()))
            except ValueError:
                raise HTTPException(400, f"Unknown registry: {registry}")

        if search:
            q = q.filter(Package.name.ilike(f"%{search.strip()}%"))

        if risk_level:
            try:
                rl = RiskLevel(risk_level.lower())
                q  = q.join(ScanResult).filter(ScanResult.risk_level == rl)
            except ValueError:
                raise HTTPException(400, f"Unknown risk level: {risk_level}")

        total  = q.count()
        offset = (page - 1) * per_page
        pkgs   = q.order_by(Package.discovered_at.desc()).offset(offset).limit(per_page).all()

        items = []
        for pkg in pkgs:
            sr = pkg.scan_result
            items.append({
                "id":          pkg.id,
                "registry":    pkg.registry.value if pkg.registry else "unknown",
                "name":        pkg.name,
                "version":     pkg.version,
                "author":      pkg.author,
                "discovered":  pkg.discovered_at.isoformat() if pkg.discovered_at else None,
                "risk_level":  sr.risk_level.value if sr and sr.risk_level else "unknown",
                "score":       sr.risk_score if sr else None,
                "verdict":     sr.verdict_reason if sr else None,
                "closest":     sr.closest_match if sr else None,
            })

    return {"total": total, "page": page, "per_page": per_page, "items": items}


@app.get("/api/packages/{package_id}")
def get_package(package_id: int):
    """Full detail for one package including all detector signals."""
    with get_session(engine) as session:
        pkg = session.query(Package).filter_by(id=package_id).first()
        if not pkg:
            raise HTTPException(404, "Package not found")

        sr = pkg.scan_result
        signals = []
        if sr:
            for sig in sr.signals:
                signals.append({
                    "detector":    sig.detector_name,
                    "value":       sig.signal_value,
                    "weight":      sig.weight,
                    "contribution":sig.contribution,
                    "detail":      sig.detail,
                })

        return {
            "id":           pkg.id,
            "registry":     pkg.registry.value if pkg.registry else "unknown",
            "name":         pkg.name,
            "version":      pkg.version,
            "author":       pkg.author,
            "description":  pkg.description,
            "homepage":     pkg.homepage,
            "repository":   pkg.repository_url,
            "first_published": pkg.first_published.isoformat() if pkg.first_published else None,
            "discovered":   pkg.discovered_at.isoformat() if pkg.discovered_at else None,
            "scan": {
                "risk_level":  sr.risk_level.value if sr and sr.risk_level else "unknown",
                "score":       sr.risk_score if sr else None,
                "is_typosquat":           sr.is_typosquat if sr else False,
                "is_dependency_confusion":sr.is_dependency_confusion if sr else False,
                "has_suspicious_scripts": sr.has_suspicious_scripts if sr else False,
                "closest_match":  sr.closest_match if sr else None,
                "closest_dist":   sr.closest_match_dist if sr else None,
                "verdict":        sr.verdict_reason if sr else None,
                "scanned_at":     sr.scanned_at.isoformat() if sr and sr.scanned_at else None,
            } if sr else None,
            "signals": signals,
        }


@app.post("/api/scan")
async def scan_package(req: ScanRequest):
    """Fetch and scan a package on demand. Returns the risk result."""
    reg = req.registry.strip().lower()
    pkg_name = req.name.strip()

    if reg not in ("npm", "pypi"):
        raise HTTPException(400, f"Unknown registry: {req.registry}")

    fetcher = PackageFetcher(db_engine=engine)
    try:
        pkg = await fetcher.fetch_and_save(reg, pkg_name)
    except Exception as exc:
        logger.error(f"Error fetching package {reg}/{pkg_name}: {exc}")
        raise HTTPException(500, f"Failed to fetch package: {str(exc)}")

    if not pkg:
        raise HTTPException(404, f"Package {reg}/{pkg_name} not found on registry")

    ref_list = POPULAR_PYPI if reg == "pypi" else POPULAR_NPM
    scorer = RiskScorer(
        similarity_detector=NameSimilarityDetector(ref_list),
        metadata_detector=MetadataHeuristicsDetector(),
        static_detector=StaticAnalysisDetector(),
        db_engine=engine,
    )
    bd = scorer.score(pkg)

    # Fire alerts if warranted
    with get_session(engine) as session:
        result_row = session.query(ScanResult).filter_by(package_id=pkg.id).first()
        if result_row:
            AlertEngine().dispatch(pkg, result_row)

    return {
        "registry":   reg,
        "name":       pkg_name,
        "version":    pkg.version,
        "risk_level": bd.risk_level.value,
        "score":      bd.total,
        "breakdown": {
            "similarity": bd.similarity_score,
            "metadata":   bd.metadata_score,
            "static":     bd.static_score,
        },
        "verdict": bd.verdict_reason,
    }


@app.get("/api/allowlist")
def get_allowlist():
    with get_session(engine) as session:
        rows = session.query(Allowlist).all()
        return [
            {
                "registry": r.registry.value if r.registry else "unknown",
                "name": r.name,
                "reason": r.reason,
                "added_at": r.added_at.isoformat() if r.added_at else None
            }
            for r in rows
        ]


@app.post("/api/allowlist")
def add_to_allowlist(req: AllowlistRequest):
    reg = req.registry.strip().lower()
    pkg_name = req.name.strip()

    if reg not in ("npm", "pypi"):
        raise HTTPException(400, f"Unknown registry: {req.registry}")
    with get_session(engine) as session:
        existing = session.query(Allowlist).filter_by(
            registry=Registry(reg), name=pkg_name
        ).first()
        if existing:
            return {"status": "already_exists"}
        entry = Allowlist(registry=Registry(reg), name=pkg_name, reason=req.reason)
        session.add(entry)
        session.commit()
    return {"status": "added", "registry": reg, "name": pkg_name}


# ── HTML Dashboard ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Supply Chain Detector</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f0; color: #1a1a1a; line-height: 1.6; }
  header { background: #1a1a1a; color: #fff; padding: 16px 32px;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; font-weight: 500; }
  header span { font-size: 13px; color: #888; margin-left: auto; }
  .container { max-width: 1100px; margin: 0 auto; padding: 24px 24px; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
  .stat-card { background: #fff; border-radius: 8px; border: 0.5px solid #e0e0da;
               padding: 16px 20px; }
  .stat-card .label { font-size: 12px; color: #888; text-transform: uppercase;
                      letter-spacing: .04em; margin-bottom: 4px; }
  .stat-card .value { font-size: 28px; font-weight: 500; }
  .value.high   { color: #c0392b; }
  .value.medium { color: #e67e22; }
  .value.low    { color: #27ae60; }
  .card { background: #fff; border-radius: 10px; border: 0.5px solid #e0e0da;
          padding: 20px 24px; margin-bottom: 20px; }
  .card h2 { font-size: 15px; font-weight: 500; margin-bottom: 16px; }
  .scan-row { display: flex; gap: 10px; margin-bottom: 16px; }
  .scan-row input, .scan-row select {
    border: 0.5px solid #d0d0ca; border-radius: 6px;
    padding: 8px 12px; font-size: 14px; background: #fafafa; }
  .scan-row input { flex: 1; }
  .scan-row button {
    background: #1a1a1a; color: #fff; border: none;
    border-radius: 6px; padding: 8px 20px; font-size: 14px; cursor: pointer; }
  .scan-row button:hover { background: #333; }
  .filters { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
  .filter-btn { padding: 5px 14px; border-radius: 999px; font-size: 13px; cursor: pointer;
                border: 0.5px solid #d0d0ca; background: transparent; color: #555; }
  .filter-btn.active { background: #1a1a1a; color: #fff; border-color: #1a1a1a; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; font-size: 11px; font-weight: 500; color: #888;
       text-transform: uppercase; letter-spacing: .04em;
       padding: 0 12px 10px 0; border-bottom: 0.5px solid #e0e0da; }
  td { padding: 10px 12px 10px 0; border-bottom: 0.5px solid #f0f0ea;
       vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #fafafa; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
           font-size: 11px; font-weight: 500; }
  .badge.high   { background: #fde8e8; color: #a32d2d; }
  .badge.medium { background: #fef3e2; color: #854f0b; }
  .badge.low    { background: #eaf3de; color: #3b6d11; }
  .badge.unknown{ background: #f0f0ea; color: #888; }
  .score-bar { display: flex; align-items: center; gap: 8px; }
  .bar-bg { flex: 1; height: 4px; background: #f0f0ea; border-radius: 2px; }
  .bar-fill { height: 4px; border-radius: 2px; background: #ccc; }
  .mono { font-family: 'SF Mono', 'Fira Code', monospace; }
  .pkg-link { color: #1a1a1a; cursor: pointer; text-decoration: underline;
              text-decoration-color: #ccc; }
  .pkg-link:hover { text-decoration-color: #1a1a1a; }
  #detail-panel { display: none; }
  #detail-panel.open { display: block; }
  #result-msg { font-size: 13px; color: #555; margin-bottom: 12px; min-height: 20px; }
  .signal-row { display: flex; align-items: center; gap: 10px; padding: 6px 0;
                border-bottom: 0.5px solid #f0f0ea; font-size: 13px; }
  .signal-row:last-child { border-bottom: none; }
  .sig-name { width: 220px; color: #555; flex-shrink: 0; font-family: monospace; font-size: 12px; }
  .sig-bar-bg { flex: 1; height: 3px; background: #f0f0ea; border-radius: 2px; }
  .sig-bar-fill { height: 3px; border-radius: 2px; }
  .sig-contrib { width: 40px; text-align: right; color: #888; font-size: 12px; }
  .pagination { display: flex; gap: 8px; margin-top: 16px; align-items: center;
                font-size: 13px; color: #555; }
  .page-btn { padding: 4px 12px; border: 0.5px solid #d0d0ca; border-radius: 4px;
              background: transparent; cursor: pointer; font-size: 13px; }
  .page-btn:disabled { opacity: 0.4; cursor: default; }
</style>
</head>
<body>
<header>
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <path d="M10 2L18 6V10C18 14.4 14.4 17.9 10 19C5.6 17.9 2 14.4 2 10V6L10 2Z"
          stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/>
    <path d="M7 10L9 12L13 8" stroke="#fff" stroke-width="1.5"
          stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <h1>Supply Chain Detector</h1>
  <span id="header-time"></span>
</header>

<div class="container">
  <!-- Stats row -->
  <div class="stats" id="stats-row">
    <div class="stat-card"><div class="label">Total scanned</div><div class="value" id="s-total">—</div></div>
    <div class="stat-card"><div class="label">High risk</div><div class="value high" id="s-high">—</div></div>
    <div class="stat-card"><div class="label">Medium risk</div><div class="value medium" id="s-medium">—</div></div>
    <div class="stat-card"><div class="label">Low risk</div><div class="value low" id="s-low">—</div></div>
  </div>

  <!-- Scan a package -->
  <div class="card">
    <h2>Scan a package</h2>
    <div class="scan-row">
      <select id="scan-registry">
        <option value="pypi">PyPI</option>
        <option value="npm">npm</option>
      </select>
      <input id="scan-name" type="text" placeholder="Package name, e.g. reqests" />
      <button onclick="scanPackage()">Scan ↗</button>
    </div>
    <div id="result-msg"></div>
    <div id="scan-result"></div>
  </div>

  <!-- Package list -->
  <div class="card">
    <h2>Scan history</h2>
    <div class="filters">
      <button class="filter-btn active" onclick="setFilter('all',this)">All</button>
      <button class="filter-btn" onclick="setFilter('high',this)">High</button>
      <button class="filter-btn" onclick="setFilter('medium',this)">Medium</button>
      <button class="filter-btn" onclick="setFilter('low',this)">Low</button>
    </div>
    <table>
      <thead>
        <tr>
          <th>Package</th>
          <th>Registry</th>
          <th>Version</th>
          <th>Risk</th>
          <th>Score</th>
          <th>Verdict</th>
        </tr>
      </thead>
      <tbody id="pkg-table-body">
        <tr><td colspan="6" style="color:#888;padding:20px 0">Loading...</td></tr>
      </tbody>
    </table>
    <div class="pagination">
      <button class="page-btn" id="prev-btn" onclick="changePage(-1)" disabled>← Prev</button>
      <span id="page-info">Page 1</span>
      <button class="page-btn" id="next-btn" onclick="changePage(1)">Next →</button>
      <span id="total-info" style="margin-left:auto;color:#aaa"></span>
    </div>
  </div>

  <!-- Detail panel -->
  <div class="card" id="detail-panel">
    <h2 id="detail-title">Package detail</h2>
    <div id="detail-body"></div>
  </div>
</div>

<script>
let currentFilter = 'all';
let currentPage   = 1;
const PER_PAGE    = 15;

document.getElementById('header-time').textContent = new Date().toLocaleString();

async function loadStats() {
  const r = await fetch('/api/stats').then(r => r.json());
  document.getElementById('s-total').textContent  = r.total;
  document.getElementById('s-high').textContent   = r.high;
  document.getElementById('s-medium').textContent = r.medium;
  document.getElementById('s-low').textContent    = r.low;
}

async function loadPackages() {
  const params = new URLSearchParams({
    page: currentPage, per_page: PER_PAGE
  });
  if (currentFilter !== 'all') params.set('risk_level', currentFilter);

  const data = await fetch('/api/packages?' + params).then(r => r.json());
  const body = document.getElementById('pkg-table-body');

  if (!data.items.length) {
    body.innerHTML = '<tr><td colspan="6" style="color:#888;padding:20px 0">No packages found.</td></tr>';
    return;
  }

  const barColor = { high:'#c0392b', medium:'#e67e22', low:'#27ae60', unknown:'#ccc' };

  body.innerHTML = data.items.map(p => {
    const score  = p.score != null ? p.score.toFixed(1) : '—';
    const fill   = p.score != null ? p.score : 0;
    const color  = barColor[p.risk_level] || '#ccc';
    const verdict= (p.verdict || '').slice(0, 60) + ((p.verdict || '').length > 60 ? '…' : '');
    return `<tr>
      <td><span class="pkg-link mono" onclick="loadDetail(${p.id})">${p.name}</span></td>
      <td style="color:#888">${p.registry}</td>
      <td style="color:#888" class="mono">${p.version || '—'}</td>
      <td><span class="badge ${p.risk_level}">${p.risk_level}</span></td>
      <td>
        <div class="score-bar">
          <div class="bar-bg"><div class="bar-fill" style="width:${fill}%;background:${color}"></div></div>
          <span style="font-size:12px;color:#888;min-width:30px">${score}</span>
        </div>
      </td>
      <td style="color:#888;font-size:12px">${verdict || '—'}</td>
    </tr>`;
  }).join('');

  const totalPages = Math.ceil(data.total / PER_PAGE);
  document.getElementById('page-info').textContent  = `Page ${currentPage} of ${totalPages}`;
  document.getElementById('total-info').textContent = `${data.total} total`;
  document.getElementById('prev-btn').disabled      = currentPage <= 1;
  document.getElementById('next-btn').disabled      = currentPage >= totalPages;
}

function setFilter(f, btn) {
  currentFilter = f;
  currentPage   = 1;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  loadPackages();
}

function changePage(delta) {
  currentPage += delta;
  loadPackages();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function loadDetail(id) {
  const p = await fetch('/api/packages/' + id).then(r => r.json());
  const panel = document.getElementById('detail-panel');
  document.getElementById('detail-title').textContent = `${p.registry}/${p.name}`;

  const sc = p.scan || {};
  const barColor = { high:'#c0392b', medium:'#e67e22', low:'#27ae60', unknown:'#ccc' };
  const color = barColor[sc.risk_level] || '#ccc';

  const sigRows = (p.signals || []).map(s => {
    const pct = Math.min(s.contribution, 40);
    const c2  = s.contribution > 15 ? '#c0392b' : s.contribution > 5 ? '#e67e22' : '#27ae60';
    return `<div class="signal-row">
      <span class="sig-name">${s.detector}</span>
      <div class="sig-bar-bg"><div class="sig-bar-fill" style="width:${pct*2.5}%;background:${c2}"></div></div>
      <span class="sig-contrib">+${s.contribution.toFixed(1)}</span>
      <span style="font-size:12px;color:#aaa;flex:2">${s.detail || ''}</span>
    </div>`;
  }).join('');

  document.getElementById('detail-body').innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
      <div>
        <div style="font-size:12px;color:#888;margin-bottom:4px">Author</div>
        <div>${p.author || '—'}</div>
      </div>
      <div>
        <div style="font-size:12px;color:#888;margin-bottom:4px">Version</div>
        <div class="mono">${p.version || '—'}</div>
      </div>
      <div>
        <div style="font-size:12px;color:#888;margin-bottom:4px">Risk level</div>
        <div><span class="badge ${sc.risk_level}">${sc.risk_level || 'unknown'}</span></div>
      </div>
      <div>
        <div style="font-size:12px;color:#888;margin-bottom:4px">Score</div>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="bar-bg" style="width:120px"><div class="bar-fill"
            style="width:${sc.score || 0}%;background:${color}"></div></div>
          <span>${sc.score != null ? sc.score.toFixed(1) : '—'} / 100</span>
        </div>
      </div>
    </div>
    <div style="margin-bottom:16px">
      <div style="font-size:12px;color:#888;margin-bottom:4px">Verdict</div>
      <div style="font-size:13px">${sc.verdict || '—'}</div>
    </div>
    ${sigRows ? `<div style="font-size:12px;color:#888;margin-bottom:8px;text-transform:uppercase;letter-spacing:.04em">Detector signals</div>${sigRows}` : ''}
  `;

  panel.classList.add('open');
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function scanPackage() {
  const registry = document.getElementById('scan-registry').value;
  const name     = document.getElementById('scan-name').value.trim();
  const msgEl    = document.getElementById('result-msg');
  const resEl    = document.getElementById('scan-result');

  if (!name) { msgEl.textContent = 'Enter a package name.'; return; }

  msgEl.textContent = `Scanning ${registry}/${name}...`;
  resEl.innerHTML   = '';

  try {
    const data = await fetch('/api/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ registry, name })
    }).then(r => r.json());

    if (!data.risk_level) {
      msgEl.textContent = data.detail || 'Scan failed.';
      return;
    }

    const color = { high:'#c0392b', medium:'#e67e22', low:'#27ae60' }[data.risk_level] || '#888';
    msgEl.textContent = '';
    resEl.innerHTML = `
      <div style="display:flex;align-items:center;gap:12px;padding:12px 0;border-top:0.5px solid #f0f0ea">
        <span class="badge ${data.risk_level}" style="font-size:13px;padding:4px 12px">${data.risk_level.toUpperCase()}</span>
        <span style="font-size:22px;font-weight:500;color:${color}">${data.score.toFixed(1)}<span style="font-size:14px;font-weight:400;color:#888"> / 100</span></span>
        <span style="font-size:13px;color:#555;flex:1">${data.verdict || ''}</span>
      </div>
      <div style="font-size:12px;color:#888;margin-top:4px">
        similarity ${data.breakdown.similarity.toFixed(1)} · 
        metadata ${data.breakdown.metadata.toFixed(1)} · 
        static ${data.breakdown.static.toFixed(1)}
      </div>
    `;
    loadStats();
    loadPackages();
  } catch(e) {
    msgEl.textContent = 'Scan failed — is the server running?';
  }
}

document.getElementById('scan-name').addEventListener('keydown', e => {
  if (e.key === 'Enter') scanPackage();
});

loadStats();
loadPackages();
setInterval(loadStats, 30000);
</script>
</body>
</html>
"""