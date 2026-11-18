// node theory_probe.mjs — ④理论计算域候选批量核实
// shields(stars/last-commit/license) + commits.ecosyste.ms 月末探针(2026-05..2026-11) + PyPI JSON
// 用法: node theory_probe.mjs   （候选清单写死在 REPOS/PKGS）
const REPOS = [
  ['sympy/sympy', 'sympy'],
  ['scipy/scipy', 'scipy'],
  ['jax-ml/jax', 'jax'],
  ['sagemath/sage', null],
  ['numpy/numpy', 'numpy'],
  ['mpmath/mpmath', 'mpmath'],
  ['hgrecco/pint', 'Pint'],
  ['gnu-octave/octave', null],
  ['calyau/maxima', null],
  ['SciML/DifferentialEquations.jl', null],
  ['einsteinpy/einsteinpy', 'einsteinpy'],
  ['symengine/symengine.py', 'symengine'],
];
const MONTHS = ['2026-05', '2026-06', '2026-07', '2026-08', '2026-09', '2026-10', '2026-11'];
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
const UA = { 'user-agent': 'pysci-research/1.0 (community survey)', accept: 'application/json' };

function lastDay(m) { const [y, mo] = m.split('-').map(Number); return new Date(Date.UTC(y, mo, 0)).getUTCDate(); }
async function getText(url, html) {
  try {
    const r = await fetch(url, { headers: html ? { 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0', accept: 'text/html' } : UA, redirect: 'follow' });
    return { status: r.status, text: await r.text() };
  } catch (e) { return { status: 0, text: 'FAIL ' + (e.cause ? e.cause.message : e.message) }; }
}

async function shields(repo) {
  const out = {};
  for (const kind of ['stars', 'last-commit', 'license']) {
    const { text } = await getText(`https://img.shields.io/github/${kind}/${repo}${kind === 'last-commit' ? '?format=undefined' : ''}`);
    // shields SVG 里 value 在 <g class="shifted ..."> 的 <text>...</text>；宽松提取所有 <text>
    const vals = [...text.matchAll(/<text[^>]*>([^<]*)<\/text>/g)].map((x) => x[1].trim()).filter(Boolean);
    out[kind] = vals.length ? vals[vals.length - 1] : text.slice(0, 80);
    await delay(300);
  }
  return out;
}

async function monthly(repo) {
  const months = {};
  for (const m of MONTHS) {
    const until = `${m}-${String(lastDay(m)).padStart(2, '0')}`;
    const url = `https://commits.ecosyste.ms/api/v1/hosts/GitHub/repositories/${encodeURIComponent(repo)}/commits?per_page=1&until=${until}`;
    const { status, text } = await getText(url);
    try {
      const j = JSON.parse(text);
      const ts = Array.isArray(j) && j[0] ? (j[0].timestamp || j[0].committed_at || j[0].authored_at || '') : '';
      months[m] = ts.startsWith(m) ? ts.slice(0, 10) : (ts ? `out:${ts.slice(0, 10)}` : 'none');
    } catch { months[m] = `err:${status}`; }
    await delay(400);
  }
  return months;
}

async function pypi(pkg) {
  const { status, text } = await getText(`https://pypi.org/pypi/${encodeURIComponent(pkg)}/json`);
  if (status !== 200) return { pkg, err: `status=${status}` };
  const j = JSON.parse(text);
  const info = j.info || {};
  const ver = info.version;
  let cp313Win = null, cp313Any = null, purePy = false;
  for (const [v, fl] of Object.entries(j.releases || {})) {
    for (const f of fl) {
      const n = f.filename || '';
      if (!n.endsWith('.whl')) continue;
      if (/py3-none-any/.test(n)) purePy = true;
      if (/cp313/.test(n)) { cp313Any = cp313Any || v; if (/win_amd64/.test(n)) cp313Win = cp313Win || v; }
      if (/py3-none-any/.test(n)) { cp313Any = cp313Any || v; cp313Win = cp313Win || v; }
    }
  }
  const cls = (info.classifiers || []).filter((c) => c.startsWith('License ::') || /^Programming Language :: Python :: 3\.1[2-4]$/.test(c));
  // 当前版本是否有 cp313 win_amd64（或纯 py3 wheel）
  const cur = (j.releases && j.releases[ver]) || [];
  const cur_ok = cur.some((f) => /py3-none-any/.test(f.filename || '') || (/cp313/.test(f.filename || '') && /win_amd64/.test(f.filename || '')));
  return { pkg, version: ver, requires_python: info.requires_python, license: info.license ? String(info.license).slice(0, 50) : null, classifiers: cls, cur_cp313_win: cur_ok, cp313_win_anyver: cp313Win, pure_wheel: purePy };
}

for (const [repo, pkg] of REPOS) {
  const out = { repo, pkg };
  out.shields = await shields(repo);
  out.months = await monthly(repo);
  out.missing = MONTHS.filter((m) => !/^\d{4}-\d{2}-\d{2}$/.test(out.months[m] || ''));
  if (pkg) out.pypi = await pypi(pkg);
  console.log(JSON.stringify(out));
}
