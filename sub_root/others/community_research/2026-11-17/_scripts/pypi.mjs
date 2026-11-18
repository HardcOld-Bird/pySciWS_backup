// node pypi.mjs <pkg> — PyPI JSON：版本/日期/requires_python/许可证/是否含 cp313 win_amd64 wheel
const pkg = process.argv[2];
const url = `https://pypi.org/pypi/${encodeURIComponent(pkg)}/json`;
try {
  const r = await fetch(url, { headers: { 'user-agent': 'pysci-research/1.0', 'accept': 'application/json' } });
  if (r.status !== 200) { console.log(`# ${pkg}: status=${r.status}`); process.exit(0); }
  const j = await r.json();
  const info = j.info || {};
  const ver = info.version;
  const files = (j.releases && j.releases[ver]) || [];
  const now = new Date('2026-11-17T00:00:00Z');
  let latestUpload = null, anyUpload = null;
  for (const f of files) { if (f.upload_time_iso_8601) { if (!latestUpload || f.upload_time_iso_8601 > latestUpload) latestUpload = f.upload_time_iso_8601; } }
  // 扫描全部 release 是否存在 cp313 win_amd64 或 cp313 abi3 win_amd64 wheel
  let cp313Win = null, cp313Any = null, winAnyTag = null;
  for (const [v, fl] of Object.entries(j.releases)) {
    for (const f of fl) {
      const n = f.filename || '';
      if (!n.endsWith('.whl')) continue;
      if (/cp313/.test(n)) { cp313Any = v; if (/win_amd64/.test(n)) { cp313Win = v; } }
      if (/win_amd64/.test(n)) winAnyTag = v + ':' + n;
    }
  }
  const urls = info.project_urls || {};
  const cls = (info.classifiers || []).filter(c => c.startsWith('License ::')).slice(0, 3);
  const pyCls = (info.classifiers || []).filter(c => /^Programming Language :: Python :: 3\.1[2-4]$/.test(c));
  console.log(JSON.stringify({
    pkg, version: ver, latest_upload: latestUpload, requires_python: info.requires_python,
    license_field: info.license ? String(info.license).slice(0, 60) : null, license_classifiers: cls,
    py313_classifier: pyCls.length > 0, cp313_win_amd64: cp313Win, cp313_any_wheel: cp313Any,
    home: info.home_page || urls.Homepage || urls.Home || urls.Source || urls.Repository || null,
    urls,
  }));
} catch (e) { console.log(`# ${pkg} FAIL ` + (e.cause ? String(e.cause) : e.message)); }
