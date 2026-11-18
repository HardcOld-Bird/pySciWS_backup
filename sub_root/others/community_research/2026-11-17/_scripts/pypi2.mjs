// node pypi2.mjs <pkg> — 检查最新版在 win_amd64 下的 wheel 标签（cp313/abi3/pure）
const pkgs = process.argv.slice(2);
const delay = (ms) => new Promise(r => setTimeout(r, ms));
for (const pkg of pkgs) {
  const out = { pkg };
  try {
    const r = await fetch(`https://pypi.org/pypi/${encodeURIComponent(pkg)}/json`, { headers: { 'user-agent': 'pysci-research/1.0' } });
    if (r.status !== 200) { out.err = `status=${r.status}`; console.log(JSON.stringify(out)); continue; }
    const j = await r.json();
    const ver = j.info.version;
    const files = (j.releases && j.releases[ver]) || [];
    const winWheels = files.filter(f => f.filename.endsWith('.whl') && /win_amd64/.test(f.filename)).map(f => f.filename);
    const pureWheels = files.filter(f => f.filename.endsWith('.whl') && /-any\.whl$/.test(f.filename) && /py3-none|py2.py3-none/.test(f.filename)).map(f => f.filename);
    out.version = ver;
    out.upload = files.length ? files.map(f => f.upload_time_iso_8601).sort().pop() : null;
    out.requires_python = j.info.requires_python;
    out.win_wheel_sample = winWheels.slice(0, 4);
    out.pure_wheel = pureWheels.length > 0;
    out.has_sdist = files.some(f => f.filename.endsWith('.tar.gz'));
    out.win_cp313 = winWheels.some(f => /cp313|abi3/.test(f));
    out.win_any_tag = winWheels.some(f => /-any\.whl$/.test(f));
    out.n_files = files.length;
  } catch (e) { out.err = e.message; }
  console.log(JSON.stringify(out));
  await delay(250);
}
