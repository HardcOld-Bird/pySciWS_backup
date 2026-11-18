// node eco.mjs <owner/repo> — ecosyste.ms 完整 JSON（非浏览器 UA）
const repo = process.argv[2];
const url = `https://repos.ecosyste.ms/api/v1/hosts/GitHub/repositories/${encodeURIComponent(repo)}`;
try {
  const r = await fetch(url, { headers: { 'user-agent': 'pysci-research/1.0 (community survey)', 'accept': 'application/json' } });
  const t = await r.text();
  console.log(`# status=${r.status}`);
  try {
    const j = JSON.parse(t);
    console.log('KEYS: ' + Object.keys(j).join(','));
    console.log(JSON.stringify(j, null, 1).slice(0, 4000));
  } catch { console.log(t.slice(0, 1000)); }
} catch (e) { console.log('FAIL ' + (e.cause ? String(e.cause) : e.message)); }
