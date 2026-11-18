// node d03_month.mjs <owner/repo> ... — ecosyste.ms 逐月提交探测 2026-05..2026-11（domain-03 绘图域复用 domain-05 方法）
// 输出 JSON 行：{repo, monthly:{月:count}, present, missing, all_months, errors}
const repos = process.argv.slice(2);
const MONTHS = ['2026-05', '2026-06', '2026-07', '2026-08', '2026-09', '2026-10', '2026-11'];
const delay = (ms) => new Promise(r => setTimeout(r, ms));
const H = { 'user-agent': 'pysci-research/1.0 (community survey)', 'accept': 'application/json' };
function lastDay(m) { const [y, mo] = m.split('-').map(Number); return new Date(Date.UTC(y, mo, 0)).getUTCDate(); }

for (const repo of repos) {
  const out = { repo, monthly: {}, errors: [] };
  const enc = encodeURIComponent(repo);
  for (const m of MONTHS) {
    const since = `${m}-01`;
    const until = `${m}-${String(lastDay(m)).padStart(2, '0')}`;
    const url = `https://commits.ecosyste.ms/api/v1/hosts/GitHub/repositories/${enc}/commits/?since=${since}&until=${until}&per_page=1`;
    try {
      const r = await fetch(url, { headers: H });
      const t = await r.text();
      if (r.status !== 200) { out.errors.push(`${m}:status=${r.status}`); out.monthly[m] = null; await delay(800); continue; }
      const j = JSON.parse(t);
      out.monthly[m] = Array.isArray(j) ? j.length : (j && j.commits ? j.commits.length : null);
      if (out.monthly[m] === null) out.errors.push(`${m}:shape=${t.slice(0, 80)}`);
    } catch (e) { out.errors.push(`${m}:${e.message}`); out.monthly[m] = null; }
    await delay(500);
  }
  out.present = MONTHS.filter(m => out.monthly[m] > 0);
  out.missing = MONTHS.filter(m => !(out.monthly[m] > 0));
  out.all_months = out.missing.length === 0;
  console.log(JSON.stringify(out));
  await delay(400);
}
