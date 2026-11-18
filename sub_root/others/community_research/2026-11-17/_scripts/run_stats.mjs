// node run_stats.mjs — GitHub API stats/commit_activity：按月统计 2026-05..2026-11 提交数
// 用法: node run_stats.mjs repo1 repo2 ... （顺序执行，打印剩余配额）
const repos = process.argv.slice(2);
const delay = (ms) => new Promise(r => setTimeout(r, ms));
const H = { 'user-agent': 'pysci-research/1.0', 'accept': 'application/vnd.github+json' };
const MONTHS = ['2026-05', '2026-06', '2026-07', '2026-08', '2026-09', '2026-10', '2026-11'];

async function getStats(repo) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const r = await fetch(`https://api.github.com/repos/${repo}/stats/commit_activity`, { headers: H });
    const rem = r.headers.get('x-ratelimit-remaining');
    if (r.status === 200) { const j = await r.json(); return { data: j, rem }; }
    if (r.status === 202) { await delay(2500); continue; }
    return { err: `status=${r.status}`, rem };
  }
  return { err: '202 timeout' };
}

for (const repo of repos) {
  const out = { repo };
  try {
    const { data, err, rem } = await getStats(repo);
    out.rem = rem;
    if (err) { out.err = err; }
    else if (!Array.isArray(data) || data.length === 0) { out.err = 'empty'; }
    else {
      const counts = Object.fromEntries(MONTHS.map(m => [m, 0]));
      let overall = 0;
      for (const w of data) {
        const start = w.week * 1000;
        for (let i = 0; i < 7; i++) {
          const d = new Date(start + i * 86400 * 1000);
          const key = d.toISOString().slice(0, 7);
          if (key in counts) { counts[key] += w.days[i] || 0; overall += w.days[i] || 0; }
        }
      }
      const earliest = new Date(data[0].week * 1000).toISOString().slice(0, 10);
      const latest = new Date(data[data.length - 1].week * 1000).toISOString().slice(0, 10);
      out.monthly = counts;
      out.window_total = overall;
      out.all_months = MONTHS.every(m => counts[m] > 0);
      out.missing = MONTHS.filter(m => !(counts[m] > 0));
      out.data_span = `${earliest}..${latest}`;
    }
  } catch (e) { out.err = e.message; }
  console.log(JSON.stringify(out));
  await delay(700);
}
