// node run_monthprobe.mjs <owner/repo> [branch] — 逐月探测默认分支 2026-05..2026-11 是否有提交（HTML，since/until 过滤）
const args = process.argv.slice(2);
const repo = args[0];
const branch = args[1];
const MONTHS = ['2026-05', '2026-06', '2026-07', '2026-08', '2026-09', '2026-10', '2026-11'];
const delay = (ms) => new Promise(r => setTimeout(r, ms));
function lastDay(m) { const [y, mo] = m.split('-').map(Number); return new Date(Date.UTC(y, mo, 0)).getUTCDate(); }

const result = { repo, months: {}, errors: [] };
for (const m of MONTHS) {
  const since = `${m}-01`;
  const until = `${m}-${String(lastDay(m)).padStart(2, '0')}`;
  const path = branch ? `commits/${branch}` : 'commits';
  const url = `https://github.com/${repo}/${path}?since=${since}&until=${until}`;
  try {
    const r = await fetch(url, {
      headers: { 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36', 'accept': 'text/html' },
      redirect: 'follow',
    });
    const t = await r.text();
    if (r.status !== 200) { result.errors.push(`${m}:status=${r.status}`); result.months[m] = null; await delay(1200); continue; }
    const set = new Set([...t.matchAll(/committedDate":"(20\d\d-\d\d-\d\d)/g)].map(x => x[1]));
    const inMonth = [...set].filter(d => d.startsWith(m));
    // no-commits 页面会给出提示；有日期则为在月内提交（页面最多列出约 35 条）
    result.months[m] = inMonth.length > 0 ? inMonth.length : (set.size === 0 ? 0 : 0);
    if (set.size > 0 && inMonth.length === 0) result.errors.push(`${m}:dates_out_of_range(${[...set].slice(0, 3).join(',')})`);
  } catch (e) { result.errors.push(`${m}:${e.message}`); result.months[m] = null; }
  await delay(900);
}
result.present = MONTHS.filter(m => result.months[m] > 0);
result.missing = MONTHS.filter(m => !(result.months[m] > 0));
result.all_months = result.missing.length === 0;
console.log(JSON.stringify(result));
