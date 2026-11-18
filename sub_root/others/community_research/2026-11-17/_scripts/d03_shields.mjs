// node d03_shields.mjs <owner/repo> ... — shields.io stars + last-commit 核验（domain-03）
const repos = process.argv.slice(2);
const delay = (ms) => new Promise(r => setTimeout(r, ms));
const H = { 'user-agent': 'Mozilla/5.0', 'accept': 'application/json' };
for (const repo of repos) {
  const out = { repo };
  for (const kind of ['stars', 'last-commit']) {
    try {
      const r = await fetch(`https://img.shields.io/github/${kind}/${repo}.json`, { headers: H });
      const j = await r.json();
      out[kind] = j.value;
    } catch (e) { out[kind] = 'FAIL ' + e.message; }
    await delay(300);
  }
  console.log(JSON.stringify(out));
  await delay(300);
}
