// node ghprobe.mjs <owner/repo> — 依次尝试 GitHub API、ecosyste.ms、HTML 解析
const repo = process.argv[2];
if (!repo) { console.error('usage: node ghprobe.mjs <owner/repo>'); process.exit(2); }
const UA = { 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'accept': 'application/vnd.github+json,application/json' };

async function jget(url, hdrs = {}) {
  const r = await fetch(url, { headers: { ...UA, ...hdrs }, redirect: 'follow' });
  const t = await r.text();
  let body = null; try { body = JSON.parse(t); } catch { }
  return { status: r.status, body, raw: t };
}

// 1) GitHub REST API
try {
  const r = await jget(`https://api.github.com/repos/${repo}`);
  console.log(`# api.github.com -> ${r.status}`);
  if (r.status === 200 && r.body) {
    const b = r.body;
    console.log(JSON.stringify({
      stars: b.stargazers_count, forks: b.forks_count, open_issues: b.open_issues_count,
      license: b.license && b.license.spdx_id, default_branch: b.default_branch,
      pushed_at: b.pushed_at, updated_at: b.updated_at, archived: b.archived,
      created_at: b.created_at, subscribers: b.subscribers_count,
    }));
  } else { console.log('# api body: ' + r.raw.slice(0, 300)); }
} catch (e) { console.log('# api FAIL: ' + (e.cause ? String(e.cause) : e.message)); }

// 2) ecosyste.ms
try {
  const r = await jget(`https://repos.ecosyste.ms/api/v1/hosts/GitHub/repositories/${encodeURIComponent(repo)}`);
  console.log(`# ecosyste.ms -> ${r.status}`);
  if (r.status === 200 && r.body) {
    const b = r.body;
    console.log(JSON.stringify({
      stars: b.star_count, forks: b.forks_count, license: b.license,
      last_synced_at: b.last_synced_at, updated_at: b.updated_at,
      default_branch: b.default_branch, language: b.language, status: b.status,
      archived: b.archived,
    }));
  } else { console.log('# eco body: ' + r.raw.slice(0, 300)); }
} catch (e) { console.log('# eco FAIL: ' + (e.cause ? String(e.cause) : e.message)); }
