// node run_meta.mjs — 批量采集 ecosyste.ms 元数据 + releases（不占 GitHub API 配额）
// 用法: node run_meta.mjs repo1 repo2 ...   输出 JSON 行
const repos = process.argv.slice(2);
const delay = (ms) => new Promise(r => setTimeout(r, ms));
const H = { 'user-agent': 'pysci-research/1.0 (community survey)', 'accept': 'application/json' };

async function jget(url) {
  const r = await fetch(url, { headers: H, redirect: 'follow' });
  const t = await r.text();
  try { return { status: r.status, body: JSON.parse(t) }; } catch { return { status: r.status, body: null, raw: t.slice(0, 200) }; }
}

for (const repo of repos) {
  const out = { repo };
  try {
    const enc = encodeURIComponent(repo);
    const m = await jget(`https://repos.ecosyste.ms/api/v1/hosts/GitHub/repositories/${enc}`);
    if (m.status === 200 && m.body) {
      const b = m.body;
      out.stars = b.stargazers_count; out.license = b.license; out.pushed_at = b.pushed_at;
      out.archived = b.archived; out.default_branch = b.default_branch; out.language = b.language;
      out.last_synced = b.last_synced_at; out.desc = (b.description || '').slice(0, 90);
      out.created = b.created_at; out.forks = b.forks_count;
    } else { out.meta_err = `status=${m.status}`; }
    await delay(400);
    const rel = await jget(`https://repos.ecosyste.ms/api/v1/hosts/GitHub/repositories/${enc}/releases`);
    if (rel.status === 200 && Array.isArray(rel.body)) {
      const rs = rel.body.slice(0, 3).map(x => ({ tag: x.tag_name, name: (x.name || '').slice(0, 50), published: x.published_at }));
      out.releases = rs;
    } else { out.rel_err = `status=${rel.status}`; }
  } catch (e) { out.err = e.message; }
  console.log(JSON.stringify(out));
  await delay(600);
}
