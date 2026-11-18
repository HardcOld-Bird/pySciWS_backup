// 通用抓取工具：node fetch.mjs <url> [maxChars] → 输出正文（UTF-8）
const url = process.argv[2];
const maxChars = parseInt(process.argv[3] || '200000', 10);
if (!url) { console.error('usage: node fetch.mjs <url> [maxChars]'); process.exit(2); }
try {
  const r = await fetch(url, {
    headers: {
      'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
      'accept': 'text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8',
      'accept-language': 'en-US,en;q=0.9',
    },
    redirect: 'follow',
  });
  const body = await r.text();
  console.log(`# status=${r.status} final=${r.url} len=${body.length}`);
  console.log(body.slice(0, maxChars));
} catch (e) {
  console.error('FETCH FAIL: ' + (e && e.cause ? String(e.cause) : e.message));
  process.exit(1);
}
