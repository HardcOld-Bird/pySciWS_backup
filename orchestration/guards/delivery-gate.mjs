// delivery-gate：交付协议格式强制（README §3.4 / §9）。
// Stop hook：成员收尾时校验最终回复——缺 <result>/<blocked> 标签块则 exit 2 退回改正。
// 防循环：stop_hook_active 为真时放行（避免死循环）；字段缺失 fail-open。

let raw = '';
process.stdin.setEncoding('utf8');
for await (const chunk of process.stdin) raw += chunk;

let data = {};
try { data = JSON.parse(raw); } catch { process.exit(0); }

if (data.stop_hook_active) process.exit(0);

const msg = data.last_assistant_message || '';
if (!msg) process.exit(0); // 字段缺失 fail-open

if (/<result>[\s\S]*<\/result>|<blocked>[\s\S]*<\/blocked>/.test(msg)) process.exit(0);

console.error(
  '交付格式不合规：最终回复必须含且仅含一个 <result>...</result>（成功）或 <blocked>...</blocked>（失败）标签块；' +
  '可选 <infra_suggestion>...</infra_suggestion>；产物用 <artifact check="类型|none" reason="...">路径</artifact> 声明。' +
  '请按 charter 交付协议重新组织最终回复（工作内容不必重做），然后结束。'
);
process.exit(2);
