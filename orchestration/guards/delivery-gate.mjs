// delivery-gate：交付协议格式强制 + 组员自维护层 harness 预算硬闸（README §3.4 / §9；basic.md §1 三档裁决）。
// Stop hook：① 最终回复缺 <result>/<blocked> → exit 2 退回改正；② pod 的 AGENTS.md / 自建 rules /
// 自建 skills 超预算或违规 always_on → exit 2 + 精确整改指令。两类问题一次全报（合并成一轮整改）。
//
// 防循环：stop_hook_active 为真时放行（避免死循环）。
// fail-open：stdin 解析失败、或预算段自身异常（缺目录、坏 frontmatter、编码意外）一律放行——
// 门禁故障不阻塞生产，也不把「拒绝清理」伪装成「保护」。**但 fail-open 必须留痕**：任何
// 吞异常处都在 stderr 打 `[delivery-gate fail-open] <原因>` 再 exit 0，让上层（doctor 的
// 活体探针 / orch runner 的告警扫描）能机械识别「本该拦但放行了」的静默退化；否则 guard
// 有 bug 时会伪装成一切正常（backlog 20261010-193601-devops 的教训）。
//
// 计量口径：fs.statSync().size = 落盘字节（Windows 检出为 CRLF），与 `wc -c` /
// `find -size +8192c` 同口径，即 harness 实际注入所付的税。
//
// 环境变量（由 orch dispatch 的 runner.build_env 注入，与 pod-guard 同一套）：
//   PYSCI_POD 本 pod 绝对路径；PYSCI_DEPLOYED_SKILLS 部署技能目录名清单（分号分隔）。
//   PYSCI_DEPLOYED_SKILLS 缺失时**跳过自建技能那一支**：此时无法区分自建与部署副本，
//   而部署副本的尺寸/描述是 devops 的责任，不该由组员的 Stop 买单。

import fs from 'node:fs';
import path from 'node:path';

const LIMIT = 8192;
const CHARTER = 'charter.md'; // 只读层，always_on 槽位专属；尺寸由 pysci-dev doctor 审计
const TAG_RE = /<result>[\s\S]*<\/result>|<blocked>[\s\S]*<\/blocked>/;

let raw = '';
process.stdin.setEncoding('utf8');
for await (const chunk of process.stdin) raw += chunk;

// 执行流见文件末尾的 run()：ESM 顶层的 `const` 有 TDZ，若在声明之前调用 run()，
// 其中的 size/mdIn/subDirs 都还是未初始化绑定（fail-open 会把这种自伤静默吞掉）。
// fail-open 留痕：任何吞异常处调用此 helper，stderr 打一行标签让上层能机械识别。
function failOpen(where, err) {
  const msg = err && (err.stack || err.message) ? err.stack || err.message : String(err);
  console.error(`[delivery-gate fail-open] ${where} 异常，本次放行：${msg}`);
}

function run() {
  let data = {};
  try {
    data = JSON.parse(raw);
  } catch (e) {
    failOpen('stdin JSON 解析', e);
    process.exit(0);
  }
  if (data.stop_hook_active) process.exit(0);

  const problems = [];

  const msg = data.last_assistant_message || '';
  if (msg && !TAG_RE.test(msg)) {
    problems.push(
      '交付格式不合规：最终回复必须含且仅含一个 <result>...</result>（成功）或 <blocked>...</blocked>（失败）标签块；' +
        '可选 <infra_suggestion>...</infra_suggestion>；产物用 <artifact check="类型|none" reason="...">路径</artifact> 声明。' +
        '请按 charter 交付协议重新组织最终回复（工作内容不必重做）。'
    );
  }

  try {
    budgetProblems(
      path.resolve(process.env.PYSCI_POD || data.cwd || process.cwd()),
      problems
    );
  } catch (e) {
    /* fail-open：预算校验自身异常不得卡死交付；但必须留痕（见文件头说明） */
    failOpen('预算校验段', e);
  }

  if (!problems.length) process.exit(0);

  console.error(
    `delivery-gate：本次交付存在以下硬闸问题（harness 预算三层制之组员侧，上限均为 ${LIMIT}B 落盘字节）：`
  );
  for (const p of problems) console.error(`  · ${p}`);
  console.error('请逐条整改后重新结束；工作内容不必重做。');
  process.exit(2);
}

// ---------------------------------------------------------------------------
// 预算校验（只读 pod 自维护层：AGENTS.md 与 .qoder/ 下非只读层的 rules/skills）
// ---------------------------------------------------------------------------
function budgetProblems(pod, out) {
  // ① 全量注入档：AGENTS.md 每文件 ≤8192B
  const agents = path.join(pod, 'AGENTS.md');
  if (fs.existsSync(agents)) {
    over(
      agents,
      'AGENTS.md',
      size(agents),
      '全量注入档（每跳全文进上下文）',
      '精简——只留稳定事实与约定，操作细节移到自建技能或条件式 rules（按需档）',
      out
    );
  }

  const deployed = (process.env.PYSCI_DEPLOYED_SKILLS || '')
    .split(';')
    .map(s => s.trim().toLowerCase())
    .filter(Boolean);

  // ②③ 自建 rules：禁 always_on；description 行合计 ≤8192B（常驻暴露档）
  let ruleDesc = 0;
  for (const p of mdIn(path.join(pod, '.qoder', 'rules'))) {
    if (path.basename(p) === CHARTER) continue; // 只读层不在组员侧执法范围
    const fm = frontmatter(p);
    if (fmTriggerIsAlwaysOn(fm)) {
      out.push(
        `${p}：自建 rules 使用了 trigger: always_on —— 该槽位只留给 charter（只读层）。` +
          `动作：改为条件式（model_decision / glob / manual），并把「何时该读」写进 description。`
      );
    }
    ruleDesc += descBytes(fm);
  }
  if (ruleDesc > LIMIT) {
    out.push(
      `自建 rules 的 description 合计实测 ${ruleDesc}B > 常驻暴露档 ${LIMIT}B。` +
        `动作：逐条压缩 description（一句话讲清触发场景即可），或合并/删除不再需要的自建规则。`
    );
  }

  // ③④ 自建 skills：每个 SKILL.md ≤8192B；description 行合计 ≤8192B
  if (!deployed.length) return; // 无法区分自建与部署副本 → 该支不执法（见文件头说明）
  let skillDesc = 0;
  const own = [];
  for (const dir of subDirs(path.join(pod, '.qoder', 'skills'))) {
    if (deployed.includes(dir.toLowerCase())) continue;
    const skillMd = path.join(path.join(pod, '.qoder', 'skills', dir), 'SKILL.md');
    if (fs.existsSync(skillMd)) own.push({ dir, skillMd });
  }
  for (const { dir, skillMd } of own) {
    over(
      skillMd,
      `自建技能 ${dir}/SKILL.md`,
      size(skillMd),
      'SKILL.md 预算',
      '把细节移到同目录 references/*.md（按需档，单文件同样 ≤8192B），SKILL.md 只留决策与调用面 + 一行「遇到 X 时读 references/x.md」索引',
      out
    );
    skillDesc += descBytes(frontmatter(skillMd));
  }
  if (skillDesc > LIMIT) {
    out.push(
      `自建 skills 的 description 合计实测 ${skillDesc}B > 常驻暴露档 ${LIMIT}B。` +
        `动作：逐条压缩 description（一行内说清「何时用」），或合并同主题的自建技能。`
    );
  }
}

function over(file, label, bytes, tier, action, out) {
  if (bytes <= LIMIT) return;
  out.push(
    `${file}：${label} 实测 ${bytes}B > ${tier}上限 ${LIMIT}B。动作：${action}。`
  );
}

const size = p => fs.statSync(p).size;

const mdIn = dir =>
  fs.existsSync(dir) && fs.statSync(dir).isDirectory()
    ? fs
        .readdirSync(dir)
        .filter(n => n.toLowerCase().endsWith('.md'))
        .map(n => path.join(dir, n))
    : [];

const subDirs = dir =>
  fs.existsSync(dir) && fs.statSync(dir).isDirectory()
    ? fs
        .readdirSync(dir, { withFileTypes: true })
        .filter(e => e.isDirectory())
        .map(e => e.name)
    : [];

/** 取文件首个 `---` 围栏内的 frontmatter 行（无围栏 / 未闭合 → null，不抛）。 */
function frontmatter(file) {
  const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/);
  if (!lines.length || lines[0].trim() !== '---') return null;
  const fm = [];
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === '---') return fm;
    fm.push(lines[i]);
  }
  return null; // 未闭合：当作没有 frontmatter
}

function fmValue(fm, key) {
  if (!fm) return '';
  const p = key + ':';
  const ln = fm.find(l => l.trim().startsWith(p));
  return ln ? ln.trim().slice(p.length).trim().replace(/^["'](.*)["']$/, '$1') : '';
}

/** 大小写不敏感、容忍 `always-on` 写法；只认顶层 trigger 键。 */
function fmTriggerIsAlwaysOn(fm) {
  return /^always[-_]on$/i.test(fmValue(fm, 'trigger'));
}

/** description 行（含 `description: ` 前缀）的落盘字节数，与 pysci-dev doctor 同口径；
 *  折行块标量（下一行有缩进）一并计入。 */
function descBytes(fm) {
  if (!fm) return 0;
  const i = fm.findIndex(l => l.trim().startsWith('description:'));
  if (i < 0) return 0;
  let total = Buffer.byteLength(fm[i], 'utf8');
  for (let j = i + 1; j < fm.length; j++) {
    if (!fm[j].trim() || !/^\s/.test(fm[j])) break;
    total += Buffer.byteLength(fm[j], 'utf8') + 1;
  }
  return total;
}

try {
  run();
} catch (e) {
  // 兜底：任何未捕获异常（含未来重构把执行流放回 const 声明前的 TDZ 事故）都必须留痕后放行，
  // 不得静默——否则 guard 退化成一个空壳、观测面看不到它的失效（backlog 20261010-193601-devops）。
  failOpen('run() 未捕获异常', e);
  process.exit(0);
}
