// delivery-gate：交付协议格式强制 + 失败标签词表校验 + 组员自维护层 harness 预算硬闸
//（README §3.4 / §9；basic.md §1 三档裁决；README §5.4 tags）。
// Stop hook：① 最终回复缺 <result>/<blocked> → exit 2 退回改正；② <infra_suggestion> 的
// kind、<blocked> 的 tags 若存在必须落在受控词表内 → exit 2 + 精确整改指令；③ pod 的
// AGENTS.md / 自建 rules / 自建 skills 超预算或违规 always_on → exit 2 + 精确整改指令。
// 多类问题一次全报（合并成一轮整改）。
//
// 防循环：stop_hook_active 为真时放行（避免死循环）。
// fail-open：stdin 解析失败、或预算段自身异常（缺目录、坏 frontmatter、编码意外）一律放行——
// 门禁故障不阻塞生产，也不把「拒绝清理」伪装成「保护」。**但 fail-open 必须留痕**：任何
// 吞异常处都在 stderr 打 `[delivery-gate fail-open] <原因>` 再 exit 0，让上层（doctor 的
// 活体探针 / orch runner 的告警扫描）能机械识别「本该拦但放行了」的静默退化；否则 guard
// 有 bug 时会伪装成一切正常（backlog 20261010-193601-devops 的教训）。
//
// 计量口径：CRLF 等价字节（裸 LF 各补一个 \r）——与 `wc -c`（CRLF 文件）/
// `find -size +8192c` 同口径，即 harness 实际注入所付的税；LF 检出（worktree 里脚本
// 写出的文件）也预演出 main 检出后的字节，避免「本地绿、合并后红」（backlog 20261017-125557）。
//
// 环境变量（由 orch dispatch 的 runner.build_env 注入，与 pod-guard 同一套）：
//   PYSCI_POD 本 pod 绝对路径；PYSCI_DEPLOYED_SKILLS 部署技能目录名清单（分号分隔）。
//   PYSCI_DEPLOYED_SKILLS 缺失时**跳过自建技能那一支**：此时无法区分自建与部署副本，
//   而部署副本的尺寸/描述是 devops 的责任，不该由组员的 Stop 买单。

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const LIMIT = 8192;
const CHARTER = 'charter.md'; // 只读层，always_on 槽位专属；尺寸由 pysci-dev doctor 审计
// 交付块允许带属性（E1 的 `<blocked tags="…">`），写死字面 `<blocked>` 会把**带属性的合法
// 交付**判成「缺交付块」，把组员逼成不用标签——故属性面必须匹配属性，但不能用裸 `[^>]*`：
// 属性**值里**的 `>`（`reason="优于 > 阈值"`）会让它在引号内部提前收尾，tags/kind 被截在
// 属性面之外 → gate 看不见标签、跳过词表校验，而 Python 侧读得到（同一交付两侧读数分裂）。
// 属性面必须**引号感知**（与 Python 侧 delivery._ATTRS 同形，backlog 20261012-025325）：
// 先吃成对引号段（值内 `>`/`</` 一律放行），再吃不参与引号与 `>` 的普通字符。引号段禁跨行，
// 避免未闭合引号吞掉整段交付。
const ATTRS = String.raw`(?:"[^"\n]*"|'[^'\n]*'|[^>"'])*`;
const TAG_RE = new RegExp(
  `<result\\b${ATTRS}>[\\s\\S]*</result>|<blocked\\b${ATTRS}>[\\s\\S]*</blocked>`,
);
// E0-B（README §6 判据纪律）：改进提案必须结构化分类。infra_suggestion 块非空时
// kind 属性必填且须落在词表内——这是后续判决包分流（E5）的地基。
// 词表**只有一份**：state/failure-tags.json 的 infra_kinds 段（Python 侧
// failure_tags.load_infra_kinds() 读同一段）。下面的清单仅是文件读不到时的兜底，
// 与文件键集的一致性由 tests/.../test_infra_kind_vocab.py 钉住。
// 执法面必须覆盖**每一个**块（旧写法 .match 只看第一块）：组员在同一条交付里连发两块
// （110101 那跳：占位块在前、正式块在后），gate 只查第一块、解析器也只取第一块，
// 于是非法 kind 的块被落盘、合法块反而没入库（backlog 20261012-120001-leader）。
const INFRA_KINDS_DEFAULT = ['defect', 'usability'];
// 块体**不得越过另一个开口标签**（backlog 20261017-053820-devops）：交付正文引用/讨论标签
// 本身时，先出现的一个**裸开口标签**（无属性）曾与后文完整块的闭标签**跨段落配对**成一个
// 假块（属性面＝空）→ 误判「缺 kind」退回。现在块体遇到下一个 `<infra_suggestion` 即止，
// 于是裸引用不参与配对；而**整块自身**（裸开＋同处闭）仍照常被执法（缺 kind 依旧 exit 2）。
const INFRA_RE = new RegExp(
  `<infra_suggestion\\b(${ATTRS})>` +
    `((?:(?!<infra_suggestion\\b)[\\s\\S])*?)</infra_suggestion>`,
  'g',
);
// E1（README §5.4）：`<blocked tags="a,b">` 的失败标签受控词表。词表**只有一份**，
// 落在 state/failure-tags.json（Python 侧 failure_tags.load_tags() 读同一文件）——
// 这里不复制清单，否则两侧口径必然漂移（漂移的后果是组员被拦在两边不同的词表下）。
const BLOCKED_RE = new RegExp(`<blocked\\b(${ATTRS})>([\\s\\S]*?)</blocked>`);
const VOCAB_REL = ['orchestration', 'state', 'failure-tags.json'];
// <blocked reason="…"> 的成因档位（backlog 20261013-191213-devops）：单值受控词表，落在
// state/failure-tags.json 的 reasons 段（Python 侧 failure_tags.load_reasons() 读同一段）。
// 与 infra_kinds 同策略：文件读不到时回落下面这份兜底清单（而非放行）——分类只有几个稳定
// 值，失去执法才是要防的病灶。与文件键集的一致性由 tests/.../test_failure_reason.py 钉住。
const REASONS_DEFAULT = [
  'needs_decision',
  'capacity',
  'blocked_by_env',
  'blocked_by_upstream',
];

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
        '可选 <infra_suggestion kind="defect|usability">...</infra_suggestion>（非空时 kind 必填且须在词表内）；产物用 <artifact check="类型|none" reason="...">路径</artifact> 声明。' +
        '请按 charter 交付协议重新组织最终回复（工作内容不必重做）。'
    );
  }

  // E0-B（README §6 判据纪律）：非空 <infra_suggestion> 必须带合法 kind 属性——
  // 结构化分类是后续判决包分流（E5）的地基。块为空则不执法（无建议可分类）。
  // 逐块校验：解析器（delivery._INFRA_RE）只取第一块，所以「第一块合法、后块违规」的
  // 交付同样会让违规内容在队列里隐形——这里把每一块都过一遍词表。
  const infraBlocks = [...msg.matchAll(INFRA_RE)].filter((b) => b[2].trim());
  if (infraBlocks.length > 1) {
    problems.push(
      `交付含 ${infraBlocks.length} 个非空 <infra_suggestion> 块，而解析器只取第一块（其余会被静默丢弃）。` +
        '动作：把各块内容合并成**一个** <infra_suggestion kind="…"> 块。'
    );
  }
  if (infraBlocks.length) {
    const vocab = readInfraKinds();
    for (const b of infraBlocks) {
      const kindM = (b[1] || '').match(/\bkind\s*=\s*"([^"]*)"/);
      const kind = kindM ? kindM[1] : '';
      if (!vocab.includes(kind)) {
        // 回显被配对的那段原文：让成员一眼分辨「是正文里的引用被误配」而不是「我的提案缺属性」
        // （backlog 20261017-053820-devops）。
        const excerpt = (b[2] || '').replace(/\s+/g, ' ').trim().slice(0, 80);
        problems.push(
          `<infra_suggestion> 必须带 kind 属性且取值 ${vocab.join(' 或 ')}（当前` +
            (kindM ? ` kind="${kind}"` : '缺 kind') +
            '）。动作：<infra_suggestion kind="defect">（缺陷，附复现证据）' +
            '或 kind="usability"（难用/体验类）；块正文含 kind: 与 success_criteria 标记行供 approve 解析。' +
            (excerpt
              ? `被配对的块开头：「${excerpt}${excerpt.length >= 80 ? '…' : ''}」——若这是正文里`
                + '对标签的**引用**而非你的提案，请把它转义成 `&lt;infra_suggestion&gt;` 后重发。'
              : '')
        );
      }
    }
  }

  // E1（README §5.4）：<blocked tags="a,b"> 的失败标签必须落在受控词表内——同一条
  // 「分类必须结构化」的纪律（上面 kind 那条）之失败侧：没有标签，§2.4 的滑窗统计
  // 就没有原料，深层结构问题永远藏在单条 blocked 里。缺省不标注**合法**（空列表），
  // 只有标了才校验标得对不对——否则组员会被「不许沉默」和「不许标错」夹住。
  const blockedMatch = msg.match(BLOCKED_RE);
  if (blockedMatch) {
    const tagsM = (blockedMatch[1] || '').match(/\btags\s*=\s*"([^"]*)"/);
    const tags = tagsM ? splitTags(tagsM[1]) : [];
    if (tags.length) {
      const vocab = readVocab();
      if (vocab === null) {
        /* fail-open：词表读不到时无法校验，只能放行（留痕见 failOpen） */
      } else {
        const bad = tags.filter((t) => !vocab.includes(t));
        if (bad.length) {
          problems.push(
            `<blocked tags> 含词表外标签 ${bad.map((b) => `"${b}"`).join(', ')}（当前 tags="${tagsM[1]}"）。` +
              `动作：改用下列受控标签，或删掉 tags 属性（不标注合法）；词表在 ` +
              `orchestration/state/failure-tags.json，增补须附定义并经 devops 交付：\n` +
              `    ${vocab.join(', ')}`
          );
        }
      }
    }
  }

  // <blocked reason="…"> 的成因档位（backlog 20261013-191213-devops）：单值受控词表，同
  // 「分类必须结构化」纪律之失败侧——缺省不标注合法，只有标了才校验标得对不对。它与 tags
  // 关注面不同：tags 说「失败是什么类」，reason 说「该由谁、怎么处置」，故 drain/status
  // 据 reason 把 blocked **分列**（待裁决 vs 待排期）。
  if (blockedMatch) {
    const reasonM = (blockedMatch[1] || '').match(/\breason\s*=\s*"([^"]*)"/);
    const reason = (reasonM ? reasonM[1] : '').trim().toLowerCase();
    if (reason) {
      const vocab = readReasons();
      if (vocab !== null && !vocab.includes(reason)) {
        problems.push(
          `<blocked reason> 取值 "${reason}" 不在受控词表内。动作：改用下列档位之一，` +
            `或删掉 reason 属性（不标注合法）；词表在 orchestration/state/failure-tags.json 的 ` +
            `reasons 段：\n    ${vocab.join(', ')}`
        );
      }
    }
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

// ---------------------------------------------------------------------------
// 失败标签词表（E1，README §5.4）
// ---------------------------------------------------------------------------

/** tags 属性值 → 标签数组：分隔符宽容（半/全角逗号、分号、空白）、小写化、去重。
 *  与 Python 侧 failure_tags.parse_tags() 同语义——两侧不一致会让组员看到
 *  「gate 放行但统计里查无此标签」的分裂行为。 */
function splitTags(raw) {
  const out = [];
  for (const part of (raw || '').split(/[,，;；\s]+/)) {
    const t = part.trim().toLowerCase();
    if (t && !out.includes(t)) out.push(t);
  }
  return out;
}

/** 读词表文件原文；根路径优先取 runner.build_env 注入的 PYSCI_PROJECT_ROOT，缺席时按
 *  guard 自身位置回溯（orchestration/guards/ → 项目根），使 pod 外的手工调用也能校验。 */
function vocabData() {
  const root =
    process.env.PYSCI_PROJECT_ROOT ||
    path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
  const file = path.join(root, ...VOCAB_REL);
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

/** 读受控词表的标签键；读不到或形状不符返回 null（调用方放行 + 留痕）。 */
function readVocab() {
  try {
    const data = vocabData();
    const tags = Object.keys((data && data.tags) || {});
    return tags.length ? tags : null;
  } catch (e) {
    /* fail-open：词表缺失/损坏时**不**拦交付——门禁自身故障不得伪装成组员违规 */
    failOpen('失败标签词表读取', e);
    return null;
  }
}

// 提案分类词表：读不到文件时回落到 INFRA_KINDS_DEFAULT 而不是「不执法」——分类只有两
// 个值且协议文档写死，兜底清单比 tags 那份稳定得多；失去执法才是本单要修的病灶。
function readInfraKinds() {
  try {
    const keys = Object.keys((vocabData() || {}).infra_kinds || {});
    return keys.length ? keys : INFRA_KINDS_DEFAULT;
  } catch (e) {
    failOpen('提案分类词表读取（改用兜底清单）', e);
    return INFRA_KINDS_DEFAULT;
  }
}

// blocked 成因档位词表：口径同 readInfraKinds——读不到文件时回落兜底清单而不是「不执法」。
function readReasons() {
  try {
    const keys = Object.keys((vocabData() || {}).reasons || {});
    return keys.length ? keys : REASONS_DEFAULT;
  } catch (e) {
    failOpen('blocked 成因档位词表读取（改用兜底清单）', e);
    return REASONS_DEFAULT;
  }
}

function over(file, label, bytes, tier, action, out) {  if (bytes <= LIMIT) return;
  out.push(
    `${file}：${label} 实测 ${bytes}B > ${tier}上限 ${LIMIT}B。动作：${action}。`
  );
}

const size = p => {
  const raw = fs.readFileSync(p);
  let bareLf = 0;
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] === 0x0a && (i === 0 || raw[i - 1] !== 0x0d)) bareLf++;
  }
  return raw.length + bareLf;
};

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
