// pod-guard：组员 pod 的路径纪律硬防线（README §9）。
// PreToolUse matcher Write|Edit|NotebookEdit；exit 2 = 拦截，stderr 注入给成员。
//
// 白名单模型（README §2 矩阵「专职组员」列）：
//   pod 内：除只读层（PYSCI_READONLY）与部署技能（PYSCI_DEPLOYED_SKILLS）外全部放行
//           （bench/outbox/AGENTS.md/自建 rules/自建 skills = 自维护层）；
//   pod 外：仅任务白名单目录（PYSCI_TASK_DIRS，项目相对或绝对路径）与系统临时目录；
//   其余一律拦截。
// 环境变量由 orch dispatch 注入（runner.build_env）：
//   PYSCI_POD / PYSCI_PROJECT_ROOT / PYSCI_READONLY / PYSCI_DEPLOYED_SKILLS / PYSCI_TASK_DIRS
// 诚实条款：只拦 Write/Edit/NotebookEdit 工具，Bash 重定向不在射程（README §9）。
// fail-open：输入解析失败时放行（门禁自身故障不阻塞生产）。

import path from 'node:path';
import os from 'node:os';

let raw = '';
process.stdin.setEncoding('utf8');
for await (const chunk of process.stdin) raw += chunk;

let data = {};
try { data = JSON.parse(raw); } catch { process.exit(0); }

const fp = (data.tool_input && data.tool_input.file_path) || '';
if (!fp) process.exit(0);

const sep = p => p.replace(/\\/g, '/');
const lower = p => sep(p).toLowerCase();

const pod = process.env.PYSCI_POD || process.cwd();
const projectRoot = process.env.PYSCI_PROJECT_ROOT || '';
const listEnv = v => (v || '').split(';').filter(Boolean); // 固定分号分隔（Windows 路径含冒号）
const readonly = listEnv(process.env.PYSCI_READONLY);
const deployed = listEnv(process.env.PYSCI_DEPLOYED_SKILLS);
const taskDirs = listEnv(process.env.PYSCI_TASK_DIRS);

// 目标路径解析为绝对路径：相对路径按 pod 解析（成员视角的相对路径即 pod 相对；
// 真实 hook 运行 cwd 就是 pod，这里不依赖 process.cwd() 以便测试与异常场景稳健）
let abs = fp;
if (!path.isAbsolute(abs)) abs = path.resolve(pod, abs);
const absL = lower(abs);
const podL = lower(path.resolve(pod));

const block = reason => {
  console.error(`pod-guard: 拦截对 ${fp} 的写入——${reason}。任务白名单由任务书声明；如需扩大范围或修改只读层，请在交付中提出 <infra_suggestion>。`);
  process.exit(2);
};

// 系统临时目录放行
const tmpL = lower(os.tmpdir());
if (absL.startsWith(tmpL)) process.exit(0);

if (absL.startsWith(podL + '/') || absL === podL) {
  // pod 内：计算 pod 相对路径，先查只读层，再查部署技能
  const rel = sep(abs).slice(podL.length + 1);
  for (const r of readonly) {
    const rn = sep(r).replace(/^\/+/, '');
    if (rel === rn || rel.startsWith(rn + '/')) block(`命中只读层 ${rn}`);
  }
  for (const s of deployed) {
    const prefix = `.qoder/skills/${sep(s)}`;
    if (rel.startsWith(prefix + '/') || rel === prefix) block(`${s} 为 devops 部署的技能副本（可自建其他技能，勿改部署副本）`);
  }
  process.exit(0);
}

// pod 外：仅任务白名单目录
for (const d of taskDirs) {
  let dd = d;
  if (!path.isAbsolute(dd)) {
    dd = projectRoot ? path.resolve(projectRoot, dd) : path.resolve(process.cwd(), dd);
  }
  const ddL = lower(path.resolve(dd));
  if (absL.startsWith(ddL + '/') || absL === ddL) process.exit(0);
}

block('目标在 pod 与本任务白名单目录之外');
