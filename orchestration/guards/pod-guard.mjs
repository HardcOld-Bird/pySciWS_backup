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
// fail-open：输入解析失败或门禁自身异常时放行——门禁故障不阻塞生产；**但必须留痕**：
// 所有吞异常处 stderr 打 `[pod-guard fail-open] <原因>` 再 exit 0，让上层（doctor 活体
// 探针 / runner 告警扫描）能机械识别「本该拦但放行了」的静默退化（backlog
// 20261010-193601-devops）。

import path from 'node:path';
import os from 'node:os';

function failOpen(where, err) {
  const msg = err && (err.stack || err.message) ? err.stack || err.message : String(err);
  console.error(`[pod-guard fail-open] ${where} 异常，本次放行：${msg}`);
}

let raw = '';
process.stdin.setEncoding('utf8');
for await (const chunk of process.stdin) raw += chunk;

let data = {};
try {
  data = JSON.parse(raw);
} catch (e) {
  failOpen('stdin JSON 解析', e);
  process.exit(0);
}

try {
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

  // 白名单顺序（backlog 20261010-193601-devops 调整）：pod 内**先于**系统 tmp 判定，
  // 这样 fixture 或探针即便把 pod 建在 Temp 下（pytest tmp_path 的常规形态），只读层
  // 与部署副本仍被守；tmp 兜底只放行「pod 之外的 Temp 路径」（无 pod 归属的 scratch 写）。
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

  // 系统临时目录放行（仅对 pod 外的 scratch 生效）
  const tmpL = lower(os.tmpdir());
  if (absL.startsWith(tmpL)) process.exit(0);

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
} catch (e) {
  // 兜底：路径/环境变量解析等意外异常不得静默——留痕后放行，让上层观测得到（见文件头说明）。
  failOpen('主逻辑', e);
  process.exit(0);
}
