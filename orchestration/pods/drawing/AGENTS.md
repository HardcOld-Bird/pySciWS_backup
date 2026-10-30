# drawing 组员长期记忆

> 本文件由 drawing 组员自维护（每次会话启动自动加载）。只记「下次会话还要知道的
> 事实与约定」：生成图配方/审美范本、各研究线视觉风格、LEDGER 入库踩过的坑。保持精简；
> 主题多了以后可拆分到 `.qoder/rules/` 自建规则（model_decision/glob 触发按需加载）。

## 身份

- pySci 编排体系 drawing 组员（AI 绘图专员），charter 见 `.qoder/rules/charter.md`。
- 工作区：`bench/` 草稿、`outbox/` 交付副本、正式产物落任务书白名单目录。
- 专业知识：部署技能 ai-drawing（Skill 工具调用）；CLI `pysci-imagine`。

## 经验

### pysci-imagine 踩坑（2026-10-15 实测，代价：2 张已计费图被误删）

1. **`adjust` 的 src 是位置参数，`palette` 的 src 是 `--src`**（同一 CLI 内不一致，易错）。
   正确：`pysci-imagine adjust --out x.png src.jpg`；`pysci-imagine palette --src x.png`。
2. **禁止把 CLI 串进 `| tail` / `| head` 再用 `&&` 串联**：管道退出码取末段命令（tail=0），
   会吞掉 CLI 的失败退出码，导致后续 `rm` 等无条件执行。**要串联就裸跑、看 `echo rc=$?`**。
3. **flash 档只出 jpeg**（`--output-format png` 仅 5.0 主档支持）；任务书常要求 `.png` →
   出图后用 `adjust` 转格式即可（免费）。
4. **`runs/*.json` 快照里 `b64_json` 落盘时已被剥离**（占位 `<... chars, stripped>`）→
   **误删/丢失的生成图无法从快照恢复**。故：**Read 满意后立刻 `gallery --add` 归档**，
   不要等到整理阶段；gallery/ 才是唯一可靠留存。
5. **Tier 0 内置 ImageGen 当前不可用**（`invalid_model_error`，2026-10-15 复现 2 次）；
   方舟无 key/超额时无免费兜底可用。
6. **安全出图流水线（2026-10-15 补单验证通过）**：`gen`（flash 出 .jpg）→
   `adjust --out x.png x.jpg` → **立即** `gallery --add x.png --slug S` → `Read` 判读 →
   最后才 `rm` 临时 .jpg。串联时**绝不带 `| tail`/`| head`**，逐条裸跑看 `echo rc=$?`。
   补单说明：`runs/<slug>.json` 已存在时快照自动改名 `<slug>_1.json`，不影响。
7. **5.0 pro/flash 不支持 `output_format=png`**（`imagine models` 实测：仅已下架的
   `doubao-seedream-5-0-260128` 支持）→ **jpeg→`adjust` 转出的 png 不含 `pysci:*` 内嵌溯源**
   （`gen` 的 jpeg 本身也无）。charter 要求「PNG 内嵌溯源」时须手工补：
   `uv run --no-sync python -c` 读 `runs/<slug>.json` 的 `request.prompt/model`，用
   `PngInfo` 写 `pysci:prompt`/`pysci:model`/`pysci:backend`(=ark) 后 `im.save(..., pnginfo=...)`
   （纯本地免费，不计费）。
