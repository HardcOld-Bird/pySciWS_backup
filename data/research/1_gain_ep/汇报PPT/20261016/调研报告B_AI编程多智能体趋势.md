# 调研报告 B：AI 编程的多智能体范式趋势

> 为 PPT 页4「AI 编程 / 多智能体范式」供料 · 检索日期 2026-10-16 ·
> **【事实】**=有来源可核；**【推断】**=本报告判断。所有结论附来源，来源清单一。
> 原始提问真源：行业引领者投入方向 / Claude Code 与 Codex 功能演进 / copilot→agentic→多 agent 是否成立。

## 结论先行

1. **大厂主战场已从「模型」转到「Agent 运行时与编排」。【事实】**
   OpenAI 2026-09-10 公测 **Agents API**（把驱动 Codex 的 agent harness 托管化），
   2026-09-29 DevDay 又发 **Dots**（常驻个人智能体）、**ChatGPT Space**、**GPT-6.1 Sol**、
   Decisions API、Codex Cloud、Marketplace；Anthropic 2026-04 推 **Claude Managed Agents**
   （Session/Harness/Sandbox 三级抽象），10 月再加 **Dynamic Workflows**。
   两家不约而同在卖「执行层」，而非「更强的模型」。

2. **Claude Code 与 Codex 在 2026 年都把「多智能体」写进了产品面。【事实，版本级证据 §二】**
   Claude Code：Agent Teams 研究预览（2026-02）→ `/ultrareview` 并行多 agent 审查（04）→
   **dynamic workflows**（05，跨「数十到数百」agent）→ subagent forking 默认开（08）→
   企业级 self-hosted runner+orchestrator（10）。
   Codex：**multi-agent V2**、subagent、agent Command Center、远程 agent 留言板（10 月密集落地）。

3. **用户「从单 agent 转向多 agent 协同、并不遥远」的判断——成立，且已经兑现到「产品级可用」。【推断】**
   2026-10 的时点上，多 agent 协同**不再是论文/框架阶段**：主 Agent 编排、子 Agent 并行、
   互相通信、跨 agent 记忆，都已是可直接使用的产品能力（§三·甲层）。
   但需分三层看：**编排基础设施已落地 / 可靠性与经济性仍在早期**（§四）。

---

## 一、大厂在押什么（OpenAI / Anthropic）【事实】

### OpenAI：把 Codex 的 harness 变成平台

| 时间 | 事件 | 要点 |
|---|---|---|
| 2026-04-06 | ChatGPT Super App | ChatGPT + Codex CLI + Atlas 浏览器三合一（二手来源，谨慎采信） |
| **2026-09-10** | **Agents API 公测** | 托管式 **Codex harness**：持久会话、上下文自动压缩、tool search、programmatic tool calling、**多 Agent/子 Agent 并行**、MCP 为一等工具、托管/自有/合作方沙箱、vault 密钥托管、computer use；beta 期不额外收费 |
| **2026-09-29/30** | **DevDay 2026**（旧金山，20+ 项） | **Dots**（对标 Meta Muse 的常驻智能体，自带云端电脑与浏览器，跨 ChatGPT/邮件/Slack 持续干活）；**ChatGPT Space**（人 + ChatGPT + Codex + Dots 共享协作空间）；**GPT-6.1 Sol**（面向 agentic coding 的性价比模型，已成 Codex CLI 默认）；Decisions API、Marketplace、Codex Cloud；ChatGPT Pro 涨到 **$500/月**；周活 **12 亿** |

- 一句话：OpenAI 这半年在**把「运行时」产品化**——用户不必自维护 agent loop、上下文压缩、
  子 agent 并发与沙箱，改由平台托管。⚠️ openai.com 对我方网络返回 403（反爬），官方页**未直取**，
  采信 The Verge（英文一手报道）+ 国盛证券研报（转述官网）+ 多篇独立实测文互证（§七/§八）。

### Anthropic：把 Agent 做成「可运维的云服务」

- **2026-04-08 Claude Managed Agents 公测**：把运行时抽象为 **Session / Harness / Sandbox**
  三个稳定契约（检查点续跑、凭证管理、端到端追踪），首批接入 Notion/Asana/Sentry/Rakuten；
  2026-05-25 增自托管沙箱与 **MCP 隧道**。
- **2026-10-09/10 上线 Dynamic Workflows 公测**：主智能体**写一段编排程序**（拆阶段、定并行/
  依赖、按中间发现动态加任务），后台跑成一次 workflow run；官方限额**单次最多累计 1000 个
  agent、并发 64 线程、生命周期 24h**，另收 **$0.08/小时** + token。同日 **Claude Code Projects**
  向 Pro/Max 开放（项目级记忆 + 多线程协调）。
- 模型节奏：Opus 5.5（09-22）、Sonnet 5.5（09-28）、Haiku 5.5（10-07）。
- **共同信号**：两家都把「**编排 + 沙箱 + 记忆 + 可观测性**」当新战场——即用户所说的 "harnessing"。

---

## 二、Claude Code 与 Codex 的功能演进（版本级证据）【事实】

### Claude Code（来源：官方 `CHANGELOG.md`（GitHub）+ Releases API 日期）

| 版本 | 日期 | 与多智能体相关的更新（原文摘录） |
|---|---|---|
| 2.1.32 | 2026-02-05 | **Agent Teams 研究预览**（多智能体协作，需 `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`） |
| 2.1.111 | 2026-04-16 | `/ultrareview`：云端**并行多 agent** 分析与批判式代码审查 |
| **2.1.154** | **2026-05-28** | **dynamic workflows**：Claude 自行创建 workflow，**「在后台编排数十到数百个 agent」**，`/workflows` 查看 |
| 2.1.202 | 2026-07-06 | `/code-review <level>`：按 effort 的**多 agent** 审查 |
| 2.1.232 | 2026-08-13 | **subagent forking 默认开启**；非 teammate 的 agent spawn 默认转后台 |
| 2.1.29x–2.1.300 | 2026-10 | 企业**self-hosted runner + orchestrator**、cloud sessions；teammate 有 `agent_id`/`SendMessage` 邮箱、`TeammateIdle`/`SubagentStart` 钩子；agent teams 已产品化（teammate 默认沿用 leader 模型） |

> 读法：Claude Code 的路径是「**一个会话里的 subagent（2 月前）→ 平级 agent teams（2 月）→
> 后台动态工作流（5 月，数十到数百）→ 企业编排器（10 月）**」——多 agent 是逐版加码的主线。

### Codex（来源：官方仓库 `github.com/openai/codex` Releases）

| 版本 | 日期 | 多智能体相关（原文摘录） |
|---|---|---|
| v0.160.0 | 2026-10-01 | subagent 保留待启动的环境；**agent Command Center** 增加历史分页 |
| **v0.161.0** | **2026-10-07** | **GPT-6.1 Sol 成为默认模型**；**multi-agent V2**、多 agent 会话中的**远程 agent 留言板**、子 agent 被 Guardian 拦截时**通知父 agent**；multi-agent V2 上 Amazon Bedrock |
| v0.162.0 | 2026-10-08 | **托管 Git worktree**（并行隔离）；Command Center 任务置顶；V2 subagent 动态继承工具 |

- 更早铺垫：GPT-5.5 GA（2026-06-04）、《Codex-maxxing for long-running work》长任务实践
  （2026-06-22）、OSS/本地模型模式（2026-06）。
- 读法：Codex 的路径是「**长会话（6 月）→ 托管 harness 外露为 API（9 月）→ 多 agent V2 +
  Command Center 调度台（10 月）**」。

---

## 三、范式主线：copilot → agentic → 多 agent 协同【事实 + 推断】

**【事实】三级台阶的公认定义**
1. **Copilot**：人在循环里逐行/逐函数补全，AI 是「工具」；
2. **Agentic**：单个 Agent 在项目中自主多步执行（读改文件、跑命令、自纠错），人是验收者；
3. **多 Agent 协同 / 编排**：多个 Agent 分工、并行、互相通信，由 **lead agent 或一段编排程序**
   调度，产出合并。

**【事实】甲层——产品级支持（今天就能用，不是玩具）**
- **Anthropic**：Managed Agents **Dynamic Workflows**（单次至多 1000 agent / 并发 64）[2026-10]；
  Claude Code Agent Teams + 后台 dynamic workflows（数十到数百 agent）[2026-02→10]。
- **OpenAI**：Agents API **`multi_agent: {enabled: true, max_concurrent_subagents: 3}`**（配置即成）
  [2026-09]；Codex **multi-agent V2 + Command Center** [2026-10]。
- **Cursor**：**Projects**（2026-09-10 beta）——一个 **coordinator agent 不写代码**，把任务委派给
  **数千个子 agent** 在云端跑，跨月维护上下文、按事件（PR/CI/Slack）自触发；官方数据
  **新用户合并 PR 数 +30%、以 Projects 为主者合并量 6×**，并引用其 2026-02 提出的
  **「软件开发的第三个时代：agent 编队承接整块工作」**。
- **Google**：Antigravity（agent 为核心的开发环境）+ ADK/**A2A** 跨语言多 agent 演示（2026-06-22）、
  开源编排 runtime「ax」（2026-09）。
- **其他**：AWS **Bedrock Managed Agents**、n8n Agents、Amp（底层改 Claude Agent SDK 并保留
  thread 间消息与编排）。

**【事实】乙层——编排原语已标准化**
MCP（工具接入）、Git worktree（并行隔离，Claude Code/Codex/Cursor 都有）、
agent 间邮箱/消息板（Claude Code `SendMessage`、Codex 远程留言板）、
调度台（Codex Command Center）、子 agent 生命周期钩子。**这些正是"harnessing"的零件。**

**【事实】丙层——学术与量化锚点（关键，附反方）**
- **正面 · 研究任务**：Anthropic 工程博客（2025-06-13）lead 用 Opus 4 + 子 agent 用 Sonnet 4，
  内部评测**比单 agent 高 90.2%**，代价是**多 agent 约为普通对话 15× token**。
- **转折 · 编码任务**：同一篇 2025 博客明确写「**大多数编码任务的可并行度低于研究**，
  且 LLM 目前不擅实时协调/委派」。**但在 2026-10**，Anthropic 用 Dynamic Workflows 在
  **11.6 万行代码、预埋 70 个 bug** 的代码库上给出反转证据：**单 agent 三次分别查出 14–27 个
  （最好 38.6%），动态工作流三次**均**查出 66 个（94.3%）**。
  → **编码任务只要「可切分」，多 agent 就从"不适合"变成"显著更好"**，这是本报告最关键的事实。
- **反方 · 成本**：Anthropic 自称会「烧掉大量 token」；**一位资深 OpenAI 工程师公开称
  agent 集群是"巨大的 token 浪费"**；Cognition（2025）《Don't Build Multi-Agent Systems》主张
  慎用。动态工作流的官方测试**未披露模型、实际 agent 数、耗时、token 与误报**，**无第三方复现**。
- **反方 · harness**：Apple 论文 *How Much of a Harness Does a Strong Agent Need for Autonomous
  ML Engineering?*（arXiv:2609.40303，2026-09-30）在**等时间预算、同 backbone** 下发现：
  精巧的多 agent harness **相对「最小 harness 的单 agent」并无优势**，backbone 才是主因。
  → 但这限于 **ML 工程基准**，与"可切分的代码审查/大规模迁移"并不矛盾。

---

## 四、对用户判断的明确回应【推断】

> 用户原话：「当今 agentic 范式仍在发展甚至转向更高级范式——从单 agent 转向多 agent 协同
> （经 harnessing 完全可做到，并不遥远）。」

**判定：成立，且比"并不遥远"更进一步——产品级能力已经就位；真正未解决的是可靠性与经济性。**
分三层给结论：

| 层 | 状态（2026-10） | 依据 |
|---|---|---|
| **① 能力/产品层** | ✅ **已到位，今天就能用** | Claude Code Agent Teams/动态工作流、Codex multi-agent V2、OpenAI Agents API、Cursor Projects、Managed Agents |
| **② 编排原语层** | ✅ **已标准化** | MCP / worktree 隔离 / agent 间消息 / 调度台 / 邮箱 / 生命周期钩子；「harnessing」已是各家竞争焦点 |
| **③ 可靠性/经济性层** | ⚠️ **仍在早期** | token 成本 4–15×；只适合「可切分/广度优先」任务，强共享上下文的任务反而更差；Anthropic 自测无第三方复现；Apple 论文质疑 harness 增益 |

**三条可操作结论（给页4 的"观点句"）**
1. **范式确实转段了**：竞争焦点已从"模型多强"转向"**谁能把多 agent 编排做得可靠、可运维**"。
2. **门槛在"可切分性"**：代码审查、批量迁移、合同核对、深度研究这类天然可切分的任务，
   多 agent 收益明确（官方 94.3% 检出率）；紧密耦合的重构则可能被合并冲突抵消。
3. **"并不遥远"准确**——基础设施已到 (①/②)，**"便宜、可靠、可信"才是下一道坎 (③)**。

---

## 五、页4 取用建议（一页信息量）

1. **主叙事**：一条三级台阶箭头图 **Copilot → Agentic → Multi-Agent 编队**，每级配一个官方证据
   （补全 / 单 agent 自主 / Cursor coordinator→数千子 agent、Anthropic 单次≤1000 agent）。
2. **反转金句**：**「66 vs 27」**——11.6 万行代码埋 70 个 bug，单 agent 最好 38.6%，
   多 agent 动态工作流稳定 94.3%（Anthropic 自测，2026-10）。一个数字讲清"为何要多 agent"。
3. **平衡句**：配一句反方（Apple：最小 harness 未必输；token 15×）以显判断力，避免单向乐观。

---

## 六、截图清单（`screenshots/`，均取自官方域名，已实拍）

| 文件 | 出处 URL | 说明 |
|---|---|---|
| `B01_anthropic_multi_agent_research_system.png` | https://www.anthropic.com/engineering/multi-agent-research-system | **Anthropic 官方**多 agent 研究系统页（含 orchestrator-worker 架构，90.2% 与 15× 出处） |
| `B02_anthropic_news_index.png` | https://www.anthropic.com/news | **Anthropic 官方 Newsroom**（Opus/Sonnet/Haiku 5.5 发布节奏，佐证"大厂投入"） |
| `B03_anthropic_claude_code_page.png` | https://www.anthropic.com/claude-code | **Claude Code 官方产品页**（Projects/Scheduled/multi-day、多端入口） |

> 注：OpenAI 官网 `openai.com` 反爬 403，Codex changelog 页亦 403，**未取到 OpenAI 官方截图**；
> 本机代理 7890 当时不可用，未能绕过。

---

## 七、来源清单

**官方一手（可直接访问/已实取）**
- Claude Code `CHANGELOG.md`（GitHub，含 dynamic workflows / agent teams 原文）：
  https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md
- Claude Code Releases（日期戳）：`https://api.github.com/repos/anthropics/claude-code/releases`
- Codex Releases（multi-agent V2 / Command Center 原文）：
  https://github.com/openai/codex/releases
- Anthropic 工程博客《How we built our multi-agent research system》（2025-06-13，90.2% / 15×）：
  https://www.anthropic.com/engineering/multi-agent-research-system
- Anthropic Newsroom：https://www.anthropic.com/news
- **Cursor《Introducing Projects》（2026-09-10，coordinator→数千 subagent、6×）**：
  https://cursor.com/blog/projects
- arXiv:2609.40303（Apple，harness 对比，2026-09-30）／arXiv:2605.27922（Harness-Bench）

**官方页存在但被反爬/区域策略拦截（仅列 URL，未直取）**
- OpenAI Agents API：https://openai.com/index/introducing-the-agents-api/
- OpenAI DevDay 2026 recap：https://openai.com/index/devday-2026-recap
- Codex changelog：https://developers.openai.com/codex/changelog
- Anthropic docs（Managed Agents / Dynamic Workflows）：https://docs.anthropic.com/（→ 地区不可用）

**权威二手（用于互证上述官方页）**
- The Verge，《OpenAI DevDay 2026: The biggest news and announcements》（2026-10-01）
- 国盛证券《计算机行业周报：海外模型密集迭代，Agent 布局持续加速》（2026-10-13，转述 OpenAI 官网）
- 36Kr/IT之家/搜狐等对 Anthropic Dynamic Workflows 的报道（2026-10-10 前后，可交叉）

---

## 八、方法与局限

- 方法：WebSearch 多引擎 + 官方仓库 Releases/CHANGELOG（一手）＋ 官方博客直取 ＋
  Playwright（系统 Chrome 通道）对官方页实拍。
- **未直取**：OpenAI 官网、Anthropic docs（403 / 地区限制），故其细节依赖权威二手互证，
  已逐条标注；**GPT-6.1 Sol 价格、DevDay 20+ 项明细**等为二手转述，**置信度中等**。
- **未证实**：网传「Anthropic 内部管理约 3 万个活跃 Agent 并免费开放」——**素材极稀，本报告不采信**；
  「单次 1000 agent」是**累计调度规模**，非**同时** 1000（并发上限 64），勿混淆。
- Dynamic Workflows 的 94.3% 为 **Anthropic 自建单一任务评测，无第三方复现**，引用时应标"官方自测"。
