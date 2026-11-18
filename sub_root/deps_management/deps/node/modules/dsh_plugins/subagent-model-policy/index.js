/**
 * 子代理动态模型路由薄插件（profile 全局 Host 插件）。
 *
 * 职责只有两件：
 * 1. 按会话折叠宿主内置的授权事件 `subagent/model-selection-policy`，得到「这个会话
 *    能用哪些模型」（授权层，由 GUI 的 model-selection 设置持有，本插件只读）。
 * 2. 用 `ctx.tools.guard` 在每次 `subagent` / `subagent_fork` 派发前，把「此刻应当
 *    用哪些」算出来并拒绝越界路由；拒绝文本自带相位、允许集、默认值与档位选择规则
 *    （flash 优先，见 `TIER_GUIDANCE`），且列举时 flash 恒排在 pro/max 之前。
 *
 * 设计取舍见 `sub_root/others/plans/archived/subagent-model-policy.md`（§1 选型、
 * §4 决策表、§9 边界；评估口径背景见
 * `.agents/shared-skills/blueprint-canon/references/abstract-support-systems.md`）：
 * - guard 是同步硬拒、只拒不改；v1 不注入常驻上下文（零常驻 token 成本）。
 * - 旧会话（设置启用前创建、没有授权事件）fail-open 并记一次 warning，避免安装
 *   插件就打死存量会话；新会话出生即带授权，严格生效。
 * - 选路判定看「provider+model 成对声明」而非单个 provider 参数（v1.1；v1 按
 *   provider 单参判定，把 cordis_inspect_query（provider=Inspect Provider ID）
 *   与 list_subagent_models（provider/model=查询过滤器）误伤为选路工具，见
 *   `declaresRoutePair` 与 `READ_ONLY_DISCOVERY_TOOLS`）。
 * - `workflow` 的 `agent(...)` 与 `subagent_fork` 继承父路由，是已知边界（计划 §9）。
 * - link 安装下修改本文件**不会**热替换模块代：改完要重启 `dsh web` 才进进程。
 */
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { z } from 'zod'
import { createCalendar, loadHolidayData } from './calendar.js'
import { CONFIG_PATH_ENV, createConfigStore, resolveConfigCandidates, resolveConfigPath } from './config.js'
import { flashFirst, formatRoute, resolvePolicy, routeKey, sameRoute } from './policy.js'

/** 插件名（Cordis 行名）。 */
export const name = 'subagent-model-policy'

/** 依赖的宿主服务；缺任一个都不激活，而不是半残运行。 */
export const inject = ['tools', 'llm', 'sessionProjections']

/** 授权事件类型：宿主内置设置服务在 `agent/created` 时为每个会话盖章。 */
const POLICY_EVENT = 'subagent/model-selection-policy'

/** 继承父路由、不能选模型的委派工具（`tool-subagent` 的 fork 行用的默认名）。 */
const FORK_TOOL_NAMES = Object.freeze(new Set(['subagent_fork']))

/**
 * 只读发现类工具：参数里的 `provider`/`model` 是查询过滤器而非派发路由，guard 不看。
 *
 * 误伤教训（2026-11-16 实测）：`list_subagent_models` 曾被当作选路工具——无参（列
 * provider）与只给 provider（列模型）的查询在父路由≠默认时被拒，且无法查看本相位
 * 允许集之外的模型（想为谷段规划也不行）。发现 ≠ 派发，按稳定上游名豁免；新增此类
 * 工具时在此追加。
 */
const READ_ONLY_DISCOVERY_TOOLS = Object.freeze(new Set(['list_subagent_models']))

/**
 * 授权集的投影单元。
 *
 * 键、`stateVersion` 与形状刻意和宿主内置的 `subagent-model-selection-settings`
 * 完全一致：投影注册表按 key 共享同一个单元并计数，因此本插件与内置工具读到的是
 * 同一份状态，不会出现「guard 放行、原生随后拒绝」的错位。
 */
export const policyProjection = Object.freeze({
  key: 'subagentModelSelectionPolicy',
  stateVersion: 1,
  stateSchema: z
    .array(z.object({ provider: z.string().min(1), model: z.string().min(1) }).strict())
    .min(1)
    .nullable(),
  init: () => null,
  apply: (state, event) => {
    if (state !== null || event.type !== POLICY_EVENT) return state
    const allowedModels = event.data?.allowedModels
    if (!Array.isArray(allowedModels) || allowedModels.length === 0) return state
    return allowedModels.map((route) => ({ provider: route.provider, model: route.model }))
  },
})

/**
 * 判断一个工具的参数声明是否暴露一条完整的可选路由（`provider`+`model` 成对）。
 *
 * 成对口径（v1.1）与 `readExplicitRoute` 一致：`provider`/`model` 一起才构成一次
 * 路由选择。宿主 `subagent`（`modelSelection: true`）同时声明两者；而
 * `cordis_inspect_query` 的 `provider` 是 Cordis Inspect Provider ID（与 LLM 路由
 * 无关）且没有 `model`，按成对判定天然免疫这类语义碰撞（v1 的单参判定会把它误伤
 * 为选路工具并几乎永远拒绝）。
 *
 * 参数声明是 `defineTool` 编译后的对象根 JSON Schema，形如
 * `{ type: 'object', properties: { provider: …, model: … }, required: […] }`——参数名在
 * `properties` 下，**不在**顶层。也接受以属性表直接声明的形状，避免宿主换代后
 * 静默失效（曾在这里踩坑：按顶层键判断会让 `subagent` 永远被归为无关工具而 fail-open）。
 * @param {unknown} parameters - 工具的 `parameters` 声明（JSON Schema 或属性表）。
 * @returns {boolean} 是否声明了 `provider`+`model` 路由对。
 */
export function declaresRoutePair(parameters) {
  if (parameters === null || typeof parameters !== 'object') return false
  const properties = /** @type {{ properties?: unknown }} */ (parameters).properties
  const table =
    properties !== null && typeof properties === 'object'
      ? /** @type {Record<string, unknown>} */ (properties)
      : /** @type {Record<string, unknown>} */ (parameters)
  return Object.hasOwn(table, 'provider') && Object.hasOwn(table, 'model')
}

/**
 * 判断一次调用属于哪类委派工具。
 *
 * 这是「插件该不该看这次调用」的唯一判据：暴露 `provider`+`model` 路由对的工具才
 * 参与选路，`subagent_fork` 这类继承型委派工具只校验父路由，只读发现类工具（
 * `READ_ONLY_DISCOVERY_TOOLS`）与其余工具一律不看。判定必须先于读配置——否则配置
 * 损坏会把无关工具一起打死（曾经的实现缺陷，见计划 §4 最后一行的「其他工具 不看」）。
 * @param {string} toolName - 工具名。
 * @param {boolean} declaresRoute - 该工具实例是否声明了 `provider`+`model` 路由对。
 * @returns {{ relevant: boolean, isFork: boolean, declaresRoute: boolean }} 分类结果。
 */
export function delegationKind(toolName, declaresRoute) {
  const isFork = FORK_TOOL_NAMES.has(toolName)
  const relevant = (declaresRoute || isFork) && !READ_ONLY_DISCOVERY_TOOLS.has(toolName)
  return { relevant, isFork, declaresRoute }
}

/**
 * 判断一次工具调用是否放行。
 *
 * 纯函数：所有外部事实（会话授权集、父路由、已注册 provider、配置、日历、时钟）
 * 都由调用方传入，便于用固定输入覆盖判定矩阵（计划 §7.1）。
 * @param {object} input - 判定输入。
 * @param {string} input.toolName - 工具名（如 `subagent`）。
 * @param {Record<string, unknown> | undefined} input.parameters - 该工具实例的 `parameters` 声明（JSON Schema），用于判断是否暴露路由对。
 * @param {unknown} input.args - 已解析的调用参数。
 * @param {Array<{ provider: string, model: string }> | null} input.policy - 会话授权集；`null` 表示旧会话无授权。
 * @param {{ provider: string, model: string } | undefined} input.parentRoute - 父会话当前路由。
 * @param {object} input.config - 已校验的策略配置。
 * @param {object} input.calendar - 节假日日历。
 * @param {readonly string[]} [input.providerIds] - 已注册的 LLM provider id。
 * @param {Date} [input.now] - 判定时刻。
 * @returns {{ reason?: string, warnings: string[] }} 拒绝理由（放行时无）与需要记录的告警。
 */
export function decideGuard({
  toolName,
  parameters,
  args,
  policy,
  parentRoute,
  config,
  calendar,
  providerIds = [],
  now = new Date(),
}) {
  const warnings = []
  if (config.enabled !== true) return { warnings }
  const { relevant, isFork: isForkTool } = delegationKind(
    toolName,
    declaresRoutePair(parameters),
  )
  // 只看委派类工具：没有路由对、也不是 fork 的工具（如 standard preset 的
  // subagent）由原生逻辑把关，插件 fail-open，避免装插件即改变无关工具的行为。
  if (!relevant) return { warnings }
  // 旧会话没有授权事件：fail-open，避免安装插件即打死存量会话。
  if (policy === null) return { warnings }

  const resolved = resolvePolicy({ now, config, calendar })
  warnings.push(...resolved.warnings)

  const providers = new Set(providerIds)
  const missing = missingPlanProvider(config, providers)
  if (missing !== undefined) return { reason: missing, warnings }

  const policyKeys = new Set(policy.map(routeKey))
  const effective = resolved.allowed.filter((route) => policyKeys.has(routeKey(route)))
  if (effective.length === 0) {
    return { reason: emptyIntersection(toolName, resolved, policy), warnings }
  }

  const explicit = readExplicitRoute(args)
  if (explicit !== undefined) {
    if (!resolved.allowed.some((route) => sameRoute(route, explicit))) {
      return { reason: notAllowed(toolName, resolved, effective, explicit), warnings }
    }
    if (!policyKeys.has(routeKey(explicit))) {
      return { reason: staleSessionPolicy(toolName, resolved, effective, explicit, policy), warnings }
    }
    return { warnings }
  }

  if (isForkTool) {
    if (config.guardFork !== true) return { warnings }
    if (parentRoute === undefined) return { warnings }
    if (policyKeys.has(routeKey(parentRoute))) {
      const inPlan = resolved.allowed.some((route) => sameRoute(route, parentRoute))
      if (inPlan) return { warnings }
      return { reason: forkNotAllowed(toolName, resolved, effective, parentRoute), warnings }
    }
    return { reason: forkNotAllowed(toolName, resolved, effective, parentRoute), warnings }
  }

  // 未显式选路：继承只有在「父路由恰好等于本次默认」时才被接受（计划 §4）。
  if (
    parentRoute !== undefined &&
    sameRoute(parentRoute, resolved.default) &&
    effective.some((route) => sameRoute(route, resolved.default))
  ) {
    return { warnings }
  }
  if (!effective.some((route) => sameRoute(route, resolved.default))) {
    return { reason: defaultNotAuthorized(toolName, resolved, effective, policy), warnings }
  }
  return { reason: needsExplicitRoute(toolName, resolved, effective), warnings }
}

/**
 * 把一次真实工具执行折算成放行 / 拒绝：先分类、再读配置、再读会话授权。
 *
 * 顺序是契约的一部分（计划 §4「其他工具 不看」）：非委派工具绝不能因为策略配置损坏
 * 而被拒绝；旧会话没有授权记录时 fail-open 并记一次 warning。
 *
 * 「这个工具是否暴露路由对」必须问工具注册表：原生调用不带 `exec.schema`
 * （那是 PTC 内层调用才有的绑定时刻 schema），只有 `exec.toolSet` / 注册表能给出
 * 该实例真实的参数声明。判定失败时按 fail-open 处理并告警，绝不因为认不出工具而
 * 影响无关工具。
 * @param {object} input - 判定输入。
 * @param {{ name: string, schema?: { parameters?: Record<string, unknown> }, toolSet?: object, arguments?: unknown, agent?: object }} input.exec - 工具执行视图。
 * @param {{ read: () => object }} input.store - 配置存取器。
 * @param {object} input.calendar - 节假日日历。
 * @param {{ stateOf: (session: object) => Array<{ provider: string, model: string }> | null }} input.handle - 授权投影句柄。
 * @param {readonly string[]} input.providerIds - 已注册的 LLM provider id。
 * @param {{ get: (name: string, scope?: object, toolSet?: object) => { parameters?: Record<string, unknown> } | undefined }} [input.tools] - 工具注册表端口（`ctx.tools`）。
 * @param {(text: string) => void} [input.warn] - 告警收集器。
 * @returns {string | undefined} 拒绝理由（放行时无）。
 */
export function evaluateGuard({
  exec,
  store,
  calendar,
  handle,
  providerIds,
  tools,
  warn = () => {},
}) {
  const parameters = parametersOf(exec, tools)
  if (!delegationKind(exec.name, declaresRoutePair(parameters)).relevant) {
    if (parameters === undefined && looksLikeDelegation(exec.name)) {
      warn(
        `工具 "${exec.name}" 的声明无法解析（没有 exec.schema，注册表也未命中），` +
          `子代理模型策略对它 fail-open；若要恢复严格拦截请检查 tools.get 的解析路径。`,
      )
    }
    return undefined
  }

  let config
  try {
    config = store.read()
  } catch (error) {
    warn(`子代理模型策略配置不可用：${error.message}`)
    return `${exec.name} 路由被拒绝：策略配置不可用——${error.message}`
  }

  const agent = exec.agent
  if (agent === undefined || agent.session === undefined) return undefined
  const policy = handle.stateOf(agent.session)
  if (policy === null) {
    warn(
      `会话 ${String(agent.session.id)} 没有 model-selection 授权记录（设置启用前创建），` +
        `子代理模型策略对它 fail-open；严格拦截需新开一个会话。`,
    )
    return undefined
  }
  const decision = decideGuard({
    toolName: exec.name,
    parameters,
    args: exec.arguments,
    policy,
    parentRoute: parentRouteOf(agent),
    config,
    calendar,
    providerIds,
  })
  for (const warning of decision.warnings) warn(warning)
  return decision.reason
}

/**
 * 激活插件：解析配置与日历、注册投影单元、挂上 guard。
 * @param {import('@deepseek-ai/cordis').Context} ctx - 插件上下文。
 * @returns {Promise<void>} 激活完成。
 */
export async function apply(ctx) {
  const pluginDir = import.meta.dirname ?? fileURLToPath(new URL('.', import.meta.url))
  const candidates = resolveConfigCandidates({ pluginDir })
  const configPath = resolveConfigPath({ pluginDir })
  const store = createConfigStore(configPath)
  // 激活时就把配置与日历读一遍：缺文件、YAML 非法、数据损坏都让插件显式失败，
  // 而不是等到第一次派发才发现。
  const initial = store.read()
  const calendar = createCalendar(loadHolidayData(join(pluginDir, 'data', 'chinese-days.json')))

  const handle = await ctx.sessionProjections.register(policyProjection)
  const warned = new Set()

  /**
   * 每条告警只记一次：guard 会被每次派发调用，重复刷屏没有信息量。
   * @param {string} text - 告警文本。
   */
  const warnOnce = (text) => {
    if (warned.has(text)) return
    warned.add(text)
    ctx.logger.warn(text)
  }

  for (const definition of ctx.tools.definitions()) {
    if (!spawnsSubagent(definition.name)) continue
    // fork 型工具本来就没有 provider 参数，按父路由校验（§4 的 fork 两行），不在此告警。
    if (FORK_TOOL_NAMES.has(definition.name)) continue
    if (declaresRoutePair(definition.parameters)) continue
    warnOnce(
      `委派工具 "${definition.name}" 没有 provider/model 参数（modelSelection: false），` +
        `子代理模型策略对它 fail-open；需要动态路由时请在 preset 里打开 modelSelection。`,
    )
  }

  ctx.tools.guard((exec) =>
    evaluateGuard({
      exec,
      store,
      calendar,
      handle,
      providerIds: ctx.llm.listProviders().map((provider) => provider.id),
      tools: ctx.tools,
      warn: warnOnce,
    }),
  )

  ctx.logger.info(
    `子代理模型策略已启用：plan=${initial.plan}，配置=${configPath}` +
      (candidates.length > 1 ? `（候选：${candidates.join(' | ')}）` : ''),
  )
}

/**
 * 判断工具名是否像委派工具，只用于激活期的 fail-open 告警。
 *
 * 只读发现类工具（`READ_ONLY_DISCOVERY_TOOLS`）名字里也带 subagent，但它们按名豁免、
 * 不存在「声明解析不到」的疑点，不进告警。
 * @param {string} toolName - 工具名。
 * @returns {boolean} 是否像委派工具。
 */
function looksLikeDelegation(toolName) {
  if (READ_ONLY_DISCOVERY_TOOLS.has(toolName)) return false
  return FORK_TOOL_NAMES.has(toolName) || toolName === 'subagent' || toolName.includes('subagent')
}

/**
 * 判断工具名是否是一个「派生子代理」的工具（激活期告警用，比 `looksLikeDelegation` 严）。
 *
 * `list_subagent_models`、`session_subagent_*` 这类只是名字里带 subagent 的查询类工具
 * 不派生任何东西，不该被提示「缺 provider 参数」。
 * @param {string} toolName - 工具名。
 * @returns {boolean} 是否派生子代理。
 */
function spawnsSubagent(toolName) {
  return FORK_TOOL_NAMES.has(toolName) || /(^|_)subagent$/.test(toolName)
}

/**
 * 读出一次调用所属工具实例的 `parameters` 声明。
 *
 * 顺序是契约：注册表（`tools.get`，带 `toolSet` 可命中 preset 作用域里的实例）优先，
 * 其次是 PTC 内层调用携带的 `exec.schema`。两者都没有时返回 `undefined`，由调用方按
 * fail-open 处理并告警——绝不猜。
 * @param {{ name: string, agent?: object, toolSet?: object, schema?: { parameters?: unknown } }} exec - 工具执行视图。
 * @param {{ get: (name: string, scope?: object, toolSet?: object) => { parameters?: unknown } | undefined }} [tools] - 工具注册表端口。
 * @returns {Record<string, unknown> | undefined} 参数声明。
 */
function parametersOf(exec, tools) {
  if (tools !== undefined) {
    try {
      const definition = tools.get(exec.name, exec.agent, exec.toolSet)
      if (definition?.parameters !== undefined) {
        return /** @type {Record<string, unknown>} */ (definition.parameters)
      }
    } catch {
      // 注册表查询失败按「读不到声明」处理，继续尝试 schema。
    }
  }
  const fromSchema = exec.schema?.parameters
  return fromSchema === null || typeof fromSchema !== 'object'
    ? undefined
    : /** @type {Record<string, unknown>} */ (fromSchema)
}

/**
 * 读父会话当前路由：优先会话已记录的请求配置，其次 agent 的实时选项。
 * @param {{ options?: object, session?: { requestHeader?: () => { config?: object } | undefined } }} agent - 调用方 agent。
 * @returns {{ provider: string, model: string } | undefined} 精确路由，读不到时为 undefined。
 */
function parentRouteOf(agent) {
  let recorded
  try {
    recorded = agent.session?.requestHeader?.()?.config
  } catch {
    recorded = undefined
  }
  const options = agent.options ?? {}
  const provider = recorded?.provider ?? options.provider
  const model = recorded?.model ?? options.model
  if (typeof provider !== 'string' || provider === '') return undefined
  if (typeof model !== 'string' || model === '') return undefined
  return { provider, model }
}

/**
 * 读模型显式给出的路由；不完整（只给一边）时返回 undefined，交给原生报错。
 * @param {unknown} args - 已解析的调用参数。
 * @returns {{ provider: string, model: string } | undefined} 显式路由。
 */
function readExplicitRoute(args) {
  if (args === null || typeof args !== 'object') return undefined
  const { provider, model } = args
  if (typeof provider !== 'string' || provider === '') return undefined
  if (typeof model !== 'string' || model === '') return undefined
  return { provider, model }
}

/**
 * `plan=qwen` 时检查其路由的 provider 是否已注册；未注册即 fail-loud。
 * @param {object} config - 已校验的配置。
 * @param {ReadonlySet<string>} providers - 已注册 provider id。
 * @returns {string | undefined} 拒绝文本。
 */
function missingPlanProvider(config, providers) {
  if (config.plan !== 'qwen') return undefined
  const routes = [config.plans.qwen.flash, config.plans.qwen.pro]
  const missing = routes.find((route) => !providers.has(route.provider))
  if (missing === undefined) return undefined
  const registered = [...providers].sort().join('、') || '(none)'
  return [
    `subagent 路由被拒绝：plan=qwen，但 LLM provider "${missing.provider}" 未注册（已注册：${registered}）。`,
    `请先在 profile 中配置该 provider，并把真实路由写进 sub_root/subagent-model-policy.yaml 的 plans.qwen；` +
      `配置好之前请把 plan 改回 mimo 或 deepseek。`,
  ].join('\n')
}

/**
 * 档位选择规则教学行：拒绝文本是 v1 唯一教学点（不注入常驻上下文，零常驻 token 成本），
 * 每条路由拒绝都带上它（用户 2026-11-18 拍板）。内容是「flash 优先」的档位偏好：pro
 * 仅当任务确实复杂（较多临场决策、影响深远）才用，拿不准就选 flash；选用 pro 后必须
 * 事后向用户显式声明。注意它管的是**优先级**而非合法性——pro 恒在允许集内，只是不优先。
 */
const TIER_GUIDANCE =
  '模型选择规则：优先使用 flash；仅当任务确实较复杂（需要较多临场决策、且影响深远）时才使用 pro；' +
  '如果拿不准，就选 flash；一旦选用 pro，事后必须向用户显式声明。'

/**
 * 渲染拒绝文本的第一行：相位与可自证的理由。
 * @param {string} toolName - 工具名。
 * @param {object} resolved - `resolvePolicy` 的结果。
 * @returns {string} 首行。
 */
function denialHead(toolName, resolved) {
  return (
    `${toolName} 路由被拒绝：plan=${resolved.plan}，` +
    `相位=${resolved.phase === 'peak' ? '峰' : '谷'}（${resolved.basis}）。`
  )
}

/**
 * 渲染允许集，默认路由加标注；flash 级恒排在 pro/max 级之前（`flashFirst`）。
 * @param {Array<{ provider: string, model: string }>} routes - 允许集。
 * @param {{ provider: string, model: string }} fallback - 默认路由。
 * @returns {string} 展示文本。
 */
function formatAllowed(routes, fallback) {
  return flashFirst(routes)
    .map((route) => `${formatRoute(route)}${sameRoute(route, fallback) ? '（默认）' : ''}`)
    .join('、')
}

/**
 * 渲染会话授权集；flash 级恒排在 pro/max 级之前（GUI 持有的顺序不作数）。
 * @param {Array<{ provider: string, model: string }>} policy - 会话授权集。
 * @returns {string} 展示文本。
 */
function formatPolicy(policy) {
  return flashFirst(policy)
    .map((route) => formatRoute(route))
    .join('、')
}

/**
 * 显式路由不在本相位允许集内。
 * @param {string} toolName - 工具名。
 * @param {object} resolved - 解析结果。
 * @param {Array<{ provider: string, model: string }>} effective - 有效允许集。
 * @param {{ provider: string, model: string }} explicit - 请求的路由。
 * @returns {string} 拒绝文本。
 */
function notAllowed(toolName, resolved, effective, explicit) {
  return [
    denialHead(toolName, resolved),
    `允许：${formatAllowed(effective, resolved.default)}。`,
    `请求的路由 ${formatRoute(explicit)} 不在本相位的允许集内；请改用上面允许的路由（默认 ` +
      `${formatRoute(resolved.default)}）重试本次调用。`,
    TIER_GUIDANCE,
  ].join('\n')
}

/**
 * 显式路由在政策允许集内、但不在会话授权集内（会话策略较旧）。
 * @param {string} toolName - 工具名。
 * @param {object} resolved - 解析结果。
 * @param {Array<{ provider: string, model: string }>} effective - 有效允许集。
 * @param {{ provider: string, model: string }} explicit - 请求的路由。
 * @param {Array<{ provider: string, model: string }>} policy - 会话授权集。
 * @returns {string} 拒绝文本。
 */
function staleSessionPolicy(toolName, resolved, effective, explicit, policy) {
  return [
    denialHead(toolName, resolved),
    `允许：${formatAllowed(effective, resolved.default)}。`,
    `请求的路由 ${formatRoute(explicit)} 在政策允许集内，但不在本会话的授权集（${formatPolicy(policy)}）；` +
      `请在 GUI 的 model-selection 白名单补上该路由，然后新开一个会话再派发。`,
    TIER_GUIDANCE,
  ].join('\n')
}

/**
 * 政策允许集与会话授权集没有交集：两套名单指向了不同批次的路由。
 * @param {string} toolName - 工具名。
 * @param {object} resolved - 解析结果。
 * @param {Array<{ provider: string, model: string }>} policy - 会话授权集。
 * @returns {string} 拒绝文本。
 */
function emptyIntersection(toolName, resolved, policy) {
  return [
    denialHead(toolName, resolved),
    `计划允许集：${formatAllowed(resolved.allowed, resolved.default)}。`,
    `会话授权集：${formatPolicy(policy)}。`,
    `两侧没有交集；请检查 sub_root/subagent-model-policy.yaml 的 plans 与 GUI 白名单是否指向同一批 ` +
      `provider/model，补齐后新开一个会话再派发。`,
    TIER_GUIDANCE,
  ].join('\n')
}

/**
 * 未显式选路、父路由也不是默认：要求模型带路由重试。
 *
 * 不回显父路由（v1.2，用户 2026-11-18 拍板）：主会话常用 pro 级而子代理默认 flash 级，
 * 隐式派发恒落入本分支，回显「父会话当前路由」会被模型当作推荐照抄，恰好锚定到贵档
 * （pro 在允许集内，照抄即合法通过）。默认值已在括号里给出，足以指导重试。
 * `forkNotAllowed` 仍回显父路由——那里父路由是被拒对象，语义无歧义。
 * @param {string} toolName - 工具名。
 * @param {object} resolved - 解析结果。
 * @param {Array<{ provider: string, model: string }>} effective - 有效允许集。
 * @returns {string} 拒绝文本。
 */
function needsExplicitRoute(toolName, resolved, effective) {
  return [
    denialHead(toolName, resolved),
    `允许：${formatAllowed(effective, resolved.default)}。`,
    `请带 provider/model 重试本次调用（默认 ${formatRoute(resolved.default)}）。`,
    TIER_GUIDANCE,
  ].join('\n')
}

/**
 * 本相位的默认路由不在会话授权集内。
 * @param {string} toolName - 工具名。
 * @param {object} resolved - 解析结果。
 * @param {Array<{ provider: string, model: string }>} effective - 有效允许集。
 * @param {Array<{ provider: string, model: string }>} policy - 会话授权集。
 * @returns {string} 拒绝文本。
 */
function defaultNotAuthorized(toolName, resolved, effective, policy) {
  return [
    denialHead(toolName, resolved),
    `允许：${formatAllowed(effective, resolved.default)}。`,
    `本相位的默认路由 ${formatRoute(resolved.default)} 不在本会话的授权集（${formatPolicy(policy)}）；` +
      `请在 GUI 的 model-selection 白名单补上该路由后新开一个会话，或显式选择允许集内的路由。`,
    TIER_GUIDANCE,
  ].join('\n')
}

/**
 * fork 只能继承父路由，而父路由不在有效允许集内。
 * @param {string} toolName - 工具名。
 * @param {object} resolved - 解析结果。
 * @param {Array<{ provider: string, model: string }>} effective - 有效允许集。
 * @param {{ provider: string, model: string }} parentRoute - 父路由。
 * @returns {string} 拒绝文本。
 */
function forkNotAllowed(toolName, resolved, effective, parentRoute) {
  return [
    denialHead(toolName, resolved),
    `允许：${formatAllowed(effective, resolved.default)}。`,
    `${toolName} 不能选模型，只能继承父路由 ${formatRoute(parentRoute)}；它在当前允许集之外。` +
      `请改用带 provider/model 的 subagent 派发，或换到允许该路由的相位再派发。`,
    TIER_GUIDANCE,
  ].join('\n')
}
