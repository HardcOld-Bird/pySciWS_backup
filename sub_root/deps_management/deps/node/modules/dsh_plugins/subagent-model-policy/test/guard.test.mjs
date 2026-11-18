/**
 * guard 判定测试：固定输入覆盖计划 §4 的每一行判定，不需要 DSH 运行时（计划 §7.1）。
 */
import assert from 'node:assert/strict'
import { join } from 'node:path'
import { test } from 'node:test'
import { createCalendar, loadHolidayData } from '../calendar.js'
import { normalizeConfig } from '../config.js'
import { decideGuard, declaresRoutePair, delegationKind, evaluateGuard, policyProjection } from '../index.js'

const CALENDAR = createCalendar(
  loadHolidayData(join(import.meta.dirname, '..', 'data', 'chinese-days.json')),
)

const DEEPSEEK_FLASH = { provider: 'deepseek-account', model: 'deepseek-flash' }
const MIMO_FLASH = { provider: 'xiaomi-token-plan-cn', model: 'mimo-v2.6-flash' }
const MIMO_PRO = { provider: 'xiaomi-token-plan-cn', model: 'mimo-v2.6-pro' }
const QWEN_FLASH = { provider: 'qwen-token-plan-cn', model: 'qwen-flash' }
const ALL_ROUTES = [MIMO_FLASH, MIMO_PRO, DEEPSEEK_FLASH]
const PROVIDERS = ['deepseek-account', 'xiaomi-token-plan-cn']

/**
 * `defineTool` 编译后的参数声明（对象根 JSON Schema）：参数名在 `properties` 下，
 * 与宿主真实形状一致（`ToolDefinition.parameters` / `ToolSchema.parameters`）。
 */
const ROUTE_PARAMS = {
  type: 'object',
  properties: {
    prompt: { type: 'string' },
    provider: { type: 'string' },
    model: { type: 'string' },
  },
  required: ['prompt'],
}

/** 同形状但不暴露 `provider` 的声明（standard preset 的 subagent）。 */
const PLAIN_PARAMS = {
  type: 'object',
  properties: { prompt: { type: 'string' } },
  required: ['prompt'],
}

/** 空声明：`subagent_fork` 这类只继承父路由的工具用它。 */
const EMPTY_PARAMS = { type: 'object', properties: {} }

/**
 * `cordis_inspect_query` 的真实声明形状：`provider` 是 Inspect Provider ID，
 * 没有 `model`——v1.1 起不算路由对，guard 不看（v1 误伤回归用例）。
 */
const INSPECT_PARAMS = {
  type: 'object',
  properties: {
    platform: { type: 'string' },
    provider: { type: 'string' },
    method: { type: 'string' },
    input: { type: 'json' },
  },
  required: ['platform', 'provider', 'method'],
}

/**
 * 造一个只回答单条工具声明的注册表端口（`ctx.tools` 的最小替身）。
 * @param {Record<string, unknown> | undefined} parameters - `tools.get` 命中的声明；`undefined` 表示查不到。
 * @returns {{ get: () => { parameters?: Record<string, unknown> } | undefined }} 伪注册表。
 */
function toolsOf(parameters) {
  return { get: () => (parameters === undefined ? undefined : { parameters }) }
}

const PEAK = new Date('2026-11-09T10:00:00+08:00') // 周一峰段
const VALLEY = new Date('2026-11-09T20:00:00+08:00') // 周一谷段

/**
 * 以默认输入调用判定，按需覆盖字段。
 * @param {object} [overrides] - 覆盖字段。
 * @returns {ReturnType<typeof decideGuard>} 判定结果。
 */
function decide(overrides = {}) {
  return decideGuard({
    toolName: 'subagent',
    parameters: ROUTE_PARAMS,
    args: { prompt: 'x' },
    policy: ALL_ROUTES,
    parentRoute: MIMO_FLASH,
    config: normalizeConfig({ forcePhase: 'peak' }),
    calendar: CALENDAR,
    providerIds: PROVIDERS,
    now: PEAK,
    ...overrides,
  })
}

test('无 provider 参数的工具 fail-open（standard preset 的 subagent）', () => {
  const decision = decide({ parameters: PLAIN_PARAMS })
  assert.equal(decision.reason, undefined)
})

test('参数声明形状：JSON Schema 与旧属性表都能认出完整路由对', () => {
  assert.equal(declaresRoutePair(ROUTE_PARAMS), true)
  assert.equal(declaresRoutePair({ provider: { type: 'string' }, model: { type: 'string' } }), true)
  // 单边不是路由：provider-only（cordis_inspect_query）与 model-only 都不算。
  assert.equal(declaresRoutePair({ provider: { type: 'string' } }), false)
  assert.equal(declaresRoutePair({ model: { type: 'string' } }), false)
  assert.equal(declaresRoutePair(PLAIN_PARAMS), false)
  assert.equal(declaresRoutePair(undefined), false)
  assert.equal(declaresRoutePair(null), false)
})

test('只暴露 provider、没有 model 的工具不算选路工具 → 不看（cordis_inspect_query 误伤回归）', () => {
  // 父路由不是默认：若被误判为选路工具，这里会被 needsExplicitRoute 拒绝。
  const decision = decide({
    toolName: 'cordis_inspect_query',
    parameters: INSPECT_PARAMS,
    args: { platform: 'host', provider: 'host:some-inspect', method: 'schemas' },
    parentRoute: DEEPSEEK_FLASH,
  })
  assert.equal(decision.reason, undefined)
  assert.equal(delegationKind('cordis_inspect_query', declaresRoutePair(INSPECT_PARAMS)).relevant, false)
})

test('只读发现工具按名豁免，无参查询不被选路检查（list_subagent_models 误伤回归）', () => {
  // 无参调用（列 provider）+ 父路由不是默认：v1 会拒，v1.1 起豁免。
  const decision = decide({
    toolName: 'list_subagent_models',
    parameters: ROUTE_PARAMS,
    args: {},
    parentRoute: DEEPSEEK_FLASH,
  })
  assert.equal(decision.reason, undefined)
  assert.equal(delegationKind('list_subagent_models', declaresRoutePair(ROUTE_PARAMS)).relevant, false)
  // 即使模型硬塞完整越界路由，发现工具也不受相位限制（发现 ≠ 派发）。
  const inspected = decide({
    toolName: 'list_subagent_models',
    parameters: ROUTE_PARAMS,
    args: { provider: DEEPSEEK_FLASH.provider, model: DEEPSEEK_FLASH.model },
  })
  assert.equal(inspected.reason, undefined)
})

test('显式路由在允许集内 → 放行', () => {
  const decision = decide({ args: { prompt: 'x', provider: MIMO_FLASH.provider, model: MIMO_FLASH.model } })
  assert.equal(decision.reason, undefined)
})

test('显式路由不在本相位允许集 → 拒绝并列出允许集', () => {
  const decision = decide({
    args: { prompt: 'x', provider: DEEPSEEK_FLASH.provider, model: DEEPSEEK_FLASH.model },
  })
  assert.match(decision.reason, /相位=峰/)
  assert.match(decision.reason, /不在本相位的允许集内/)
  assert.match(decision.reason, /xiaomi-token-plan-cn\/mimo-v2\.6-flash（默认）/)
})

test('显式路由在政策允许集、但不在会话授权集 → 拒绝并提示会话策略较旧', () => {
  const decision = decide({
    args: { prompt: 'x', provider: MIMO_PRO.provider, model: MIMO_PRO.model },
    policy: [MIMO_FLASH, DEEPSEEK_FLASH],
  })
  assert.match(decision.reason, /不在本会话的授权集/)
  assert.match(decision.reason, /新开一个会话/)
})

test('未显式选路且父路由恰好等于默认 → 放行（不浪费一跳）', () => {
  const decision = decide()
  assert.equal(decision.reason, undefined)
})

test('未显式选路且父路由不是默认 → 拒绝并指示带路由重试（不回显父路由）', () => {
  const decision = decide({ parentRoute: DEEPSEEK_FLASH })
  // v1.2：回显父路由会被误读为推荐（主会话 pro 级 ≠ 子代理默认 flash 级），已删除。
  assert.doesNotMatch(decision.reason, /父会话|deepseek-account\/deepseek-flash/)
  assert.match(decision.reason, /请带 provider\/model 重试本次调用/)
  assert.match(decision.reason, /优先使用 flash/)
})

test('旧会话没有授权记录 → fail-open', () => {
  const decision = decide({ policy: null, parentRoute: DEEPSEEK_FLASH })
  assert.equal(decision.reason, undefined)
})

test('subagent_fork：父路由不在有效允许集 → 拒绝（回显父路由，它是被拒对象）', () => {
  const decision = decide({ toolName: 'subagent_fork', parameters: EMPTY_PARAMS, parentRoute: DEEPSEEK_FLASH })
  assert.match(decision.reason, /不能选模型，只能继承父路由/)
  assert.match(decision.reason, /deepseek-account\/deepseek-flash/)
})

test('subagent_fork：guardFork=false 时不拦', () => {
  const decision = decide({
    toolName: 'subagent_fork',
    parameters: EMPTY_PARAMS,
    parentRoute: DEEPSEEK_FLASH,
    config: normalizeConfig({ forcePhase: 'peak', guardFork: false }),
  })
  assert.equal(decision.reason, undefined)
})

test('plan=qwen 且 provider 未注册 → fail-loud', () => {
  const decision = decide({
    config: normalizeConfig({ forcePhase: 'peak', plan: 'qwen' }),
    policy: [QWEN_FLASH],
  })
  assert.match(decision.reason, /LLM provider "qwen-token-plan-cn" 未注册/)
  assert.match(decision.reason, /plans\.qwen/)
})

test('政策允许集与会话授权集没有交集 → 拒绝并列出两侧', () => {
  const decision = decide({ policy: [QWEN_FLASH] })
  assert.match(decision.reason, /两侧没有交集/)
  assert.match(decision.reason, /计划允许集：/)
  assert.match(decision.reason, /会话授权集：qwen-token-plan-cn\/qwen-flash/)
})

test('默认路由不在会话授权集 → 拒绝并提示补白名单', () => {
  const decision = decide({
    config: normalizeConfig({ forcePhase: 'valley' }),
    policy: [MIMO_PRO],
  })
  assert.match(decision.reason, /默认路由 deepseek-account\/deepseek-flash 不在本会话的授权集/)
})

test('拒绝文本带档位选择规则教学行，且允许集列举 flash 先于 pro（唯一教学点）', () => {
  const decision = decide({
    args: { prompt: 'x', provider: DEEPSEEK_FLASH.provider, model: DEEPSEEK_FLASH.model },
  })
  const reason = decision.reason
  assert.match(reason, /优先使用 flash/)
  assert.match(reason, /拿不准，就选 flash/)
  assert.match(reason, /事后必须向用户显式声明/)
  assert.ok(reason.indexOf('mimo-v2.6-flash') < reason.indexOf('mimo-v2.6-pro'), reason)
})

test('会话授权集列举也 flash 先行（GUI 持有的顺序不作数）', () => {
  const QWEN_MAX = { provider: 'qwen-token-plan-cn', model: 'qwen-max' }
  const decision = decide({ policy: [QWEN_MAX, QWEN_FLASH] })
  assert.match(decision.reason, /两侧没有交集/)
  const line = decision.reason.split('\n').find((candidate) => candidate.startsWith('会话授权集：'))
  assert.ok(line !== undefined, decision.reason)
  assert.ok(line.includes('qwen-flash') && line.indexOf('qwen-flash') < line.indexOf('qwen-max'), line)
})

test('只给 provider 不给 model 时按未显式处理，交给原生报错', () => {
  const decision = decide({ args: { prompt: 'x', provider: 'deepseek-account' }, parentRoute: MIMO_FLASH })
  assert.equal(decision.reason, undefined)
})

test('disabled 时全部放行', () => {
  const decision = decide({
    config: normalizeConfig({ enabled: false }),
    args: { prompt: 'x', provider: DEEPSEEK_FLASH.provider, model: DEEPSEEK_FLASH.model },
  })
  assert.equal(decision.reason, undefined)
})

test('授权投影单元与宿主内置同键同版本', () => {  assert.equal(policyProjection.key, 'subagentModelSelectionPolicy')
  assert.equal(policyProjection.stateVersion, 1)
  assert.equal(policyProjection.init(), null)
  const stamped = policyProjection.apply(null, {
    type: 'subagent/model-selection-policy',
    data: { allowedModels: [MIMO_FLASH] },
  })
  assert.deepEqual(stamped, [MIMO_FLASH])
  // 第一次盖章即永久：后续同类事件不改写状态。
  assert.equal(
    policyProjection.apply(stamped, {
      type: 'subagent/model-selection-policy',
      data: { allowedModels: [MIMO_PRO] },
    }),
    stamped,
  )
  // 无关事件返回同一引用（不触发下游重算）。
  assert.equal(policyProjection.apply(stamped, { type: 'turn/start' }), stamped)
})

/**
 * 造一个只会按给定授权集作答的投影句柄。
 * @param {Array<{ provider: string, model: string }> | null} policy - 句柄返回的授权集。
 * @returns {{ stateOf: () => Array<{ provider: string, model: string }> | null }} 伪句柄。
 */
function handleOf(policy) {
  return { stateOf: () => policy }
}

test('配置损坏时只拦委派工具，不波及无关工具（原生调用无 exec.schema 也要认出，回归）', () => {
  const brokenStore = {
    read() {
      throw new Error('plan 必须是 mimo / qwen / deepseek 之一')
    },
  }
  const warnings = []
  const readCall = evaluateGuard({
    exec: { name: 'read', arguments: {} },
    store: brokenStore,
    calendar: CALENDAR,
    handle: handleOf(ALL_ROUTES),
    providerIds: PROVIDERS,
    tools: toolsOf(PLAIN_PARAMS),
    warn: (text) => warnings.push(text),
  })
  assert.equal(readCall, undefined)

  const delegation = evaluateGuard({
    exec: {
      name: 'subagent',
      arguments: { prompt: 'x' },
      agent: { session: { id: 's1' } },
    },
    store: brokenStore,
    calendar: CALENDAR,
    handle: handleOf(ALL_ROUTES),
    providerIds: PROVIDERS,
    tools: toolsOf(ROUTE_PARAMS),
  })
  assert.match(delegation, /subagent 路由被拒绝：策略配置不可用/)
})

test('原生调用（无 exec.schema）靠注册表认出委派工具，显式越界路由仍被拒', () => {
  const reason = evaluateGuard({
    exec: {
      name: 'subagent',
      arguments: { prompt: 'x', provider: DEEPSEEK_FLASH.provider, model: DEEPSEEK_FLASH.model },
      agent: { session: { id: 'new-session' }, options: { provider: MIMO_FLASH.provider, model: MIMO_FLASH.model } },
    },
    store: { read: () => normalizeConfig({ forcePhase: 'peak' }) },
    calendar: CALENDAR,
    handle: handleOf(ALL_ROUTES),
    providerIds: PROVIDERS,
    tools: toolsOf(ROUTE_PARAMS),
  })
  assert.match(reason, /不在本相位的允许集内/)
})

test('PTC 内层调用带 exec.schema 时按其声明分类', () => {
  const reason = evaluateGuard({
    exec: {
      name: 'subagent',
      schema: { parameters: ROUTE_PARAMS },
      arguments: { prompt: 'x', provider: DEEPSEEK_FLASH.provider, model: DEEPSEEK_FLASH.model },
      agent: { session: { id: 'ptc-session' }, options: { provider: MIMO_FLASH.provider, model: MIMO_FLASH.model } },
    },
    store: { read: () => normalizeConfig({ forcePhase: 'peak' }) },
    calendar: CALENDAR,
    handle: handleOf(ALL_ROUTES),
    providerIds: PROVIDERS,
    tools: { get: () => undefined },
  })
  assert.match(reason, /不在本相位的允许集内/)
})

test('工具声明无法解析时对疑似委派工具 fail-open 并告警', () => {
  const warnings = []
  const reason = evaluateGuard({
    exec: { name: 'subagent', arguments: { prompt: 'x' }, agent: { session: { id: 's1' } } },
    store: { read: () => normalizeConfig({}) },
    calendar: CALENDAR,
    handle: handleOf(ALL_ROUTES),
    providerIds: PROVIDERS,
    tools: { get: () => undefined },
    warn: (text) => warnings.push(text),
  })
  assert.equal(reason, undefined)
  assert.equal(warnings.length, 1)
  assert.match(warnings[0], /声明无法解析/)
})

test('旧会话没有授权记录 → evaluateGuard 放行并告警一次', () => {
  const warnings = []
  const reason = evaluateGuard({
    exec: {
      name: 'subagent',
      arguments: { prompt: 'x' },
      agent: { session: { id: 'old-session' } },
    },
    store: { read: () => normalizeConfig({}) },
    calendar: CALENDAR,
    handle: handleOf(null),
    providerIds: PROVIDERS,
    tools: toolsOf(ROUTE_PARAMS),
    warn: (text) => warnings.push(text),
  })
  assert.equal(reason, undefined)
  assert.equal(warnings.length, 1)
  assert.match(warnings[0], /fail-open/)
})

test('新会话显式选了越界路由 → evaluateGuard 拒绝', () => {
  const reason = evaluateGuard({
    exec: {
      name: 'subagent',
      arguments: { prompt: 'x', provider: DEEPSEEK_FLASH.provider, model: DEEPSEEK_FLASH.model },
      agent: { session: { id: 'new-session' }, options: { provider: MIMO_FLASH.provider, model: MIMO_FLASH.model } },
    },
    store: { read: () => normalizeConfig({ forcePhase: 'peak' }) },
    calendar: CALENDAR,
    handle: handleOf(ALL_ROUTES),
    providerIds: PROVIDERS,
    tools: toolsOf(ROUTE_PARAMS),
  })
  assert.match(reason, /不在本相位的允许集内/)
})
