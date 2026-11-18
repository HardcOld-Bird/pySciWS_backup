/**
 * 策略配置：唯一用户配置点是仓库 `sub_root/subagent-model-policy.yaml`。
 *
 * 为什么配置不在插件行里：计划 §5.4 要求「用户只在一个配置点切换套餐，改完即
 * 生效」。DSH 的插件行改配置要走 profile patch 并重启，而本插件的配置是每天可能
 * 手动切换的套餐开关，放在仓库文本文件里按 mtime 热读更贴合使用方式。
 *
 * 校验是严格的：未知键、缺 provider/model、非法窗口一律 throw，不做回退（极简原则）。
 */
import { existsSync, readFileSync, statSync } from 'node:fs'
import { resolve } from 'node:path'
import { parse } from 'yaml'
import { PLAN_IDS } from './policy.js'

/** 配置文件默认名（仓库 `sub_root` 下）。 */
export const CONFIG_FILE_NAME = 'subagent-model-policy.yaml'

/** 环境变量：显式指定配置路径，优先于任何推导。 */
export const CONFIG_PATH_ENV = 'PYSCI_SUBAGENT_MODEL_POLICY'

/** 全部合法配置键，用于把拼写错误变成显式报错。 */
const KNOWN_KEYS = Object.freeze([
  'enabled',
  'plan',
  'plans',
  'peak',
  'leadtimeSeconds',
  'unknownYear',
  'guardFork',
  'forcePhase',
])

/** 未配置时使用的默认值；与 `sub_root/subagent-model-policy.yaml` 的内容保持一致。 */
export const DEFAULT_CONFIG = deepFreeze({
  enabled: true,
  plan: 'mimo',
  plans: {
    mimo: {
      flash: { provider: 'xiaomi-token-plan-cn', model: 'mimo-v2.6-flash' },
      pro: { provider: 'xiaomi-token-plan-cn', model: 'mimo-v2.6-pro' },
    },
    qwen: {
      // 占位：qwen provider 落地前，plan=qwen 会被 guard 直接拒绝（fail-loud）。
      flash: { provider: 'qwen-token-plan-cn', model: 'qwen-flash' },
      pro: { provider: 'qwen-token-plan-cn', model: 'qwen-max' },
    },
    deepseek: {
      flash: { provider: 'deepseek-account', model: 'deepseek-flash' },
    },
  },
  peak: [
    [9, 12],
    [14, 18],
  ],
  leadtimeSeconds: 600,
  unknownYear: 'peak',
  guardFork: true,
  forcePhase: 'auto',
})

/**
 * 列出配置文件候选路径：环境变量 → 插件目录上溯到仓库根 → `DSH_HOME` 上溯到仓库根。
 *
 * 插件目录位于 `modules/dsh_plugins/subagent-model-policy`，上溯六级即仓库根的
 * `sub_root`；`DSH_HOME` 位于 `modules/dsh_home`，上溯五级同样落在 `sub_root`。
 * 两条推导都是布局事实而非运行期猜测：链接安装命中前者，复制进 profile 的安装
 * 命中后者，因此按顺序取第一个存在的文件。
 * @param {{ env?: NodeJS.ProcessEnv, pluginDir: string, dshHome?: string }} input - 环境、插件目录与 DSH_HOME。
 * @returns {string[]} 去重后的绝对路径候选。
 */
export function resolveConfigCandidates({ env = process.env, pluginDir, dshHome = env.DSH_HOME }) {
  const override = env[CONFIG_PATH_ENV]
  if (override !== undefined && override.trim() !== '') return [resolve(override)]
  const candidates = [resolve(pluginDir, '..', '..', '..', '..', '..', '..', CONFIG_FILE_NAME)]
  if (dshHome !== undefined && dshHome.trim() !== '') {
    candidates.push(resolve(dshHome, '..', '..', '..', '..', '..', CONFIG_FILE_NAME))
  }
  return [...new Set(candidates)]
}

/**
 * 取第一个真实存在的候选配置路径；都不存在时返回首个候选，由读取时报错。
 * @param {{ env?: NodeJS.ProcessEnv, pluginDir: string, dshHome?: string }} input - 同 {@link resolveConfigCandidates}。
 * @returns {string} 配置文件的绝对路径。
 */
export function resolveConfigPath(input) {
  const candidates = resolveConfigCandidates(input)
  return candidates.find((candidate) => existsSync(candidate)) ?? candidates[0]
}

/**
 * 校验并归一化一份原始配置。
 *
 * 缺失的键取默认值；`plans` 按套餐逐层合并，只覆盖给出的字段。
 * @param {unknown} raw - YAML 解析结果（可以是 null）。
 * @param {{ source?: string }} [options] - 报错时使用的来源描述。
 * @returns {object} 已校验的配置（深拷贝，不会与默认值共享引用）。
 */
export function normalizeConfig(raw, { source = '策略配置' } = {}) {
  if (raw === null || raw === undefined) raw = {}
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error(`${source}：顶层必须是映射（YAML object），实际是 ${describe(raw)}`)
  }
  for (const key of Object.keys(raw)) {
    if (!KNOWN_KEYS.includes(key)) {
      throw new Error(`${source}：未知配置键 "${key}"（支持：${KNOWN_KEYS.join('、')}）`)
    }
  }
  return {
    enabled: readBoolean(raw.enabled, 'enabled', DEFAULT_CONFIG.enabled, source),
    plan: readPlan(raw.plan, source),
    plans: readPlans(raw.plans, source),
    peak: readPeak(raw.peak, source),
    leadtimeSeconds: readLeadtime(raw.leadtimeSeconds, source),
    unknownYear: readEnum(raw.unknownYear, ['peak', 'valley'], 'unknownYear', DEFAULT_CONFIG.unknownYear, source),
    guardFork: readBoolean(raw.guardFork, 'guardFork', DEFAULT_CONFIG.guardFork, source),
    forcePhase: readEnum(raw.forcePhase, ['auto', 'peak', 'valley'], 'forcePhase', DEFAULT_CONFIG.forcePhase, source),
  }
}

/**
 * 建立按 mtime 热读的配置存取器。
 *
 * 读失败（文件缺失、YAML 非法、校验不过）一律 throw：缺配置是显式错误，不静默用
 * 默认值。调用方（guard）会把异常转成拒绝理由，apply 时则让插件激活失败。
 * @param {string} filePath - 配置文件绝对路径。
 * @returns {{ path: string, read: () => object, invalidate: () => void }} 存取器。
 */
export function createConfigStore(filePath) {
  /** @type {{ mtimeMs: number, size: number, config: object } | undefined} */
  let cached
  return {
    path: filePath,
    read() {
      let info
      try {
        info = statSync(filePath)
      } catch (error) {
        throw new Error(
          `未找到策略配置文件：${filePath}（${error.code ?? error.message}）；` +
            `用 ${CONFIG_PATH_ENV} 指定路径，或在 sub_root 下创建它`,
        )
      }
      if (cached !== undefined && cached.mtimeMs === info.mtimeMs && cached.size === info.size) {
        return cached.config
      }
      const text = readFileSync(filePath, 'utf8')
      const config = normalizeConfig(parse(text), { source: filePath })
      cached = { mtimeMs: info.mtimeMs, size: info.size, config }
      return config
    },
    invalidate() {
      cached = undefined
    },
  }
}

/**
 * 读出套餐 id。
 * @param {unknown} value - 原始值。
 * @param {string} source - 来源描述。
 * @returns {string} 套餐 id。
 */
function readPlan(value, source) {
  if (value === undefined) return DEFAULT_CONFIG.plan
  if (typeof value !== 'string' || !PLAN_IDS.includes(value)) {
    throw new Error(`${source}：plan 必须是 ${PLAN_IDS.join(' / ')} 之一，实际是 ${describe(value)}`)
  }
  return value
}

/**
 * 合并并校验三家套餐的路由表。
 * @param {unknown} raw - 原始 `plans` 段。
 * @param {string} source - 来源描述。
 * @returns {Record<string, Record<string, { provider: string, model: string }>>} 完整路由表。
 */
function readPlans(raw, source) {
  const plans = structuredClone(DEFAULT_CONFIG.plans)
  if (raw !== undefined) {
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
      throw new Error(`${source}：plans 必须是映射，实际是 ${describe(raw)}`)
    }
    for (const [planId, routes] of Object.entries(raw)) {
      if (!PLAN_IDS.includes(planId)) {
        throw new Error(`${source}：plans 出现未知套餐 "${planId}"（支持：${PLAN_IDS.join('、')}）`)
      }
      if (typeof routes !== 'object' || routes === null || Array.isArray(routes)) {
        throw new Error(`${source}：plans.${planId} 必须是映射，实际是 ${describe(routes)}`)
      }
      for (const [tier, route] of Object.entries(routes)) {
        if (tier !== 'flash' && tier !== 'pro') {
          throw new Error(`${source}：plans.${planId} 出现未知档位 "${tier}"（支持：flash、pro）`)
        }
        plans[planId][tier] = readRoute(route, `plans.${planId}.${tier}`, source)
      }
    }
  }
  for (const planId of PLAN_IDS) {
    const tiers = planId === 'deepseek' ? ['flash'] : ['flash', 'pro']
    for (const tier of tiers) {
      if (plans[planId][tier] === undefined) {
        throw new Error(`${source}：plans.${planId}.${tier} 缺失（${planId} 需要 ${tiers.join(' 与 ')}）`)
      }
    }
  }
  return plans
}

/**
 * 校验一条精确路由。
 * @param {unknown} raw - 原始值。
 * @param {string} field - 字段名，用于报错。
 * @param {string} source - 来源描述。
 * @returns {{ provider: string, model: string }} 路由。
 */
function readRoute(raw, field, source) {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new Error(`${source}：${field} 必须是 { provider, model } 映射，实际是 ${describe(raw)}`)
  }
  for (const key of Object.keys(raw)) {
    if (key !== 'provider' && key !== 'model') {
      throw new Error(`${source}：${field} 出现未知键 "${key}"（只支持 provider、model）`)
    }
  }
  for (const key of ['provider', 'model']) {
    const value = raw[key]
    if (typeof value !== 'string' || value.trim() === '') {
      throw new Error(`${source}：${field}.${key} 必须是非空字符串，实际是 ${describe(value)}`)
    }
  }
  return { provider: raw.provider, model: raw.model }
}

/**
 * 校验峰段窗口：半开区间、整数小时、不重叠。
 * @param {unknown} raw - 原始 `peak` 值。
 * @param {string} source - 来源描述。
 * @returns {Array<[number, number]>} 按起点排序的窗口。
 */
function readPeak(raw, source) {
  const value = raw ?? DEFAULT_CONFIG.peak
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${source}：peak 必须是非空数组，实际是 ${describe(value)}`)
  }
  const windows = value.map((window, index) => {
    if (!Array.isArray(window) || window.length !== 2) {
      throw new Error(`${source}：peak[${index}] 必须是 [开始小时, 结束小时]`)
    }
    const [start, end] = window
    for (const [label, hour] of [['开始', start], ['结束', end]]) {
      if (!Number.isInteger(hour) || hour < 0 || hour > 24) {
        throw new Error(`${source}：peak[${index}] 的${label}小时必须是 0–24 的整数，实际是 ${describe(hour)}`)
      }
    }
    if (start >= end) {
      throw new Error(`${source}：peak[${index}] 必须满足 0 ≤ 开始 < 结束 ≤ 24，实际是 [${start}, ${end}]`)
    }
    return [start, end]
  })
  windows.sort((left, right) => left[0] - right[0])
  for (let index = 1; index < windows.length; index += 1) {
    if (windows[index][0] < windows[index - 1][1]) {
      throw new Error(
        `${source}：peak 窗口重叠（[${windows[index - 1]}] 与 [${windows[index]}]）`,
      )
    }
  }
  return windows
}

/**
 * 校验早切提前量（秒）。
 * @param {unknown} raw - 原始值。
 * @param {string} source - 来源描述。
 * @returns {number} 非负整数秒。
 */
function readLeadtime(raw, source) {
  if (raw === undefined) return DEFAULT_CONFIG.leadtimeSeconds
  if (!Number.isInteger(raw) || raw < 0) {
    throw new Error(`${source}：leadtimeSeconds 必须是非负整数秒（0 关闭早切），实际是 ${describe(raw)}`)
  }
  return raw
}

/**
 * 读布尔值。
 * @param {unknown} value - 原始值。
 * @param {string} field - 字段名。
 * @param {boolean} fallback - 默认值。
 * @param {string} source - 来源描述。
 * @returns {boolean} 布尔值。
 */
function readBoolean(value, field, fallback, source) {
  if (value === undefined) return fallback
  if (typeof value !== 'boolean') {
    throw new Error(`${source}：${field} 必须是 true / false，实际是 ${describe(value)}`)
  }
  return value
}

/**
 * 读枚举值。
 * @param {unknown} value - 原始值。
 * @param {readonly string[]} allowed - 允许取值。
 * @param {string} field - 字段名。
 * @param {string} fallback - 默认值。
 * @param {string} source - 来源描述。
 * @returns {string} 枚举值。
 */
function readEnum(value, allowed, field, fallback, source) {
  if (value === undefined) return fallback
  if (typeof value !== 'string' || !allowed.includes(value)) {
    throw new Error(`${source}：${field} 必须是 ${allowed.join(' / ')} 之一，实际是 ${describe(value)}`)
  }
  return value
}

/**
 * 把任意值渲染成一行短的错误描述。
 * @param {unknown} value - 待描述的值。
 * @returns {string} 描述文本。
 */
function describe(value) {
  if (value === null) return 'null'
  if (Array.isArray(value)) return `数组（${value.length} 项）`
  if (typeof value === 'object') return '映射'
  if (typeof value === 'string') return JSON.stringify(value)
  return String(value)
}

/**
 * 递归冻结一份配置，防止运行期误改默认值。
 * @template T
 * @param {T} value - 待冻结的值。
 * @returns {T} 同一对象。
 */
function deepFreeze(value) {
  if (value !== null && typeof value === 'object') {
    for (const child of Object.values(value)) deepFreeze(child)
    Object.freeze(value)
  }
  return value
}
