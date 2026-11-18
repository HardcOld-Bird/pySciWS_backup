/**
 * 子代理模型路由的纯判定层：把「此刻 + 套餐 + 日历」折算成相位、允许集与默认路由。
 *
 * 本文件不读文件、不碰 DSH 服务，便于用固定时钟做单元测试（计划 §7.1）。峰谷口径
 * 来自前身项目 pySciWS 的 routing.py：工作日两个半开区间为峰；周六日、法定节假日全天为谷
 * （调休上班的周六日仍按谷，`is_in_lieu` 不参与）；年份缺节假日数据时按配置保守。
 *
 * 判定表（计划 §3）：
 *
 * | plan     | 峰                              | 谷                                       |
 * |----------|---------------------------------|------------------------------------------|
 * | mimo     | flash、pro（默认 flash）        | deepseek.flash + pro（默认 deepseek.flash）|
 * | qwen     | flash、pro（默认 flash）        | deepseek.flash + pro（默认 deepseek.flash）|
 * | deepseek | 仅 deepseek.flash（默认它）     | 同峰                                     |
 */

/** 计划可选的套餐 id，与配置里的 `plan` 取值一致。 */
export const PLAN_IDS = Object.freeze(['mimo', 'qwen', 'deepseek'])

/** 相位 id：峰 / 谷。 */
export const PHASE_IDS = Object.freeze(['peak', 'valley'])

/** 本系统只按北京时区读本地时钟，不联网校时。 */
export const BEIJING_TIME_ZONE = 'Asia/Shanghai'

const WEEKDAY_NAMES = Object.freeze(['周日', '周一', '周二', '周三', '周四', '周五', '周六'])

const WEEKDAY_INDEX = Object.freeze({
  Sun: 0,
  Mon: 1,
  Tue: 2,
  Wed: 3,
  Thu: 4,
  Fri: 5,
  Sat: 6,
})

/**
 * 一条精确的 provider/model 路由。
 * @typedef {{ provider: string, model: string }} Route
 */

/**
 * 计算一条路由的稳定标识，用于集合比较。
 * @param {Route} route - 精确路由。
 * @returns {string} 不含歧义的键。
 */
export function routeKey(route) {
  return `${route.provider}\u0000${route.model}`
}

/**
 * 把一条路由渲染成人可读的 `provider/model`。
 * @param {Route} route - 精确路由。
 * @returns {string} 展示用文本。
 */
export function formatRoute(route) {
  return `${route.provider}/${route.model}`
}

/**
 * 判断两条路由是否相同。
 * @param {Route} left - 精确路由。
 * @param {Route} right - 精确路由。
 * @returns {boolean} 是否为同一路由。
 */
export function sameRoute(left, right) {
  return left.provider === right.provider && left.model === right.model
}

/**
 * 展示排序：flash 级恒排在 pro/max 级之前（用户 2026-11-18 拍板）——列举允许集与
 * 授权集时让便宜档先吸走注意力。档位按模型名识别：含 `flash` 为 flash 级，含
 * `pro`/`max` 为 pro 级（`mimo-v2.6-pro`、`qwen-max` 即此类），其余居中；同级保持
 * 原顺序（稳定排序）。`routesFor` 的数据顺序本就 flash 在前，这里兜住「列举」这一层，
 * 让 GUI 持有的会话授权集等外部顺序也归一。
 * @param {ReadonlyArray<Route>} routes - 待排列的路由表。
 * @returns {Route[]} 新的已排列副本（flash 级在前）。
 */
export function flashFirst(routes) {
  return [...routes].sort((left, right) => tierRank(left) - tierRank(right))
}

/**
 * 档位权重：flash=0、其他=1、pro/max=2；只用于 {@link flashFirst} 的稳定排序。
 * @param {Route} route - 精确路由。
 * @returns {number} 权重。
 */
function tierRank(route) {
  if (/flash/i.test(route.model)) return 0
  if (/pro|max/i.test(route.model)) return 2
  return 1
}

/**
 * 用北京时区读出一个时刻的日期、星期与当天分钟数。
 *
 * 用 `Intl` 而不是本机时区推断：本机时区可能被改成别的地区，而峰谷口径固定是
 * 北京时间（计划 §2「时区口径」）。
 * @param {Date} now - 待读的时刻。
 * @returns {{ year: number, month: string, day: string, hour: number, minute: number, weekday: number, date: string, minutes: number }} 北京本地读数。
 */
export function beijingClock(now) {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: BEIJING_TIME_ZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
    weekday: 'short',
  }).formatToParts(now)
  const read = (type) => {
    const part = parts.find((candidate) => candidate.type === type)
    if (part === undefined) throw new Error(`无法从 Intl 读出北京时间的 ${type}`)
    return part.value
  }
  const month = read('month')
  const day = read('day')
  const hour = Number(read('hour'))
  const minute = Number(read('minute'))
  const weekday = WEEKDAY_INDEX[read('weekday')]
  if (weekday === undefined) throw new Error(`无法识别的星期读数：${read('weekday')}`)
  return {
    year: Number(read('year')),
    month,
    day,
    hour,
    minute,
    weekday,
    date: `${read('year')}-${month}-${day}`,
    minutes: hour * 60 + minute,
  }
}

/**
 * 渲染配置里的峰段窗口，供 `basis` 与拒绝文本自证。
 * @param {ReadonlyArray<readonly [number, number]>} peak - 半开区间小时表。
 * @returns {string} 形如 `09:00–12:00、14:00–18:00`。
 */
export function peakWindowLabel(peak) {
  return peak
    .map(([start, end]) => `${String(start).padStart(2, '0')}:00–${String(end).padStart(2, '0')}:00`)
    .join('、')
}

/**
 * 读出某个时刻在日历与配置下的相位（不处理早切，不改写配置）。
 * @param {ReturnType<typeof beijingClock>} clock - 北京本地读数。
 * @param {object} config - 已校验的策略配置。
 * @param {{ hasYear: (year: string) => boolean, holidayName: (date: string) => string | undefined }} calendar - 节假日日历。
 * @returns {{ phase: 'peak' | 'valley', reason: string, warnings: string[] }} 相位、可自证的理由与告警。
 */
function phaseAt(clock, config, calendar) {
  const year = String(clock.year)
  const knownYear = calendar.hasYear(year)
  const warnings = knownYear
    ? []
    : [
        `日历缺 ${year} 年的节假日数据，` +
          (config.unknownYear === 'peak' ? '工作日按峰保守' : '按谷处理'),
      ]
  const holiday = knownYear ? calendar.holidayName(clock.date) : undefined
  if (holiday !== undefined) {
    return { phase: 'valley', reason: `法定节假日（${holiday}），全天谷`, warnings }
  }
  const weekend = clock.weekday === 0 || clock.weekday === 6
  if (weekend) {
    return { phase: 'valley', reason: `${WEEKDAY_NAMES[clock.weekday]}，全天谷`, warnings }
  }
  // 未知年份无法判断工作日是否被法定节假日覆盖，按配置保守处理（周末已在上方判掉，
  // 因为「周六日」只依赖星期算术，与节假日数据无关）。
  if (!knownYear) {
    return {
      phase: config.unknownYear,
      reason: `日历缺 ${year} 年，${config.unknownYear === 'peak' ? '按峰保守' : '按谷处理'}`,
      warnings,
    }
  }
  const label = peakWindowLabel(config.peak)
  const inPeak = config.peak.some(
    ([start, end]) => clock.minutes >= start * 60 && clock.minutes < end * 60,
  )
  return inPeak
    ? { phase: 'peak', reason: `工作日峰段 ${label}`, warnings }
    : { phase: 'valley', reason: `非峰段（峰段 ${label}）`, warnings }
}

/**
 * 按套餐与相位给出允许集与默认路由。
 *
 * 允许集顺序即呈现顺序：默认路由排在最前，其余按「贵 → 便宜」或配置顺序。
 * @param {object} config - 已校验的策略配置。
 * @param {'peak' | 'valley'} phase - 当前相位。
 * @returns {{ allowed: Route[], default: Route }} 允许集（已去重）与默认路由。
 */
export function routesFor(config, phase) {
  const plan = config.plans[config.plan]
  const deepseek = config.plans.deepseek.flash
  if (config.plan === 'deepseek') {
    return { allowed: [cloneRoute(deepseek)], default: cloneRoute(deepseek) }
  }
  if (phase === 'peak') {
    return { allowed: dedupeRoutes([plan.flash, plan.pro]), default: cloneRoute(plan.flash) }
  }
  return { allowed: dedupeRoutes([deepseek, plan.pro]), default: cloneRoute(deepseek) }
}

/**
 * 解析一次派发应有的相位、允许集与默认路由。
 *
 * 早切只做谷→峰：当前是谷而 `now + leadtimeSeconds` 已进峰时，按峰处理；反向不早切
 * （计划 §3）。`forcePhase` 用于测试与手动干预，最后覆盖。
 * @param {{ now: Date, config: object, calendar: object }} input - 时刻、配置与日历。
 * @returns {{ plan: string, phase: 'peak' | 'valley', allowed: Route[], default: Route, basis: string, warnings: string[] }} 解析结果。
 */
export function resolvePolicy({ now, config, calendar }) {
  const clock = beijingClock(now)
  const current = phaseAt(clock, config, calendar)
  let phase = current.phase
  let reason = current.reason
  const warnings = [...current.warnings]

  if (phase === 'valley' && config.leadtimeSeconds > 0) {
    const laterClock = beijingClock(new Date(now.getTime() + config.leadtimeSeconds * 1000))
    const later = phaseAt(laterClock, config, calendar)
    if (later.phase === 'peak') {
      phase = 'peak'
      reason = `${reason}；谷末早切：${config.leadtimeSeconds} 秒后进入峰段（${peakWindowLabel(config.peak)}）`
      warnings.push(...later.warnings)
    }
  }

  if (config.forcePhase !== 'auto') {
    phase = config.forcePhase
    reason = `forcePhase=${phase} 强制覆盖真实相位（${reason}）`
  }

  const basis =
    `${WEEKDAY_NAMES[clock.weekday]} ${String(clock.hour).padStart(2, '0')}:` +
    `${String(clock.minute).padStart(2, '0')}，北京时间；${reason}`
  const { allowed, default: fallback } = routesFor(config, phase)
  return {
    plan: config.plan,
    phase,
    allowed,
    default: fallback,
    basis,
    warnings: [...new Set(warnings)],
  }
}

/**
 * 复制一条路由，避免调用方改到配置内的对象。
 * @param {Route} route - 原始路由。
 * @returns {Route} 副本。
 */
function cloneRoute(route) {
  return { provider: route.provider, model: route.model }
}

/**
 * 按路由键去重，保留首次出现的顺序。
 * @param {Route[]} routes - 候选路由。
 * @returns {Route[]} 去重后的副本。
 */
function dedupeRoutes(routes) {
  const seen = new Set()
  const result = []
  for (const route of routes) {
    const key = routeKey(route)
    if (seen.has(key)) continue
    seen.add(key)
    result.push(cloneRoute(route))
  }
  return result
}
