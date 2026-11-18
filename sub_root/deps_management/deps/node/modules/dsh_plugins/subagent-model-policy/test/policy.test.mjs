/**
 * 判定层测试：固定时钟 + 真实日历数据，不需要 DSH 运行时（计划 §7.1）。
 */
import assert from 'node:assert/strict'
import { join } from 'node:path'
import { test } from 'node:test'
import { createCalendar, loadHolidayData } from '../calendar.js'
import { normalizeConfig } from '../config.js'
import { beijingClock, flashFirst, formatRoute, resolvePolicy } from '../policy.js'

const CALENDAR = createCalendar(
  loadHolidayData(join(import.meta.dirname, '..', 'data', 'chinese-days.json')),
)

/**
 * 在某个固定时刻解析策略。
 * @param {string} iso - 带时区偏移的时刻。
 * @param {object} [patch] - 配置覆盖。
 * @returns {ReturnType<typeof resolvePolicy>} 解析结果。
 */
function resolveAt(iso, patch = {}) {
  return resolvePolicy({ now: new Date(iso), config: normalizeConfig(patch), calendar: CALENDAR })
}

const DEEPSEEK_FLASH = 'deepseek-account/deepseek-flash'
const MIMO_FLASH = 'xiaomi-token-plan-cn/mimo-v2.6-flash'
const MIMO_PRO = 'xiaomi-token-plan-cn/mimo-v2.6-pro'

test('判定矩阵：3 套餐 × 峰 / 谷', () => {
  const cases = [
    {
      plan: 'mimo',
      phase: 'peak',
      allowed: [MIMO_FLASH, MIMO_PRO],
      fallback: MIMO_FLASH,
    },
    {
      plan: 'mimo',
      phase: 'valley',
      allowed: [DEEPSEEK_FLASH, MIMO_PRO],
      fallback: DEEPSEEK_FLASH,
    },
    {
      plan: 'qwen',
      phase: 'peak',
      allowed: ['qwen-token-plan-cn/qwen-flash', 'qwen-token-plan-cn/qwen-max'],
      fallback: 'qwen-token-plan-cn/qwen-flash',
    },
    {
      plan: 'qwen',
      phase: 'valley',
      allowed: [DEEPSEEK_FLASH, 'qwen-token-plan-cn/qwen-max'],
      fallback: DEEPSEEK_FLASH,
    },
    {
      plan: 'deepseek',
      phase: 'peak',
      allowed: [DEEPSEEK_FLASH],
      fallback: DEEPSEEK_FLASH,
    },
    {
      plan: 'deepseek',
      phase: 'valley',
      allowed: [DEEPSEEK_FLASH],
      fallback: DEEPSEEK_FLASH,
    },
  ]
  for (const item of cases) {
    const resolved = resolveAt('2026-11-09T10:00:00+08:00', {
      plan: item.plan,
      forcePhase: item.phase,
    })
    assert.equal(resolved.plan, item.plan)
    assert.equal(resolved.phase, item.phase)
    assert.deepEqual(resolved.allowed.map(formatRoute), item.allowed, item.plan + '/' + item.phase)
    assert.equal(formatRoute(resolved.default), item.fallback, item.plan + '/' + item.phase)
  }
})

test('峰谷边界、周末与法定节假日（关闭早切以单测窗口本身）', () => {
  const at = (iso) => resolveAt(iso, { leadtimeSeconds: 0 })
  assert.equal(at('2026-11-09T09:00:00+08:00').phase, 'peak')
  assert.equal(at('2026-11-09T11:59:00+08:00').phase, 'peak')
  assert.equal(at('2026-11-10T12:00:00+08:00').phase, 'valley') // 半开区间：12:00 属谷
  assert.equal(at('2026-11-09T13:59:00+08:00').phase, 'valley')
  assert.equal(at('2026-11-09T14:00:00+08:00').phase, 'peak')
  assert.equal(at('2026-11-09T18:00:00+08:00').phase, 'valley') // 半开区间：18:00 属谷
  assert.equal(at('2026-11-14T10:00:00+08:00').phase, 'valley') // 周六
  assert.equal(at('2026-11-15T10:00:00+08:00').phase, 'valley') // 周日
  assert.equal(at('2026-10-01T10:00:00+08:00').phase, 'valley') // 国庆节
  assert.equal(at('2026-10-10T10:00:00+08:00').phase, 'valley') // 调休上班的周六仍按谷
  assert.match(at('2026-10-01T10:00:00+08:00').basis, /法定节假日（国庆节），全天谷/)
})

test('早切只做谷→峰，反向不早切', () => {
  // 08:55 + 600s = 09:05，进峰 → 按峰处理。
  const early = resolveAt('2026-11-09T08:55:00+08:00')
  assert.equal(early.phase, 'peak')
  assert.match(early.basis, /谷末早切/)
  // 11:55 仍是峰；+600s 落在谷，不反向早切。
  assert.equal(resolveAt('2026-11-09T11:55:00+08:00').phase, 'peak')
  // 13:55 + 600s = 14:05，进下午峰段。
  assert.equal(resolveAt('2026-11-09T13:55:00+08:00').phase, 'peak')
  // 17:55 + 600s = 18:05，落在谷，不提前切谷。
  assert.equal(resolveAt('2026-11-09T17:55:00+08:00').phase, 'peak')
  // 提前量为 0 时关闭早切。
  assert.equal(resolveAt('2026-11-09T08:55:00+08:00', { leadtimeSeconds: 0 }).phase, 'valley')
  // 周六不会因为早切变成峰。
  assert.equal(resolveAt('2026-11-14T08:55:00+08:00').phase, 'valley')
})

test('未知年份按 unknownYear 保守并告警', () => {
  const peak = resolveAt('2028-03-15T10:00:00+08:00')
  assert.equal(peak.phase, 'peak')
  assert.ok(peak.warnings.some((warning) => warning.includes('2028')), peak.warnings.join('|'))
  assert.match(peak.basis, /日历缺 2028 年/)

  const valley = resolveAt('2028-03-15T10:00:00+08:00', { unknownYear: 'valley' })
  assert.equal(valley.phase, 'valley')
})

test('按北京时区读时钟，不受本机时区影响', () => {
  const clock = beijingClock(new Date('2026-11-12T02:12:00Z')) // 北京 10:12
  assert.equal(clock.date, '2026-11-12')
  assert.equal(clock.hour, 10)
  assert.equal(clock.minute, 12)
  assert.equal(clock.weekday, 4) // 周四
  assert.equal(clock.minutes, 612)
})

test('basis 自证：星期、时刻、峰段窗口', () => {
  const resolved = resolveAt('2026-11-12T10:12:00+08:00', { leadtimeSeconds: 0 })
  assert.equal(
    resolved.basis,
    '周四 10:12，北京时间；工作日峰段 09:00–12:00、14:00–18:00',
  )
})

test('flashFirst：flash 级恒在 pro/max 级之前，其余居中，同级稳定且不改写输入', () => {
  const routes = [
    { provider: 'qwen-token-plan-cn', model: 'qwen-max' },
    { provider: 'xiaomi-token-plan-cn', model: 'mimo-v2.6-pro' },
    { provider: 'other', model: 'unknown-tier' },
    { provider: 'deepseek-account', model: 'deepseek-flash' },
    { provider: 'xiaomi-token-plan-cn', model: 'mimo-v2.6-flash' },
  ]
  assert.deepEqual(flashFirst(routes).map(formatRoute), [
    'deepseek-account/deepseek-flash',
    'xiaomi-token-plan-cn/mimo-v2.6-flash',
    'other/unknown-tier',
    'qwen-token-plan-cn/qwen-max',
    'xiaomi-token-plan-cn/mimo-v2.6-pro',
  ])
  // 输入顺序不被改写（返回的是副本）。
  assert.deepEqual(routes.map(formatRoute), [
    'qwen-token-plan-cn/qwen-max',
    'xiaomi-token-plan-cn/mimo-v2.6-pro',
    'other/unknown-tier',
    'deepseek-account/deepseek-flash',
    'xiaomi-token-plan-cn/mimo-v2.6-flash',
  ])
})
