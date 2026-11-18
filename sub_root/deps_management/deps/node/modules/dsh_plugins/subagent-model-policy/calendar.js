/**
 * 节假日日历：把 chinese-days 的数据文件折叠成「某日期是否法定节假日」的查询面。
 *
 * 数据来自社区包 `chinese-days`（MIT，见 `data/SOURCE.md`），本插件只消费其中的
 * `holidays` 映射：判定规则只需要「这一天是不是法定节假日」，而 `workdays` /
 * `inLieuDays` 按计划 §2 的口径刻意不参与——调休上班的周六日仍按谷。
 *
 * 未知年份（数据未覆盖）不在这里兜底：相位判定在 `policy.js` 里按
 * `unknownYear` 保守处理并告警，日历只回答「我知道哪些年」。
 */
import { readFileSync } from 'node:fs'

/**
 * 读取并校验节假日数据文件。
 * @param {string} filePath - `data/chinese-days.json` 的绝对路径。
 * @returns {{ version: string, holidays: Record<string, string> }} 版本与「YYYY-MM-DD → 原始标签」映射。
 */
export function loadHolidayData(filePath) {
  let text
  try {
    text = readFileSync(filePath, 'utf8')
  } catch (error) {
    throw new Error(`读取节假日数据失败：${filePath}（${error.code ?? error.message}）`)
  }
  let data
  try {
    data = JSON.parse(text)
  } catch (error) {
    throw new Error(`节假日数据不是合法 JSON：${filePath}（${error.message}）`)
  }
  const holidays = data?.holidays
  if (holidays === null || typeof holidays !== 'object' || Array.isArray(holidays)) {
    throw new Error(`节假日数据缺少 holidays 对象：${filePath}`)
  }
  for (const date of Object.keys(holidays)) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      throw new Error(`节假日数据含非法日期键 "${date}"：${filePath}`)
    }
  }
  return { version: String(data.version ?? 'unknown'), holidays }
}

/**
 * 从上游标签里取中文短名，用于拒绝文本与日志。
 *
 * 上游值是三段式 `英文名,中文名,天数`（如 `National Day,国庆节,3`）；取不到中文名时
 * 退回整串，保证日志仍然可读。
 * @param {string} label - 上游标签。
 * @returns {string} 中文短名。
 */
export function holidayName(label) {
  if (typeof label !== 'string') return String(label)
  const parts = label.split(',')
  const name = (parts[1] ?? parts[0] ?? '').trim()
  return name.length > 0 ? name : label
}

/**
 * 建立日历查询面。
 * @param {{ version: string, holidays: Record<string, string> }} data - `loadHolidayData` 的返回值。
 * @returns {{ version: string, hasYear: (year: string | number) => boolean, holidayName: (date: string) => string | undefined, years: ReadonlySet<string> }} 日历。
 */
export function createCalendar(data) {
  const years = new Set()
  const holidays = new Map()
  for (const [date, label] of Object.entries(data.holidays)) {
    years.add(date.slice(0, 4))
    holidays.set(date, holidayName(label))
  }
  return {
    version: data.version,
    years,
    hasYear: (year) => years.has(String(year)),
    holidayName: (date) => holidays.get(date),
  }
}
