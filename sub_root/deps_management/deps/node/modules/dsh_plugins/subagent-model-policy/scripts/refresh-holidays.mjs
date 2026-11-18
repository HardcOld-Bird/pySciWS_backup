#!/usr/bin/env node
/**
 * 刷新节假日数据：抓取 chinese-days 的上游 JSON，只保留法定节假日并覆盖
 * `data/chinese-days.json`。
 *
 * 为什么是人工年度动作而不是安装脚本：节假日安排每年由国务院办公厅公布，上游随后
 * 更新；数据过期不会让判定出错，只会让未知年份按 `unknownYear` 保守处理。因此刷新
 * 不需要联网能力常驻，也不需要构建期依赖。
 *
 * 用法（在插件目录执行）：`node scripts/refresh-holidays.mjs [版本号]`
 * 数据形状与来源说明见 `data/SOURCE.md`。
 */
import { writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { holidayName } from '../calendar.js'

const DEFAULT_VERSION = '1.5.10'
const version = process.argv[2] ?? DEFAULT_VERSION
const upstream = `https://cdn.jsdelivr.net/npm/chinese-days@${version}/dist/chinese-days.json`

const response = await fetch(upstream)
if (!response.ok) throw new Error(`下载失败：${upstream} → HTTP ${response.status}`)
const payload = await response.json()
const source = payload?.holidays
if (source === null || typeof source !== 'object' || Array.isArray(source)) {
  throw new Error(`上游数据缺少 holidays 映射：${upstream}`)
}

const dates = Object.keys(source).sort()
const holidays = {}
for (const date of dates) holidays[date] = holidayName(source[date])

const data = {
  source: 'chinese-days',
  version,
  license: 'MIT',
  upstream,
  fetchedAt: new Date().toISOString().slice(0, 10),
  note: '只保留法定节假日日期；workdays / inLieuDays 不参与判定（调休上班的周六日仍按谷），见 data/SOURCE.md。',
  holidays,
}
const target = join(import.meta.dirname, '..', 'data', 'chinese-days.json')
await writeFile(target, `${JSON.stringify(data, null, 2)}\n`, 'utf8')

const years = [...new Set(dates.map((date) => date.slice(0, 4)))]
console.log(
  `已写入 ${target}：${dates.length} 个节假日日期，覆盖 ${years[0]}–${years[years.length - 1]}。`,
)
