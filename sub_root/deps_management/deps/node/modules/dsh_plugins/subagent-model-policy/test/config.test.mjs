/**
 * 配置校验与热读测试（计划 §5.4 的「非法值直接 throw」与按 mtime 热读）。
 */
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { test } from 'node:test'
import {
  CONFIG_FILE_NAME,
  DEFAULT_CONFIG,
  createConfigStore,
  normalizeConfig,
  resolveConfigCandidates,
  resolveConfigPath,
} from '../config.js'

test('缺省配置等于内置默认值', () => {
  assert.deepEqual(normalizeConfig(undefined), JSON.parse(JSON.stringify(DEFAULT_CONFIG)))
  assert.deepEqual(normalizeConfig(null), JSON.parse(JSON.stringify(DEFAULT_CONFIG)))
  assert.deepEqual(normalizeConfig({}), JSON.parse(JSON.stringify(DEFAULT_CONFIG)))
})

test('plans 逐层合并：只覆盖给出的字段', () => {
  const config = normalizeConfig({
    plans: { mimo: { pro: { provider: 'p', model: 'm' } } },
  })
  assert.deepEqual(config.plans.mimo.pro, { provider: 'p', model: 'm' })
  assert.deepEqual(config.plans.mimo.flash, DEFAULT_CONFIG.plans.mimo.flash)
  assert.deepEqual(config.plans.deepseek.flash, DEFAULT_CONFIG.plans.deepseek.flash)
})

test('非法值一律 throw，不回退', () => {
  const cases = [
    [{ plann: 'mimo' }, /未知配置键 "plann"/],
    [{ plan: 'gpt' }, /plan 必须是 mimo \/ qwen \/ deepseek 之一/],
    [{ plans: { mimo: { flash: { provider: 'p' } } } }, /plans\.mimo\.flash\.model 必须是非空字符串/],
    [{ plans: { mimo: { flash: { provider: 'p', model: 'm', extra: 1 } } } }, /未知键 "extra"/],
    [{ plans: { other: {} } }, /plans 出现未知套餐 "other"/],
    [{ peak: [] }, /peak 必须是非空数组/],
    [{ peak: [[12, 9]] }, /0 ≤ 开始 < 结束 ≤ 24/],
    [{ peak: [[9, 12], [11, 14]] }, /peak 窗口重叠/],
    [{ peak: [[9.5, 12]] }, /0–24 的整数/],
    [{ leadtimeSeconds: -1 }, /非负整数秒/],
    [{ unknownYear: 'sometimes' }, /unknownYear 必须是 peak \/ valley 之一/],
    [{ forcePhase: 'always' }, /forcePhase 必须是 auto \/ peak \/ valley 之一/],
    [{ guardFork: 'yes' }, /guardFork 必须是 true \/ false/],
  ]
  for (const [raw, pattern] of cases) {
    assert.throws(() => normalizeConfig(raw, { source: 'test' }), pattern, JSON.stringify(raw))
  }
})

test('配置路径：环境变量优先，其余推导落在 sub_root', () => {
  const env = { PYSCI_SUBAGENT_MODEL_POLICY: 'C:/tmp/policy.yaml' }
  assert.deepEqual(
    resolveConfigCandidates({ env, pluginDir: 'C:/plugins/p', dshHome: 'C:/modules/dsh_home' }),
    [resolve('C:/tmp/policy.yaml')],
  )
  const derived = resolveConfigCandidates({
    env: {},
    pluginDir: join('C:', 'repo', 'sub_root', 'deps_management', 'deps', 'node', 'modules', 'dsh_plugins', 'p'),
    dshHome: join('C:', 'repo', 'sub_root', 'deps_management', 'deps', 'node', 'modules', 'dsh_home'),
  })
  assert.equal(derived.length, 1) // 两条推导殊途同归，去重后只剩一条
  assert.ok(derived[0].endsWith(CONFIG_FILE_NAME))
  assert.ok(derived[0].includes(join('repo', 'sub_root')))
})

test('配置缺失时报显式错误', () => {
  const dir = mkdtempSync(join(tmpdir(), 'subagent-policy-'))
  try {
    const store = createConfigStore(join(dir, 'missing.yaml'))
    assert.throws(() => store.read(), /未找到策略配置文件/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('按 mtime 热读：改完下一次读取即生效', () => {
  const dir = mkdtempSync(join(tmpdir(), 'subagent-policy-'))
  try {
    const file = join(dir, 'policy.yaml')
    writeFileSync(file, 'plan: mimo\n', 'utf8')
    const store = createConfigStore(file)
    assert.equal(store.read().plan, 'mimo')
    // 内容长度不同，确保 size 变化触发重读（mtime 分辨率在部分文件系统上很粗）。
    writeFileSync(file, 'plan: deepseek\n# 触发重读\n', 'utf8')
    assert.equal(store.read().plan, 'deepseek')
    // 非法内容在下一次读取时抛错，而不是沿用旧值。
    writeFileSync(file, 'plan: nope\n# 非法\n', 'utf8')
    assert.throws(() => store.read(), /plan 必须是/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
