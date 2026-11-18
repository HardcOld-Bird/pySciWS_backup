# Java/mph 坑点册（COMSOL 6.4 + mph 1.4.0 + JPype1 实测）

> 全部事实实机验证（旧 pySciWS 2026-09 beam-scatter 任务 + 本工作区 2026-11-17
> hardware 冒烟）。遇到「未知类型/未知属性/结果不对」：先 `mph.inspect(node)` 拿
> ground truth，或 COMSOL-MCP `pdf_search` 查手册；不要手写一次性探针脚本（用完即删）。

## 1. 会话/许可证管理约定（单证机器，最高优先）

- **显式 start 才占证**：mph `session.start_client()` / COMSOL-MCP `comsol_start`；
  「Nothing starts implicitly」——空闲 MCP server / 仅 import mph 不占证。
- **JPype JVM 启动后不可卸载**：`client.disconnect()` 只清模型会话，**不放许可证**；
  释放许可证的唯一途径是**退出 Python 进程**。批任务按进程级编排；一次 Python 进程
  只允许一个 mph Client 单例（`mph.start()` 第二次调用返回同一实例）。
- **占证探测判据 = `jvm.dll` 加载**（JVM 真启动）；`_jpype.pyd` 仅表示 import 过
  jpype——**误报源**，不得用作判据（`session.probe_license_holders()` 实现此口径）。
- 多代理机注意：别的 JVM 进程（JetBrains/他代理 MCP server）也带 `jvm.dll` 但未必
  占 COMSOL 证；hardware 冒烟只拦 `comsol*` 进程，许可证真被占时 start 报错转 skip。
- **本机（2026-11-17）会话模式用 stand-alone**：mph ≥1.3 默认 client–server 后端，
  `comsolmphserver -port 0` 起服务即炸 `Internal error. Invalid port: 65536`
  （COMSOL getNextAvailablePortNumber 溢出）；`mph.option('session', 'stand-alone')`
  （Windows 文档支持）→ `session.start_client()` 默认此模式。要 client-server 须
  **显式传空闲端口**（mph `Server(port=…)`，勿用 0/None 扫描路径）。

## 2. 版本串与发现

- mph discovery 版本串要**短名** `"6.4"`：完整串 `"6.4.0"`/`"6.4.0.293"` 会
  exact-match 失败 → `session.normalize_version()` 归一（`"6.4.0.293"→"6.4"`）。
- Windows 发现走注册表 `HKLM\SOFTWARE\Comsol\COMSOL64\COMSOLROOT` + `where comsol`
  兜底；本机 COMSOL 6.4.0.293 已迁项目内 `deps_management/deps/comsol/6.4/base`
  （注册表已同步）——非默认路径可发现。

## 3. 本机环境坑（2026-11-17 实测）

- **沙盒 Low 完整性令牌**：DSH 派生的子进程（python/JVM）跑 Low IL，写不了 Medium
  完整性的用户目录——COMSOL 用户目录 `%USERPROFILE%\.comsol\v64`（`comsol.recoveries`
  等）报「拒绝访问」，与 ACL 无关（Mandatory Integrity Control 挡写高完整性对象）。
  修复：`icacls "C:\Users\<user>\.comsol" /setintegritylevel "(OI)(CI)Low"`
  （回滚换回 Medium）。改 USERPROFILE 无效（COMSOL 走 WinAPI 取 profile 路径）。
- COMSOL 用户目录里的 `comsol.recoveries` 文件/目录形态混见过；`icacls` 显示
  完全控制却仍拒绝访问时，先想完整性级别，别怀疑 ACL。

## 4. 静默失败类（不报错但结果错，最危险）

- **材料缺 `c`/密度 ⇒ 求解静默空解**：study.run() 不抛错，`model.evaluate` 返回
  **零长数组**，导出 PNG 只剩轴框。建模必给材料（`soundspeed`+`density`）。
- **绘图组数据集绑定**：solve 前落盘 / reload 后重跑 geom/mesh/study，绘图组失去
  （或从未建立）dataset 绑定 → 空白 PNG。重求解后必
  `plotgroup.rebind_plot_groups(model)`（薄胶水已自动化；空白自检兜底告警）。
- **BackgroundPressureField 默认空选择** → p_b=0 全场 0；必须
  `bpf.java.selection().all()`。其 `PressureFieldType`/`k_src`/`c_mat` 缺省为
  **未定值** → 残差 NaN（`找到未定义的值`）；配方见 modeling-recipes §3。
- **网格序列要显式建**（`(model/'meshes').create(geometry, name='mesh')`）——
  不建序列直接 solve 也可能出空解。
- blank 自检只采**绘图区中心区**：轴框/刻度/标签混入采样会把「只剩轴框」误判为有
  内容（2026-11-17 实测教训；`export.blank_check(central=0.6)`）。

## 5. API 形态坑

- **solve 按名不按 tag**：`model.solve('study')` 走 `studies/<名>` 查找，Java tag
  常常找不到。evaluate 数据集同理用**名字**（注意名字本地化：中文机是「解 1」）。
- **显示名随界面语言本地化**：材料属性组标签中文机不叫 "Basic"——按 **tag** 走
  Java 侧：`node.java.propertyGroup('def')`。类型字符串（'Rectangle'/'Image'/
  'PressureAcoustics'）与属性名（`soundspeed`/`plist`）**不**本地化。
- **mph Node 写属性是 `node.property(name, value)`**（不是 Java 的 `set`）；
  `node.java` 是全量 Java API 逃生通道（`.selection().all()` 等）。
- Python int 传 Java `set(String, int)` 重载歧义 → `jpype.JInt(v)` 包裹（JVM 未
  启动时 JInt 不可用；`features._jint` 已处理）。
- 导出节点 tag 必须唯一：同 tag 重复 create 抛异常；薄胶水用 `img_<输出 stem>`
  幂等复用（存在即取回更新属性）。
- COMSOL 导出 CSV 有多行 `%` 头：先过滤 `%` 行再跳列名行再解析。

## 6. 几何/结果接口边界

- **PlotGroup2D 无轴限属性**（xmin/xmax… Java API 不暴露，GUI 轴限是内部 zoom 状态）
  → `--extent` 对 2D 无效；精确 pixel↔data 映射用 **auto-zoom 反演**：
  COMSOL 2D auto-zoom = 等纵横比、以几何 bbox 中心为中心、按轴框像素框长宽比在
  宽/高受限方向展开 ⇒ `sidecar.comsol_auto_window(geom_bbox, crop_box_px)`。
- PolarGroup 无矩形轴框，不适用 extent/crop 反演。
- `ExportFeatureListClient` 不可调用、无 `.feature()`；导出节点用分组
  `(model/'exports').create('Image', name=…)` 或按名取回。

## 7. 卫生约定

- 一次性诊断脚本命名 `_probe_*.py`，用完即删，不留仓库。
- 路径一律 `pysci.paths`；禁用 `Path(__file__).parents[N]` 层级猜测。
- 保存模型/产物后打印绝对路径核对；产物落 `data/`（`assert_within_data` 护栏
  默认开，测试产景 `guard_data=False`）。
