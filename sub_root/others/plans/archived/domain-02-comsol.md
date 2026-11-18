# ②COMSOL 仿真域实施计划（状态：草案，待用户确认）

> 三段式第二段产物。输入：`community_research/2026-11-17/domain-02-comsol-selection.md`（含用户裁决记录）。
> 完成后归档至 `archived/`。

## 1. 范围

**做**：②域"技能文档 + 社区工具 + 薄胶水 + 机械断言"；COMSOL-MCP 挂 DSH；手册 RAG 喂入。
**不做**（按裁决）：imgcheck（砍）、pyvista 渲染（归③）、自写 JPype 层（=自研 mph，拒绝）、
FEniCSx 换范式（拒绝）、LiveLink MATLAB（商业绑定）。

## 2. 选型定稿

| 角色 | 引入 | 口径/豁免 |
|---|---|---|
| COMSOL Python 通道 | **mph==1.4.0** + JPype1==1.7.1 | 差②**无可替代豁免**（官方无 Python API），pin+ADR |
| Agent 层 | **COMSOL-MCP**（wjc9011，981★ MIT，101 工具）**pin commit**（源仓浅克隆入 `deps/repos/`，同 paper-search-mcp 先例） | 差②临界**豁免引入**（用户裁决） |
| 手册知识 | COMSOL-MCP **pdf_search RAG**（手册 PDF 喂入） | 零额外自研 |
| 外部软件 | 本机 COMSOL 6.4（`D:\XiGPrograms\comsol\6.4`，注册表 COMSOLROOT 自发现） | 用户既有资产 |

**薄胶水 ≈800-1200 行**（差异化保留）：出版级 export+sidecar 生产端 300-500、feature-types 探测+node set
~150、plot-group 自动重绑 ~50、验证函数族 100-200、domain-check ~100、sidecar 消费端 ~200。

## 3. 目录规划（新增文件）

| 内容 | 位置 |
|---|---|
| 技能文档 | `.agents\skills\comsol-simulation\SKILL.md` + `references\`（建模配方/Java 坑点/声明模板/出图契约） |
| 薄胶水 | `sub_root\src\pysci\skills\comsol_simulation\`（export.py、sidecar.py、features.py、plotgroup.py、validate.py、session.py、\_\_init\_\_.py） |
| 内容资产 | `sub_root\src\pysci\skills\comsol_simulation\assets\`（声明模板等） |
| 测试 | `sub_root\tests\skills\comsol_simulation\`（mock mph 机械断言恒跑；真实 COMSOL 求解/出图标 `-m hardware`） |
| COMSOL-MCP 源 | `sub_root\deps_management\deps\repos\comsol-mcp\`（浅克隆 pin commit） |
| 手册 RAG 数据 | `sub_root\data\skills\comsol_simulation\manual\`（手册 PDF/索引，不入仓） |
| 运行产物 | `sub_root\data\research\<题目>\`（研究线，不入仓） |

## 4. 实施步骤

1. 依赖引入：mph==1.4.0、JPype1==1.7.1（pyproject 独占编辑归属实施代理）。
2. COMSOL-MCP：浅克隆 pin commit → `dsh plugin add link:`（同 paper-search-mcp 先例）→ cordis.patch.yml
   `- insert:` 条目 → tools/list 冒烟；手册 PDF 喂入其 RAG。
3. 技能文档：建模→求解→后处理→出版级出图→sidecar→验证 全工作流；含旧坑点（版本串短名"6.4"、
   JPype JVM 不卸载、disconnect 不放证、logo*d=off 品牌剥离）。
4. 薄胶水六模块（预算见上）；自研理由原地 ADR；出版级出图契约与③域导出契约风格一致。
5. 机械断言：mock mph 单测恒跑；真实求解冒烟（简单模型 solve+出图+sidecar）标 hardware 实跑。
6. 冒烟：真实 COMSOL 6.4 跑通"建模→solve→出版级出图→sidecar→domain-check"，产物目检（视觉策略）。

## 5. 风险与 ADR

- **环境记录（2026-11-17）**：COMSOL 6.4 已迁移项目内自包含 `deps\comsol\6.4\base`（用户裁决补充），
  注册表 `HKLM\SOFTWARE\Comsol\COMSOL64\COMSOLROOT` 已由用户提权同步至新路径；mph 发现→启动→
  模型列表实测全通。

- mph 差②豁免（发版驱动）→ pin 1.4.0，升级前跑真实冒烟；
- COMSOL-MCP 差②临界豁免 → pin commit（非版本），记录评估日期；
- COMSOL 许可证/单证会话：COMSOL-MCP 显式 comsol_start 才占证；JPype disconnect 不放证——技能文档写明会话管理约定；
- COMSOL 6.4 版本串短名坑、registry 发现非默认路径——实测注记。

## 6. 验收标准

真实模型（含 PDE/参数扫掠其一）：建模→solve→出版级出图（尺寸契约/品牌剥离/字体）→sidecar→domain-check
全通；③域 truthful colorbar 消费端可读 sidecar 叠加；COMSOL-MCP tools/list + 手册 RAG 查询可用；
`tests.cmd all` 全绿；归档。

## 7. 实施结果（2026-11-17 收尾记录）

- 验收标准**全部达成**：真实 COMSOL 声学模型（1kHz 平面波）建模→solve→出版级出图→sidecar→
  domain-check 实跑通过且目检；**③域消费端叠加实证**（overlay_demo：红色旋转矩形逐角映射无畸变、
  绿点锚定数据坐标正确）；COMSOL-MCP **135 工具**挂 DSH 热加载生效；手册 RAG 查询命中官方手册
  （全量索引后台续建，查询已可用，重建命令见 SOURCE.md）。
- 薄胶水 8 模块（逻辑 ≈900 行 + 文档/ADR ≈750）+ CMT 对账 reconcile.py（44 项测试）；测试合计
  469 passed / 13 hardware（真实冒烟另过）。
- 环境实证入坑点册：.comsol Low 完整性（icacls 修复）、mph stand-alone 口径（client-server -port 0
  本机坏）、静默空解四件套、COMSOL-MCP pin commit 53c9ca4（差②临界豁免+ADR）。
