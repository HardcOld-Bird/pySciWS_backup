# ⑥3D 建模 + ⑦AI 生图 实施计划（状态：草案，待用户确认）

> 三段式第二段产物。输入：`community_research/2026-11-17/domain-06-3d-selection.md`、
> `domain-07-imagegen-selection.md`（各附用户裁决记录）。完成后归档至 `archived/`。
> **本计划完成后七域全齐**，随后按既有裁决统一建 eval 基准层。

## 1. 范围与关键裁决

**⑥域（用户裁决 B：纯 Blender）**：bpy 无头全链主力（建模/Cycles CPU 渲染/GLB-STL-OBJ 导出）；
**build123d/trimesh/CadQuery/gmsh/FreeCAD/OpenSCAD/meshio 全不引入**（STEP 需求出现再立项）。
**⑦域**：云 API 优先（方舟 Seedream 主 + 硅基流动副，key 在 `.env` 但 ARK 已过期——真实出图冒烟
挂起待 key 更新）；本地纯后处理（Pillow + rembg 兜底透明底）；measure/align 留⑦ imaging。

## 2. 选型定稿

### ⑥域
| 角色 | 引入 | 口径/豁免 |
|---|---|---|
| 主链 | **Blender 无头**（vendor 至 `deps\blender`，自包含；Cycles CPU 渲染——EEVEE 需 GPU 不可用） | 22k★ 7/7 |
| Agent 通道 | **blender-mcp**（35k★ MIT）pin | 差②（5 月缺）**豁免**：agent↔Blender 唯一通道，ADR |
| 云生成 | **tripo3d SDK**（官方，MIT，v3）pin | 差①（star 61）**豁免**：官方出品，ADR；定位原型/示意 |
| 薄胶水 | 脚手架/验收门红线/doctor/交付物冻结回脚本（保旧形态） | 旧设计精华 |

### ⑦域
| 角色 | 引入 | 口径/豁免 |
|---|---|---|
| 云主通道 | 方舟 Seedream（OpenAI 兼容 images API） | key 过期，客户端实现+静态测试 |
| 云副通道 | 硅基流动（Qwen-Image/FLUX 系） | 同上 |
| 后处理 | Pillow + **rembg**（26k★） | 透明底兜底/任意图抠图 |
| 薄胶水 | 双 provider requests 客户端 + 出图护栏（watermark=false/seed/b64）+ 生成账本（PNG tEXt 溯源）+ imaging（measure/align） | ≈400-600 行 |

## 3. 目录规划（新增文件）

| 内容 | 位置 |
|---|---|
| ⑥技能文档 | `.agents\skills\modeling-3d\SKILL.md` + `references\`（bpy 配方/Cycles CPU 须知/验收门红线/打印渲染不混） |
| ⑥薄胶水 | `sub_root\src\pysci\skills\modeling_3d\`（scaffold.py 脚手架、acceptance.py 验收门、doctor.py、ledger.py 交付冻结） |
| ⑦技能文档 | `.agents\skills\image-generation\SKILL.md` + `references\`（prompt 配方/出图护栏/透明底策略） |
| ⑦薄胶水 | `sub_root\src\pysci\skills\image_generation\`（providers.py 双云客户端、guardrails.py、ledger.py 账本、imaging.py measure/align、postprocess.py） |
| Blender | `sub_root\deps_management\deps\blender\`（vendor，同 mySimsWS2 先例/共享 shim 源） |
| MCP | blender-mcp 入 venv/uv tool + DSH profile（同既有先例）；tripo-mcp 不引 |
| 测试 | `tests\skills\modeling_3d\`、`tests\skills\image_generation\`（mock API 恒跑；Blender 渲染标 hardware） |
| 运行产物 | `sub_root\data\research\<题目>\`（不入仓） |

## 4. 实施步骤

1. Blender vendor（复制 mySimsWS2 `deps\blender` 同版本或按共享 shim 源装）+ `blender --version` 验证；
2. 依赖引入（独占 pyproject）：blender-mcp（pin+豁免 ADR）、tripo3d（pin+豁免 ADR）、rembg、Pillow（已有）、skimage/opencv extra（imaging 需要时）；
3. ⑥技能+薄胶水：bpy 无头工作流（建模→Cycles CPU 渲染→GLB/STL 导出→验收门→账本）；Tripo 云生成接 SDK（原型/示意口径写死）；
4. ⑦技能+薄胶水：双云客户端（OpenAI 兼容 images API 统一封装）、出图护栏、生成账本（PNG tEXt）、rembg 后处理、imaging measure/align；
5. 测试：mock API/纯函数恒跑；Blender 渲染 `-m hardware`（CPU 实渲一帧）；**⑦真实出图冒烟挂起**（key 过期，留 TODO 与冒烟脚本）；
6. 冒烟：⑥ Blender 真实建模→Cycles CPU 渲染→导出目检；⑦静态链冒烟（mock 出图→护栏→账本→rembg）。

## 5. 风险与 ADR

- blender-mcp 差②豁免、tripo3d 差①豁免——ADR 记录；
- Cycles CPU 渲染慢（内存 ~5GB 约束：单帧低采样，禁止并行多实例）；
- Tripo v2 已关停——**只接 v3**；旧客户端不迁移；
- ⑦真实 API 行为（方舟 5.5 参数/价目）未实测——客户端按官方文档实现+护栏，冒烟脚本备好待 key；
- rembg 模型下载（首次运行）自包含化至 deps 缓存。

## 6. 验收标准

⑥：Blender 真实建模→Cycles CPU 渲染→GLB/STL 导出→验收门→账本全链通，产物目检；blender-mcp tools/list 冒烟；
⑦：mock 全链（出图→护栏→账本→rembg 透明底→measure/align）通，真实出图冒烟脚本就绪（挂起注明）；
`tests.cmd all` 全绿；归档；随后启 eval 基准层统一建设。

## 7. 实施结果（2026-11-17 收尾记录）

- 验收标准达成：⑥Blender 5.2.3 LTS 真实全链（建模→Cycles CPU 32 采样→GLB→验收门→冻结→
  篡改报 drift）55s 实跑通过且渲染目检；blender-mcp 10 工具冒烟（T2）；⑦mock 全链（双云客户端/
  护栏/账本 tEXt/rembg 真机 u2net/measure+align 精度实证）47 项全绿，**真实出图冒烟挂起**
  （ARK key 过期，smoke_real.py 就绪待 key）。
- 依赖：mcp-for-blender 2.3.0（=blender-mcp 更名，差②豁免）、tripo3d 1.0.0（差①豁免）、
  rembg[cpu] 2.0.87（u2net 176MB 预下载自包含）；skimage/opencv 撤销（imaging 走 scipy 单轨）。
- 薄胶水：⑥ 886 行（验收门/账本冻结/脚手架/doctor）；⑦ 1163 行（纯逻辑 ~600，含 imaging 精度修复）。
- 终态 `tests.cmd all`：ruff 全绿 + 559 passed / 14 hardware。

## 8. eval 基准层裁决（2026-11-17）
- 七域全齐后用户裁决：**eval 基准层继续暂缓**（机械断言 559 例已足，需要时再建）；原裁决点名的 pptx 保真门/文献链基准随此搁置，素材与 smoke 脚本均留存可复用。
