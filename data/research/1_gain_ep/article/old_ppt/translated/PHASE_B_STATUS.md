# Phase B 进度与记忆 —— 研究进展汇报PPT汇总版

> **续做必读**：本文件是本任务的"记忆锚"。context 被压缩或换会话后，**先读这里**，
> 再读技能手册 `.qoder/skills/document-writing/references/digest.md`（完整流程 / 编辑锚点 / 进度信号）。
> **进度真值 = 各 `part_*.md` 中剩余 `_(待填)_` 的数量**（见下"剩余待填"）。按"Next up"续做，勿从头重来。

## 工作区

- 目录：`data/research/1_gain_ep/article/old_ppt/translated`
- 源 pptx：`研究进展汇报PPT汇总版.pptx`（251 页 / 734 图 / 580 唯一 / 202 含图页 / 49 文本页）
- **Phase A：✅ 完成**（202 整页渲染 `renders/` + 45 gif 抽帧 + 1 wmf 预览 `images/previews/`；`--lint` broken=0 pending=0，2026-09-24）
- **全局语境参考**：`data/research/1_gain_ep/article/exp_report/增益管槽实验大报告_初稿.md`（**只读开头 ~200 行**建立全貌；正文主体是实验平台代码细节，按需查阅、勿通读）。全貌：背景=Fang et al.(PRA) 损耗型非厄米 metagrating(EP/DP)；本实验=**精确时间反演**（几何不变、第二槽损耗→增益）。

## 进度表（按 part）

| Part | Slides | 图页 | 已解读图页 | 状态 |
|---|---|---|---|---|
| 01 | 001–040 | 34 | 34 | ✅ 完成 |
| 02 | 041–078 | 32 | 32 | ✅ 完成 |
| 03 | 079–117 | 33 | 33 | ✅ 完成 |
| 04 | 118–153 | 25 | 25 | ✅ 完成 |
| 05 | 154–190 | 29 | 29 | ✅ 完成 |
| 06 | 191–230 | 32 | 32 | ✅ 完成 |
| 07 | 231–251 | 17 | 17 | ✅ 完成 |

- 剩余待填（`_(待填)_` 总数）：**7**（均为 7 个 part 文件头说明行的非真实占位符）。真实图页占位符 = **0**——**part_01–07 全部 202 图页解读已收官**（2026-09-24；part_07 33 图 = 24 首现 + 9 复用）。
- 叙事综述版 `NARRATIVE.md`：✅ 完成（2026-09-24，全 deck 脉络长文 + part/关键图索引）；`index.md`「图像语义清单」已回填 10 张高频复用图语义标签。
- 纯文本页体检：✅（49 页 = ~41 分节页仅节标题 + 8 散文页 001/022/023/142/149/156/177/230，均为作者原有干净中文，无损坏、无编辑目标）。

## Next up

- **✅ Phase B 全部完成**（2026-09-24）：202 图页解读（真实剩余占位=0）+ NARRATIVE.md + index 语义清单 + 文本页体检。终检 `--lint`：链接存在 986、**缺失 0、待渲染 0**（broken=0 / pending=0）。无待续项。

## 术语表 / 反复出现的母题（随解读增长，保持一致；确立后后续复用）

| 术语 / 母题 | 权威解读（一句话） | 首现 |
|---|---|---|
| 增益管槽 | 背景论文(Fang et al., PRA)以三聚氰胺棉在第二槽引入损耗(c_i>0)实现 EP；本实验做精确时间反演——几何不变、第二槽换 4016 扬声器等效增益(c_i<0)。槽深 l_s=56.9/19.5/23.2 mm、宽 t_s=16.05/8.13/10.82 mm，f=3430Hz。 | Slide 003 |
| 亚波长同相 / 单传声器 | 槽深~λ/2、宽~0.1–0.2D，槽内声压近似同相 → 单传声器即可表征槽内场（报告§2.2.1；对应 Slide 004 四幅截线声压图）。 | Slide 004 |
| EP / DP | EP=S 矩阵退化为 Jordan 块（特征值+特征向量同时简并）；DP=对角（仅特征值简并）。双通道(±45°、D=√2λ/2)仅镜面+逆行反射两模。 | Slide 002 |
| 反馈演化算法 | 以槽内传声器实测声压为反馈，迭代优化扬声器驱动幅度/相位至目标增益（报告§2.2.3/第5章）。 | — |
| 扫场 | 二维平面波导中对超表面附近散射声场做二维扫描测量（无源/有源对照），SweeperCore + 步进电机 X/Y。 | — |
| 二维波导实验平台 | Python `sweeper400`（自 LabVIEW 迁移）；NI PXIe-1083+5×4468(10AI/10AO)；YAMAHA PA2030a(±3.162V 安全限)；36 腔体喇叭平面波声源(左/右 ±45°)。 | — |
| 谱奇点 (spectral singularity) | S 矩阵所有元发散的点（分母 Δ1Δ2−κ²=0）；**与特征值/向量简并（EP）无必然联系**（Slide 016/017 实证）。 | Slide 015 |
| 谱奇点简并条件 | 判别式 Δ=(H11−H22)²+4κ²；Δ=0 时两谱奇点合并为一点（Slide 019）；与 EP 的特征值/向量简并仍非同一概念。 | Slide 019 |
| EEP / IB-EEP（待确认） | Slide 020 新异常点：单边无反射、特征值不简并但有一个归零、特征向量正交；符合「Class I」条件（β^M↛∞, β^B→∞, Λ=0, E+≠E−）；缩写展开待后续页/文献确认。 | Slide 020 |
| 零点 / 极点表征 | 极点充要条件＝共同分母 Δ1Δ2−κ²=0（谱奇点同式）；零点充要条件＝S 特征值至少一个为 0（或行列式 0）；零点解析解见补充材料（Slide 026），数值 2000+400.014i / 2000+679.986i。 | Slide 026 |
| 劈裂/简并条件对称式 | 零点简并 (ω1+i(γc1−γ1)/2−ω2−i(γc2−γ2)/2)²+4κ²=0；极点简并 (ω1−i(γc1+γ1)/2−ω2+i(γc2+γ2)/2)²+4κ²=0；两式对称（γc−γ ↔ −(γc+γ)）（Slide 027）。等频简化：零点 16κ²=(γ2−γ1+(γc1−γc2))²、极点 16κ²=(γ2−γ1−(γc1−γc2))²；Lossless EP：γc1=γc2 且 16κ²=(γ2−γ1)²（或 γ1=γ2 且 16κ²=(γc1−γc2)²）（Slide 028）。 | Slide 027 |
| CPA EP 之问（Slide 029） | 简并零点 390 处 S≈[[−1,i],[i,1]]、特征值{0,0}、特征向量仅一个（幂零部分）＝本文"系统零点"；两侧 380/400 为经典 EP 式（等特征值+缺特征向量）；作者问中间点是否是 CPA EP。 | Slide 029 |
| EP 三分（Lossless/Absorbing/Resonant） | Lossless EP＝两简并等式同时满足（Slide 028）；Absorbing EP＝仅零点简并式取等（零点简并、极点劈裂，Fig.1 D，Slide 030）；Resonant EP＝仅极点简并式取等（极点简并、零点实轴劈裂，Fig.1 E，Slide 031）；补充材料临界耦合 κ=|γ1±γc∓…|/4。 | Slide 028–031 |
| CPA EP 可实现性与 generic 之辨 | 零点平移上实轴：γc1+γc2=γ1+γ2 且 κ=|γ1−γc1|/2（Slide 034）；generic (γc1≠γc2) 输入不平衡、S 矩阵不在 EP；nongeneric (γc1=γc2) 输入平衡、S 两特征值归零+特征向量合并＝散射矩阵 EP 与波算符 EP 同时（仅 TCMT 近似，ref 28）；Slide 029 的 380/390/400 三点＝Nongeneric。统一不等式 (a−b)²≦16κ²≦(a+b)²（a=γ2−γ1, b=γc1−γc2）与三维参数空间动图（Slide 032–033）。 | Slide 032–034 |
| 第二篇 CPA EP 文章（Slide 036） | LPR "Non-Hermitian Control of Topological Scattering Singularities Emerging from BIC"（Sakotic et al.）：CPA 条件 t±r=0；EP＝两极点合并、CPA EP＝其时间反演＝两零点合并；同极性 CPA charge 合并成 EP of zeros；作者注与第一篇系统的联系待再想。 | Slide 036 |
| 增益管槽2 实验平台改造 | 方形波导+3D 打印阶梯转接（10cm→5cm→3cm→管槽，Slide 038）；喇叭连接＝接线器+端子、3W 喇叭候选 36×20/35×16 mm（Slide 039）；NI 传递函数相位每次运行漂移→每次运行自动校准（全局变量+包装子 vi mySafeWrapper/myAverager(PhaseShift)，Slide 040）。 | Slide 038–040 |
| Dirac EP (DEP) | PRL 134, 153601 (2025)（ref28）：本征值如实 DP 线性变化、本征矢如 EP 简并（⟨ψ1|ψ2⟩=1）；模型 H_k=Σ(m+k)²|m⟩⟨m|+t₋|m⟩⟨m+1|+t|m⟩⟨m−1|，t±=V0(1±τ)/2 非互易耦合；τ=1 时出现 EP 或 DEP（θ 成奇异值）。 | Slide 042–044 |
| 低维截断H：二阶 vs 三阶 | 二阶截断（2×2）：本征值 E±=(H11+H22)/2 ± √[((H11−H22)/2)²+t₊t₋]，黎曼面关于 k=0.5 对称，只有常规EP（根式尖点）、无 DEP（Slide 045–046）；三阶截断（3×3）：黎曼面关于 k=0 对称，τ=1 处 DEP（纯实线性交叉）+ τ≈1.2 处常规 EP 共存（Slide 047–048）。 | Slide 045–048 |
| 二阶中构建 DEP（ref29） | 三阶 H 消一个基矢 + k=0 近似 → 二阶 H'_a，τ=1、k=0 时 → 幂零 Jordan 块 [[0,0],[V0²,0]]＝DEP；本征值处处纯实（虚部恒 0）、实部锥形交叉；二阶 H 须**二阶微扰**才现 DEP（三阶线性即可）。TABLE I 对比 DP/DEP/常规 EP2s 八项指标。 | Slide 049–050 |
| 复频域激发 | Kim/Krasnok/Alù 综述（Lasers Photonics Rev.，OPTICS REVIEW）：用激励信号的指数增益/衰减模拟材料增益/损耗（虚频率 ωᵢ>0＝Virtual loss、ωᵢ<0＝Virtual gain）；靠系统入出射时间延迟造“稳态增益”假象（响应瞬时则失效）；虚部应远小于实部。管槽应用：将三管槽增益/损耗“同步平移”（待测增益槽→无损、其余→有损）+ 指数衰减信号激发。 | Slide 063–064 |
| 指数增长信号＝无界窗函数 | 时域衰减/增长信号是宽频的（非单频）；窗函数视角：指数包络＝“无界窗函数”，傅里叶变换虽不存在但“非单频”直觉成立（单频检测：FFT 整周期准确、非整周期幅值偏差~30%→加 Hann 窗+幅值修正）。 | Slide 080–081 |
| CPA EP 传递矩阵法（深化） | 另一篇参考（耦合谐振子 TCMT）：极点＝H₀本征值、零点＝S 本征值=0（两解组合为 CPA EP）；ABCD 传递矩阵 M_Tl=M_loss·M_TL·M_gain=[1 0;Y₁ 1][cos kd, jZ₀sin kd;(j/Z₀)sin kd, cos kd][1 0;Y₂ 1]；散射系数 r_l=q_l/p、r_r=q_r/p、t_r=2/p（极点 p=0、零点 q=0）。 | Slide 092–099 |
| 扫场半场图 | 二维扫场传递函数幅值比/相位差空间分布（原始散角色图 + 插值平滑连续色图）；按入射方向分正向/侧前向/侧后向（半场图）。 | Slide 087–089 |
| sweeper400-python 仓库 | GitHub HardcOld-Bird/sweeper400-python（MIT、Python 100%）：automated acoustic field measurement using NI PXIe-4468 DSA + MT-22E motor controller；异步架构/OOP/CI-CD/可复现。 | Slide 090 |
| S 矩阵零极点＝ABCD 高维几何 | “矩阵有为0的特征值”⇔“行列式=0”→S 零点条件 t²=r_l·r_r；ABCD 中零点 4=(B/Z₀−CZ₀)²−(A−D)²、极点 A+B/Z₀+CZ₀+D=0；ABCD 是参数函数，参数空间 Y1/Y2/ω/d（均含实虚部），零极点为复变隐函数→描绘比参数空间低两维的几何（平面点/三维空间曲线）。 | Slide 099–100 |
| 主被动混合超材料 | 参考 Hernandez/Cheer/Memoli（ISVR Southampton，2025.9.30）“Transmission Loss of a Labyrinthine Acoustic Metamaterial Augmented with Multichannel Feedforward ANC”：迷宫型被动超表面（50×5.7cm，3D 打印）+ 2D 波导多通道前馈 ANC（主声源×1/反馈声源×9/传声器×40，3×PXIe-4497），1000–3000Hz 透射损失对照。本组拟用离线 ANC（无需考虑稳定性）复现。 | Slide 101–103 |
| 声场重建 ML 调研 | “Data-driven Sound Field Reconstruction and Sound Control: A Review of ML Approaches”；Hahmann et al. JASA 150(6) 2021 DTU 机械臂传声器实测；方法清单（Diffusion Model / Physics-informed NN / Generative models 等 8 篇）。 | Slide 104 |
| 基线偏移与滤波改善 | 旧机箱信号基线偏移~−0.6→高通滤波（10Hz Butterworth IIR）：双向 sosfiltfilt 无延时但双边边缘效应；窗函数抑制起振但破坏相位（不可接受）；长延拓 padlen 越长越好；单向 sosfilt 边缘单边但有相位延迟（单频/窄带影响小）；最终 detrend 去线性趋势+sosfilt；高采样率（171500Hz）+带通滤波去高频噪声。 | Slide 106–113, 117 |
| 时变超表面（Floquet） | 参考“Exceptional Points, Lasing, and Coherent Perfect Absorption in Floquet Scattering Systems”：时空调制 ε(x,t)=1+χ₀[1−M_s cos(Ωt)]，ω_n=ω+nΩ（准频率，首时间布里渊区 0≤ω<Ω）；单层介质/球体/球体阵列频率混频；S_F†VS_F=V、S_F†V|c^in⟩=(1/λ)V|c^in⟩→时变系统 EP/CPA/Lasing 相图（λ_min vs M_s）。有源平台天然能实现（一维最简）。 | Slide 115–116 |
| ESS 扫频校准 | 真实扬声器≈带通滤波器（对频带内各成分幅值/延时不同，窄带内变化不剧）。传递函数校准法：单频逐点（慢）/白噪声（低 SNR）/ESS 扫频（主流）。具体：scipy.signal.chirp method='logarithmic' 指数变频 + Tukey 窗平滑首尾；因 NI 同步 AI/AO 起止对齐，需加空白尾以采集含飞行时间的完整声信号。 | Slide 122–123 |
| 跨机箱 PFI 同步触发 | 多台 PXIe 机箱（Thunderbolt/雷电线级联）不扩展触发总线→机箱间同步无效；需每机箱引 1 个 PFI 线互联作硬触发（SMB 同轴线接 PFI）；外部时钟（GPSDO PWM37253）自动锁定背板时钟无需显式设置。 | Slide 124/128/136 |
| Exceptional BIC（EP-BIC） | 参考 Canós Valero et al. PRL 134, 103802 (2025)：将 BIC（连续域 bound state，损耗=0）与非厄米 EP 结合；有效哈密顿量 Ĥ 参数化，无损体系（γ^int=0）复现 S-BIC 与 EP-BIC（EP 处 κ=0.050，本征值 ω̃_± 简并）。 | Slide 126–127 |
| TBIC（拓扑 BIC） | 参考 Ruizhi Dong et al. PRL 134, 206601 (2025)（同济）：Observation of Extreme Anisotropic Sensitivity at TBIC；H_Σ 矩阵（f_Σ=3890−10.29C, v_Σ=1050−12.17C, u_Σ=440+3.65C），Σ/Π 能带随 C 的 3D 动画；拓扑保护 BIC 的各向异性极致敏感度。 | Slide 131 |
| 频率梳/频率梯度超表面 | 参考 Shaltout et al.《Spatiotemporal light control with frequency-gradient metasurfaces》(Science)：相位梯度（静态偏转）vs 频率梯度（动态偏转）超表面；频率梳光源＝时域周期脉冲→频域梳状谱（Mathematica 演示）。 | Slide 134 |
| 多通道校准精度 | 8 通道传递函数测量（3430Hz）：正常通道一致性好可免校准；机箱间同步无效源于**时钟漂移**而非触发。GPSDO 外部时钟 + PFI 触发 + 3 种平均重复（连续多段/重复开关吼叭/重复启动任务）→跨机箱相位差 ~2–3°（随机触发延迟）、同机箱噪声水平。1PPS 触发无明显效果（误差源于任务启动耗时）。 | Slide 133/136–138/143 |
| 负电导增益超表面（光学） | 参考 Zhu/Qian/Li/Chen PRL 133, 113801 (2024)「Negative Conductivity Induced Reconfigurable Gain Metasurfaces」：由坡印廷定理（S̄=Ē×H̄*，本构 B=μH/D=εE/J=σE，TEM 波阻抗 η=√(μ/ε_eff)，ε_eff=ε'+i[ε''+σ/ω]）→有功功率实部→负电导⇒负有功功率⇒增益（改 TD 偏置电压进入工作点）。智能超表面综述 Tsilipakos et al.（Adv Opt Mater）；深度学习隐身衣 Qian et al.（Nature Photonics）。 | Slide 146–152 |
| 实验参考文章群 | 声学放大二极管（Wen et al. PRL 130,176101(2023) 非互易 Willis 耦合）；有源超表面非互易场变换（Wen et al. Sci Adv）；消逝波 PT 对称（Chen et al. Commun Phys）；拓扑/谷霍尔边缘态。 | Slide 144 |
| 单几何参数全相位调制 EP 文章 | Shun Wan/Hu/Huang/Franco Nori 等 PRL 136, 053801 (2026)《Spin-Selective Topological Effects without Encircling Exceptional Points》：Dual-EP metasurface，仅靠单一几何转角 θ 实现全 2π 相位调制；复频率平面 EP_R(红)/EP_L(蓝)/Pole/c-BIC 叶状能带交叉，无需环绕 EP 即得自旋选择性拓扑效应。师姐转来的 EP 文章。 | Slide 155 |
| 多通道传递函数校准（3430Hz） | 8 mic 输入→反推 8 speaker 输出使实验波场=仿真波场（两种入射）；传递函数幅值比/相位差 vs 通道序数差（AI 索引−AO 索引），同通道对齐处（序数差 0）最强（~85 万尖峰）。硬件：NI PXIe-1083 多板卡 BNC 密集接线 + 8 孔喇叭阵列 + 梳状管槽工装。 | Slide 157–158 |
| 反馈虚拟实验（from_truth 螺旋收敛） | 与实验完全平行的仿真：探针差值在复平面螺旋收敛到稳态（模式 from_truth/from_calib/constant_2b）；中间槽增益一致、边缘槽偏离（类周期拓扑）。迭代求解输出 ground_truth.pkl，修正量范数递减、3 次迭代验证误差 ~2.7e-19。 | Slide 160–181 |
| 反馈算法理论推导链 | F 矩阵/F0/D/P 向量：A_i=p理想_i/(p理想_i−d_i·F)，D=F⁻¹(P理想−d0·F0)，矩阵可逆→唯一解。理想 vs 实际系统：R_g=p_rg/p_i≈1.1、R=p_r/p_i≈0.9、B=(p_rg−p_r)/p_i≈0.2、A=p/(p−p_a)=(R_g+1)/(R+1)≈1.1。等比数列模式(only_p 发散啸叫) vs 原有模式(p_and_d 收敛)；方法1 p_n=p0(1−q^n)/(1−q),q=1−1/A vs 方法2 一步稳定。 | Slide 168–181 |
| 有源非互易声学文章群 | Geib et al. PRB 103,165427(2021)《Tunable nonlocal purely active nonreciprocal acoustic media》；讲座嘉宾文章 Reflectionless/Directional amplifier/CPA-Laser 参数面。 | Slide 170–172 |
| 有源耦合晶格（Floquet 非阿贝尔 TI） | Qiu et al. PRX 16, 011061 (2026)（武大）《Anomalous Floquet Non-Abelian Topological Insulators》：h13(kx,t)=s13+v13(t)e^{−ikx}+v13*e^{ikx}，周期方波 v13 调制 T=1.5–2.4ms，入射 3430Hz；三维有源耦合晶格。 | Slide 180 |
| ST 时空涡旋文章（张若洋） | Zhang/Cui/…/Chan，Nature (2025) DOI 10.1038/s41586-025-08948-6《Bulk–spatiotemporal vortex correspondence in gyromagnetic zero-index media》：Ordinary DZIM(Spin-1 triple point, ε=μ=0)→加 B_ext 破时间反演→Gyromagnetic DZIM(Spin-1/2 Dirac point, det[μ̿]=0)；垂直入射高斯脉冲反射产生时空涡旋。零折射率/拓扑/奇异光子学。 | Slide 185 |
| 简化实验方案（冻结输出） | 理想流程=激励波束入射→系统演化（达稳态需几 min）→场扫描。简化1：扫场前冻结所有喇叭输出避免失稳；简化2：若演化实验难实现（信噪比/稳定性）则用理论计算/仿真获取冻结输出。 | Slide 187 |
| 无盖板扫场结果 | 相控阵全同输出→测得场几乎均匀（无条纹），相控阵仿真计算结果→清晰红蓝斜条纹，对比凸显需反馈校准；传递函数幅值比随 Y 单调衰减、相位差斜向条纹卷绕。激励声源出射波束是斜的（师姐早前提及），必要时制作新声源。 | Slide 188–190 |
| 单传声器合理性（要点回顾） | 目标管槽太小放不下两传声器→改单传声器；补做仿真证管槽中入射/反射复振幅之比恒为定值（复平面比值聚于实轴~0.2）→只有一个方程一个未知量、单传声器逻辑合理。背景=Fang et al. PRA 19,054003(2023) 损耗型非厄米 metagrating 的增益时间反演。 | Slide 232–234 |
| 增益定量实现（地图选点） | "仿真增益系数＋实验传递函数"组合驱动，cr–ci 参数空间稳态模长平均值热图（exp_eight_steady_mean/eight_target_mean/exp_floquet_steady_mean）；局部放大"地图"按迭代收敛难度分安全区(ρ<1)/危险区(α*可恢复)/不可达区，标最优点与候选红圈点。Gain-EP 对增益分布偏差鲁棒（8 管槽同增益系数同样有效），解释"仿真200倍/实验2倍"。 | Slide 235–236 |
| 扫场结果（有源/无源） | fourier_active/fourier_passive：Floquet 四通道(r_0^L/r_{+1}^R/r_{-1}^L/r_0^R)＋背景归一化幅值比声场图＋散射矩阵 S。左侧点效果更好、左右响应比~22倍；无源逆反射率未达1（未找到精确EP）；有源用全部相同增益系数后逆反射两图更像平面波。 | Slide 237–241 |
| 设备升级 | 更宽声源（49cm，理论波束宽于34cm）+声源增至2个；旧功放机箱无法满足→购置 YAMAHA PA2030a（2×30W 小功率、紧凑 28.8×21.5×5.4cm）；拟购 GRAS 1/8 英寸传声器改善扫场。新功放 transfer_function_polar 极坐标校准：通道一致性无明显劣势。 | Slide 243–245, 248 |
| 新设备结果·混乱波场自圆其说 | 新设备有源扫场（freq=3460Hz）四幅图三幅已呈平面波形态；仍混乱的最后一幅：COMSOL 仿真（f=3430Hz，旋转45°菱形域）弱响应侧也存在波阵面畸变→有增益介质时连仿真都畸变，实验畸变可包容。 | Slide 249–251 |
| 扫场平台性能改进 | 961 点扫描 72→36 min（并行/提速）、单次数据量 1.8GB→0.5~0.7GB（波形类重构、删除冗余一维）。 | Slide 192–193 |
| Evolver 演化模拟 | 类名 Evolver：以 AI/AO 复振幅在复平面的轨迹展示反馈迭代演化；L/R_10step_4s 为左/右入射稳态算例；单频检测太慢（ERROR 日志）。 | Slide 194 |
| 时间晶体 EP（Photonic Time Crystals） | 参考 arXiv:2512.02945：折射率时间周期调制 n(t)=n₀+δn cos(Ωt) 构成时间晶体，其能带交叉给出「时间 EP」。 | Slide 195–196 |
| 新样件/声源 CAD 与 GRAS 42AG 同步校准 | 无源/有源镜像样件（有源顶开 4 喇叭口）；传声器孔→喇叭口、喇叭孔→背槽（超轻粘土气密）；新平面波声源框架（接线端子快拆）；GRAS 42AG 声级校准器外接多喇叭口转接件做同步校准（8 通道融合波形 171500 Hz，优于顺序校准）。 | Slide 196/197/204/205 |
| 传声器指向性测试 | 正入射 vs 侧入射扫场：幅值差异不大，但正入射波场明显更平滑、侧入射条纹更乱噪声更大——该传声器有入射角偏好，侧入射误差更大（尚无精确检测方案）。 | Slide 199 |
| 宽频往复扫频方案 | 拟将扫场从周期单频改为「往复扫频」（ESS，复用 Slide 123）——泉森师兄：扫频 SNR 远优于白噪；对架构破坏较小。 | Slide 200 |
| 房师兄类似结构（泄露式时间反演） | 师姐转来的类房师兄结构文章：泄露式损耗经正/负入射（PI/NI）组合可模拟「时间反演」，配合特殊入射定义亦可视作实现增益。 | Slide 201 |
| 喇叭高功率测试 | 幅值 0.3–0.6 驱动下出现可恢复的幅值衰减（热效应），建议最大工作幅值 ≤0.5。 | Slide 222 |
| 增益演化（Evolver 实测） | 通道间串扰可致系统发散；将声源藏于吸声棉后解决；传递函数幅值比随通道序数差变化。 | Slide 223–224 |
| 平台现状总结 & 如何达 EP | 左/右入射响应差约 5 倍；两侧响应差距（比值）**完全**取决于无源效果，有源仅**等比放大**——故要达 EP/放大非对称应着力无源（与谱奇点/所有矩阵元发散、增益系数→无穷一致，复用 Slide 139）。 | Slide 225–230 |

## 决策与备注

- **2026-09-24**：LibreOffice 装于 `D:\XiGPrograms\libreOffice\base\program`（经系统 PATH 暴露、追加在末尾以免其自带 `python.exe` 遮蔽 venv）；`compose doctor` 探测到 `soffice.COM`；`--render` 成功补 202 页整页渲染 + wmf 预览，`--lint` broken=0 pending=0。
- **2026-09-24**：part_06 收官（首现页 192–207/222–226 逐图看渲染填写；分节页跳过；复用页 209–214/216–220 为 195–207 的逐图字节级重印，227/228/229 跨 part 复用 160/187/004/139，均只做「同 Slide X 图 Y」指向）。**锚点方法论固化**：①多邻近块用 image 路径行起连续全块锚；②字节相同的复用块用 `replace_all=true` 一次填全部；③original_text 绝不纳入尾随结构行（`\n\n---\n\n### Slide` 等）除非 new_text 原样复现——否则吞掉分隔符/标题头造成破坏；④复用块若 `重复图：同 Slide X 图 Y` 行与 `解读` 行相邻，可用该两行作短唯一锚。修正 206图2 typo「尖勈→尖劈」。
- **子 Agent 不可用于核心解读**（当前仅 Browser / CodeReview，都不能读本地 PNG 写结构化 md）→ 采用**单轮内批量**（一次 Read 多渲染 + 一次 SearchReplace 多 replacement）。
- **节奏**：验证期 3–5 图页/轮；稳定后 8–12 图页/轮；文本页可大批量。每轮末更新本文件。
- **2026-09-24（b08/b09 收尾）**：纯文本页体检——49 页中约 41 个为分节页（仅节标题）、8 个为散文页（001/022/023/142/149/156/177/230），内容均为作者原有、已可读的中文标签/散文，digest 忠实提取、无损坏，故**无安全且定义明确的"润色"编辑目标**（改写会破坏忠实性），b08 以"体检确认无需编辑"收官。b09 产出 `NARRATIVE.md`（问题→理论四脉→平台→扫场结果→结论展望，链接各 part/Slide）并回填 `index.md` 图像语义清单（10 张高频复用图）。part_07 33 图 = 24 首现（看整页渲染写）+ 9 复用（指向）；分节页 231/238/242/246 跳过。**锚点方法论**：①首现图用 image 路径行起连续全块锚；②跨页复用图（同 Slide 139 图 2/图 3 各出现 3 次、字节相同）用 `重复图`+`解读` 两行短锚 + `replace_all=true` 一次填全部；③original_text 绝不纳入尾随 `\n\n---` 结构行。踩坑：一次 new_text 误在 `邻近文字` 行尾带入多余引号，随即单独 SearchReplace 修正。
