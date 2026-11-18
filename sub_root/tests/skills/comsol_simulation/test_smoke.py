"""真实 COMSOL 冒烟（``-m hardware``）：建模→solve→出版级出图→sidecar→domain-check。

守卫链（任一缺失 skip 而非红灯）：mph 可导入 → 无 ``comsol*`` 进程占证（单证机器
约定：显式 start 才占证、进程退出才放证）→ 许可证可得（start 失败转 skip）。模型用
**PressureAcoustics**（本机 Acoustics 许可证旧项目已实证）：1m×1m 矩形 + 空气材料 +
背景平面波 + 频率研究 1kHz。

产物落 ``sub_root/tests/artifacts/comsol_simulation/``（测试产物区，不入库），
供人工目检（视觉策略 2.4）。
"""

from __future__ import annotations

import pytest

from pysci.paths import TESTS_ROOT
from pysci.skills.comsol_simulation import export as ex
from pysci.skills.comsol_simulation import plotgroup, session

pytestmark = pytest.mark.hardware

_ARTIFACTS = TESTS_ROOT / "artifacts" / "comsol_simulation"


@pytest.fixture(scope="module")
def client():
    """显式启动 mph Client（占证！）；环境不满足时 skip。

    守卫语义（单证约定）：跑前确认无 comsol* 进程（GUI/comsolmphserver/其他代理）。
    「python 进程带 jvm.dll」不作拦截——多代理机上 JetBrains 等 JVM 常驻、并行代理的
    MCP server 惰性起 JVM，实测可共存；许可证真被占时 :func:`session.start_client`
    会抛许可证错误，转 skip（hardware 不可用不红）。
    """
    pytest.importorskip("mph", reason="需要 mph（MPh==1.4.0 + JPype1==1.7.1）")
    import subprocess

    names = subprocess.run(["tasklist"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=30, check=False).stdout.lower()
    busy = sorted({line.split()[0] for line in names.splitlines()
                   if line.startswith("comsol")})
    if busy:
        pytest.skip(f"COMSOL 进程在跑（{busy}）——单证约定：先停掉再跑 hardware 冒烟")
    try:
        # 版本串短名坑：完整串 "6.4.0.293" 会 exact-match 失败 → normalize 归一 "6.4"；
        # 会话模式：本机 client-server `-port 0` 已坏（java-mph-pitfalls）→ stand-alone
        return session.start_client("6.4.0.293", cores=2)
    except Exception as exc:  # noqa: BLE001 - 许可证/发现失败按不可用处理
        pytest.skip(f"COMSOL 启动失败（许可证/环境）：{exc}")


def _build_smoke_model(client):
    """建最小可解声学模型（mph 官方 demo 同款 API；坑点见 java-mph-pitfalls）。"""
    model = client.create("smoke_acpr")
    geometry = (model / "geometries").create(2, name="geometry")
    rect = geometry.create("Rectangle", name="domain")
    rect.property("size", ["1", "1"])
    model.build(geometry)

    # 坑：材料缺 c/密度 ⇒ 求解**静默**空解（evaluate 零点）——必须给材料；
    # 属性组显示名随界面语言本地化（中文机上不是 "Basic"）→ 按 tag 'def' 走 Java 侧
    materials = model / "materials"
    air = materials.create("Common", name="air")
    air_pg = air.java.propertyGroup("def")
    air_pg.set("density", "1.2")
    air_pg.set("soundspeed", "343")  # 属性名一个词（pitfalls §2）

    physics = model / "physics"
    acpr = physics.create("PressureAcoustics", geometry, name="acoustics")
    # 坑（java-mph-pitfalls §3）：新建 BackgroundPressureField 域选择默认空 →
    # p_b=0 全场为 0——必须显式 selection().all()；且 c_mat/k_src 等默认未定值 →
    # 残差 NaN（pitfalls §2 配方：PlaneWave + FromSpeedOfSound + from_mat）
    bpf = acpr.create("BackgroundPressureField", name="background")
    bpf.java.selection().all()
    bpf.property("pamp", 1)
    bpf.property("dir", ["1", "0", "0"])
    bpf.property("PressureFieldType", "PlaneWave")
    bpf.property("k_src", "FromSpeedOfSound")
    bpf.property("c_mat", "from_mat")

    (model / "meshes").create(geometry, name="mesh")  # 显式网格序列（demo 同款）
    study = (model / "studies").create(name="study")
    step = study.create("Frequency", name="frequency")
    step.property("plist", "1000")
    step.property("punit", "Hz")
    return model, acpr


def test_full_loop_smoke(client):
    model, acpr = _build_smoke_model(client)
    model.solve("study")  # study.run() 自动生成求解序列 + 数据集

    # 解非空且非常数（材料缺失误配会静默出空解——先钉死）
    import numpy as np

    arr = np.asarray(model.evaluate(f"{acpr.tag()}.p_t", "Pa"))
    assert arr.size > 0, "求解结果为空（查材料 c/密度、网格、数据集绑定）"
    assert arr.max() != arr.min(), "场为常数（查背景场选择域 selection().all()）"

    plot = (model / "plots").create("PlotGroup2D", name="field")
    surface = plot.create("Surface", name="pressure")
    surface.property("expr", f"{acpr.tag()}.p_t")

    # solve 后重绑（跨调用安全关键）：绘图组 → 首个数据集
    rebound = plotgroup.rebind_plot_groups(model)
    assert "field" in rebound

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    png = _ARTIFACTS / "smoke_field.png"
    result = ex.export_image(
        model, "field", png,
        width_mm=120, height_mm=120, dpi=150,
        clean=True, geom_bbox=(0, 1, 0, 1), guard_data=False,
    )
    assert result["png"].exists() and result["png"].stat().st_size > 0
    assert result["sidecar"].exists()

    from pysci.skills.comsol_simulation import sidecar as sc

    data = sc.read_sidecar(result["sidecar"])
    assert data["image"]["width_px"] == round(120 / 25.4 * 150)
    assert data["extent"] == pytest.approx([0, 1, 0, 1])  # 几何 bbox 恰为 1:1
    # 品牌剥离走原生 logo*d=off：plan 里 logo2d=off 必然出现
    assert any("logo2d=off" in a for a in result["applied"])
    # 空白自检：背景平面波场图不得为空白
    assert data["interior"]["blank"] is False

    # domain-check 机械检查（模型域/单位/网格质量）——事实取自本模型。
    # 网格口径：COMSOL 默认二阶元，每波长 ≥3 即可（线性元才是 ≥6）
    assert session.check_units("m", "m").ok
    assert session.check_domain_extent((0, 1, 0, 1), (0.1, 0.9, 0.1, 0.9),
                                       min_margin=0.05).ok
    assert session.check_mesh_quality(
        0.1, 343.0 / 1000, max_per_wavelength=3.0
    ).ok  # λ=0.343m, h≈0.1m 二阶元
