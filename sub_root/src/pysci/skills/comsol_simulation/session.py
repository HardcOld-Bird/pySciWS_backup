"""COMSOL 会话/许可证探测 + domain-check 机械检查（纯判定 + 极薄驱动）。

**会话管理约定**（单证机器；详版在技能文档 references/java-mph-pitfalls.md）：

- mph.start() / COMSOL-MCP 的 comsol_start 都是**显式启动才占证**——空闲态不占证。
- JPype JVM 一旦启动**不可卸载**：``client.disconnect()`` 只清会话，**不放许可证**；
  释放许可证的唯一途径是**退出 Python 进程**。批任务编排要「进程级」而不是「连接级」。
- 占证探测看 **jvm.dll 加载**（= JVM 真启动）；``_jpype.pyd`` 只表示 jpype 被 import
  过，是**误报源**（旧项目实测），不得用作判据。
- mph 每个 Python 进程只允许一个 Client 单例；Windows 上 mph ≥1.3 走 client–server
  后端（派生 ``comsolmphserver``）。

**domain-check**（域几何声明机械检查）：四问齐全 + 每问带数值 + 激励口径的工作点-EP
偏移子问；缺项 FAIL。声明模板见技能文档 references/domain-declaration.md。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pysci.skills.comsol_simulation.validate import CheckResult

#: mph discovery 的版本串口径：**短名**（"6.4"）。完整串 "6.4.0.293" 会 exact-match
#: 失败（旧项目 cli-02 实测坑）。
_VERSION_RE = re.compile(r"^\s*(\d+)\.(\d+)")


def normalize_version(version: str) -> str:
    """归一化 COMSOL 版本串为 mph discovery 短名：``"6.4.0.293" -> "6.4"``。

    Args:
        version: 任意含 ``主.次`` 前缀的版本串（"6.4" / "6.4.0" / "6.4.0.293"）。

    Returns:
        短名 ``"主.次"``。

    Raises:
        ValueError: 找不到 ``主.次`` 前缀。
    """
    m = _VERSION_RE.match(version)
    if not m:
        raise ValueError(f"无法解析 COMSOL 版本串：{version!r}（期望形如 '6.4.0.293'）")
    return f"{m.group(1)}.{m.group(2)}"


# ---------------------------------------------------------------------------
# 许可证/占证探测
# ---------------------------------------------------------------------------

_TASKLIST_LINE = re.compile(r"^\s*(\S+)\s+(\d+)\s+(.*?)\s*$")
_MODULE_TOKEN = re.compile(r"\S+\.(?:dll|pyd)$", re.IGNORECASE)


def parse_tasklist_modules(text: str) -> dict[str, set[str]]:
    """解析 ``tasklist /m <module>`` 输出 → ``{进程名: {模块名}}``（纯函数）。

    输出列数随系统区域/会话信息浮动，故按「首列进程名 + 次列 PID + 其余列里
    形如 ``*.dll`` / ``*.pyd`` 的模块名」宽松解析::

        java.exe                        1234 Console                    1  jvm.dll

    Args:
        text: tasklist 标准输出原文。

    Returns:
        进程名到已加载模块集合的映射（模块名小写；无模块列的行不计）。
    """
    out: dict[str, set[str]] = {}
    for line in text.splitlines():
        m = _TASKLIST_LINE.match(line)
        if not m:
            continue  # 表头/空行/「信息: 没有运行的任务…」
        name, _pid, rest = m.groups()
        mods = {t.lower() for t in rest.split() if _MODULE_TOKEN.match(t)}
        if mods:
            out.setdefault(name.lower(), set()).update(mods)
    return out


def license_holders(procs: dict[str, set[str]]) -> list[str]:
    """从进程-模块映射中挑出**真占证**候选进程名（纯函数）。

    判据：加载了 ``jvm.dll`` 的进程（JVM 真启动，COMSOL 客户端/服务端）。
    仅加载 ``_jpype.pyd`` 的 Python 进程**不算**（import jpype ≠ 启动 JVM，误报源）。

    Args:
        procs: :func:`parse_tasklist_modules` 的输出。

    Returns:
        排序后的进程名列表。
    """
    return sorted(name for name, mods in procs.items() if "jvm.dll" in mods)


def probe_license_holders() -> list[str]:
    """探测本机当前持有 JVM 的进程（= 可能占用 COMSOL 许可证）。

    薄驱动：Windows 自带 ``tasklist /m jvm.dll``，无第三方依赖。

    Returns:
        进程名列表（空 = 无 JVM 进程，许可证大概率空闲）。
    """
    import subprocess  # 局部导入：纯函数测试不触碰进程

    out = subprocess.run(
        ["tasklist", "/m", "jvm.dll"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    return license_holders(parse_tasklist_modules(out.stdout))


def start_client(
    version: str | None = None,
    *,
    cores: int | None = None,
    session_mode: str = "stand-alone",
):
    """显式启动 mph Client（**占证**——用完请让进程退出以放证）。

    本机坑（2026-11-17 实测）：mph ≥1.3 默认 client–server 后端，``comsolmphserver``
    以 ``-port 0`` 起服务报 ``Internal error. Invalid port: 65536``
    （COMSOL getNextAvailablePortNumber 溢出）→ 本项目默认 ``session_mode="stand-alone"``
    （Windows 文档支持；进程退出放证语义不变）。

    Args:
        version: COMSOL 版本串（任意长度，内部 :func:`normalize_version` 归一短名）；
            None → mph 自动发现。
        cores: 求解核数上限（16GB 机器建议显式限核）。
        session_mode: ``"stand-alone"`` / ``"client-server"``（后者需另传可用端口，
            见 mph ``Server(port=…)``；本机 ``-port 0`` 路径已坏）。

    Returns:
        ``mph.Client``。

    Raises:
        ImportError: mph 未安装。
    """
    try:
        import mph
    except ImportError as exc:  # pragma: no cover - 依赖位
        raise ImportError(
            "需要 mph（pip 名 MPh==1.4.0 + JPype1==1.7.1）；import 失败先装依赖"
        ) from exc
    mph.option("session", session_mode)
    kwargs: dict[str, object] = {}
    if version is not None:
        kwargs["version"] = normalize_version(version)
    if cores is not None:
        kwargs["cores"] = cores
    return mph.start(**kwargs)


# ---------------------------------------------------------------------------
# domain-check 机械检查
# ---------------------------------------------------------------------------

#: 域几何声明四问键（模板见 references/domain-declaration.md）。
DECLARATION_KEYS: tuple[str, ...] = ("兼容性", "覆盖率", "边界适配", "激励口径")

_KEY_LINE = re.compile(r"^-\s*\*\*(?P<key>[^*]+)\*\*\s*[:：]\s*(?P<ans>.+)$")
_NUM_RE = re.compile(r"\d")
#: 敷衍答案黑名单（**子串**命中即视为无实质内容）。
_FUZZY_SUBSTR = ("待补", "TBD", "tbd", "待定", "暂无")
#: 敷衍答案黑名单（**整答**等于其一即为空；破折号只按整答判，避免误伤正文里的——）。
_FUZZY_EXACT = {"—", "-", "－", "/", "…", "..."}


@dataclass(frozen=True)
class DeclarationReport:
    """域几何声明机械检查结论。

    Attributes:
        ok: 四问齐全且每问含数值（硬门）。
        missing: 缺失的问键。
        fuzzy: 有键但答案无数值/敷衍的问键。
        soft_warnings: 软告警（如域版本与寻点版本不一致）。
        answers: 解析到的 ``{键: 答案全文}``。
    """

    ok: bool
    missing: list[str] = field(default_factory=list)
    fuzzy: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    answers: dict[str, str] = field(default_factory=dict)


def check_declaration(text: str) -> DeclarationReport:
    """机械检查 ``## 域几何声明`` 四问（纯函数，不启 JVM）。

    判定口径（旧 domain-check 同语义，机械重写）：

    - 必须有 ``## 域几何声明`` 节；四问逐行 ``- **键**：答案``；
    - 每问答案必须**含数字**且不属敷衍词（待补/TBD/—）；
    - 激励口径须含「工作点」「异常点」「偏移」「判据」要素（增益/损耗仿真硬门）；
    - 「域版本=…」「寻点版本=…」不一致 → **软告警**（不判 FAIL，交付前须补重寻读数）。

    Args:
        text: 声明 markdown 全文（任务书或交付物）。

    Returns:
        :class:`DeclarationReport`。
    """
    if "## 域几何声明" not in text:
        return DeclarationReport(
            ok=False, missing=list(DECLARATION_KEYS),
            soft_warnings=["缺 `## 域几何声明` 节"],
        )
    answers: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        m = _KEY_LINE.match(line.strip())
        if m and m.group("key").strip() in DECLARATION_KEYS:
            current = m.group("key").strip()
            answers[current] = m.group("ans").strip()
        elif current is not None and line[:1].isspace() and line.strip():
            # 续行（缩进）并入当前问键的答案全文——EP 子问常写在续行里
            answers[current] += " " + line.strip()
        elif line.strip():
            current = None

    missing: list[str] = []
    fuzzy: list[str] = []
    for key in DECLARATION_KEYS:
        ans = answers.get(key)
        if ans is None:
            missing.append(key)
        elif (
            not _NUM_RE.search(ans)
            or ans.strip() in _FUZZY_EXACT
            or any(f in ans for f in _FUZZY_SUBSTR)
        ):
            fuzzy.append(key)

    soft: list[str] = []
    excite = answers.get("激励口径", "")
    if "激励口径" in answers:
        for need in ("工作点", "异常点", "偏移", "判据"):
            if need not in excite:
                fuzzy.append(f"激励口径(缺「{need}」)")
    dom_v = re.search(r"域版本\s*=\s*([^\s，。；]+)", text)
    find_v = re.search(r"寻点版本\s*=\s*([^\s，。；]+)", text)
    if dom_v and find_v and dom_v.group(1) != find_v.group(1):
        soft.append(
            f"域版本={dom_v.group(1)} 与 寻点版本={find_v.group(1)} 不一致且无重寻读数"
            "——交付前须在新域重寻异常点"
        )
    ok = not missing and not fuzzy
    return DeclarationReport(
        ok=ok, missing=missing, fuzzy=sorted(set(fuzzy)), soft_warnings=soft,
        answers=answers,
    )


def check_units(model_unit: str, expected_unit: str = "m") -> CheckResult:
    """模型长度单位断言（米制口径统一后比较）。

    Args:
        model_unit: 模型长度单位字符串（"m" / "mm" / "cm" / "um"…）。
        expected_unit: 声明的期望单位（默认 "m"）。

    Returns:
        :class:`CheckResult`（值换算到米后比较）。
    """
    to_meter = {"m": 1.0, "meter": 1.0, "metre": 1.0, "cm": 1e-2, "mm": 1e-3,
                "um": 1e-6, "µm": 1e-6, "nm": 1e-9}
    try:
        ratio = to_meter[model_unit.strip().lower()] / to_meter[expected_unit.strip().lower()]
    except KeyError as exc:
        return CheckResult(
            name="units", ok=False, value=None,
            message=f"FAIL units：未知单位 {exc}（已知：{sorted(to_meter)}）",
        )
    ok = ratio == 1.0
    return CheckResult(
        name="units", ok=ok, value=ratio,
        message=f"{'PASS' if ok else 'FAIL'} units：模型单位 {model_unit} vs 期望 {expected_unit}",
    )


def check_domain_extent(
    domain_bbox: tuple[float, float, float, float],
    target_bbox: tuple[float, float, float, float],
    *,
    min_margin: float = 0.0,
) -> CheckResult:
    """兼容性硬门：目标区完整位于求解域内且最小边距 ≥ min_margin。

    Args:
        domain_bbox: 求解域包围盒 ``(x0, x1, y0, y1)``。
        target_bbox: 目标场区/标注区包围盒（同序）。
        min_margin: 要求的最小边距（单位同 bbox）。

    Returns:
        :class:`CheckResult`；``value`` = 实测最小边距（负值 = 已越界）。
    """
    dx0, dx1, dy0, dy1 = domain_bbox
    tx0, tx1, ty0, ty1 = target_bbox
    margins = (tx0 - dx0, dx1 - tx1, ty0 - dy0, dy1 - ty1)
    margin = min(margins)
    ok = margin >= min_margin
    return CheckResult(
        name="domain_extent", ok=ok, value=margin,
        message=(
            f"{'PASS' if ok else 'FAIL'} domain_extent：最小边距 {margin:.4g} "
            f"{'>=' if ok else '<'} 要求 {min_margin}（域 {domain_bbox}，目标 {target_bbox}）"
        ),
    )


def check_mesh_quality(
    min_element: float,
    wavelength: float,
    *,
    max_per_wavelength: float = 6.0,
) -> CheckResult:
    """网格质量断言：最粗波长方向每波长至少 ``max_per_wavelength`` 个单元。

    Args:
        min_element: 最小/代表性单元尺寸（同波长单位）。
        wavelength: 工作波长。
        max_per_wavelength: 每波长最少单元数（默认 6，二阶元可放宽到 3~4）。

    Returns:
        :class:`CheckResult`；``value`` = 每波长单元数。

    Raises:
        ValueError: 输入非正。
    """
    if min_element <= 0 or wavelength <= 0:
        raise ValueError(f"单元尺寸与波长须为正：{min_element}, {wavelength}")
    per_wl = wavelength / min_element
    ok = per_wl >= max_per_wavelength
    return CheckResult(
        name="mesh_quality", ok=ok, value=per_wl,
        message=(
            f"{'PASS' if ok else 'FAIL'} mesh_quality：每波长 {per_wl:.3g} 单元 "
            f"{'>=' if ok else '<'} 要求 {max_per_wavelength}"
            f"（λ={wavelength}, h={min_element}）"
        ),
    )
