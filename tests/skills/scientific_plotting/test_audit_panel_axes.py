"""`audit --panels` 的子图计数口径（backlog 20261018-110827-figure）。

实案：为让 `audit --panels` 不把 colorbar 轴算成「缺字母的子图」，作者把 colorbar 放进
`Axes.inset_axes` 宿主；随后 matplotlib 的 auto 标题定位把父轴标题挪到错误高度（第三列
R 面板标题跳到首行 L 标题位置），最终靠 `set_title(..., y=1.0)` 才稳定。

根因在审计口径：`fig.colorbar(..., ax=...)` 会注册一个 label=``"<colorbar>"`` 的轴，
旧版按 `len(fig.axes)` 计数把它当成子图。本文件钉住修正后的口径：

- 外置 colorbar 不计入面板数 → 带 colorbar 且角标齐全的图**零 panel-labels WARN**；
- 真正缺角标仍报 WARN（口径不能把漏号也一并放过）；
- inset 宿主 colorbar（历史绕行写法）同样不虚增计数；
- 不可见轴不参与计数。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from pysci.skills.scientific_plotting.tools import audit


def _panels(n: int = 3) -> tuple[Figure, list]:
    fig = Figure(figsize=(6, 2))
    axs = [fig.add_subplot(1, n, i + 1) for i in range(n)]
    for ax in axs:
        ax.imshow(np.zeros((2, 2)))
    return fig, axs


def _label(ax, text: str) -> None:
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left")


def _labels(axs) -> None:
    for i, ax in enumerate(axs):
        _label(ax, f"({chr(ord('a') + i)})")


def _panel_warns(rep: audit.AuditReport) -> list[audit.AuditIssue]:
    return [i for i in rep.issues if i.code == "panel-labels"]


def test_external_colorbar_is_not_counted_as_panel():
    fig, axs = _panels(3)
    fig.colorbar(axs[0].images[0], ax=axs[0])  # 外置 colorbar（进 fig.axes）
    _labels(axs)

    rep = audit.audit_figure(fig, expect_panel_labels=True)

    assert rep.metrics["n_axes"] == 4, "colorbar 轴仍在 fig.axes 里（总量读数如实）"
    assert rep.metrics["panel_axes"] == 3, "面板数必须排除 colorbar 轴"
    assert rep.metrics["panel_labels_found"] == 3
    assert _panel_warns(rep) == [], "角标齐全时不得因 colorbar 触发 panel-labels WARN"


def test_genuinely_missing_panel_label_still_warns():
    fig, axs = _panels(3)
    fig.colorbar(axs[0].images[0], ax=axs[0])
    _label(axs[0], "(a)")
    _label(axs[1], "(b)")  # 第三个面板仍缺

    rep = audit.audit_figure(fig, expect_panel_labels=True)

    warns = _panel_warns(rep)
    assert len(warns) == 1, warns
    assert "3 个子图但仅 2 个" in warns[0].message


def test_inset_hosted_colorbar_does_not_inflate_count():
    """历史绕行写法（inset 宿主）也必须零 WARN——修正口径对它同样成立。"""
    fig, axs = _panels(3)
    inset = axs[2].inset_axes([0.85, 0.1, 0.05, 0.8])
    fig.colorbar(axs[2].images[0], cax=inset)
    _labels(axs)

    rep = audit.audit_figure(fig, expect_panel_labels=True)

    assert rep.metrics["panel_axes"] == 3
    assert _panel_warns(rep) == []


def test_manual_cax_colorbar_is_not_counted():
    """手工 ``fig.add_axes`` 作 cax：matplotlib 不设 `<colorbar>` label，只留 `_colorbar` 回指。"""
    fig, axs = _panels(3)
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    fig.colorbar(axs[2].images[0], cax=cax)
    _labels(axs)

    rep = audit.audit_figure(fig, expect_panel_labels=True)

    assert rep.metrics["n_axes"] == 4
    assert rep.metrics["panel_axes"] == 3
    assert _panel_warns(rep) == []


def test_invisible_axis_is_not_counted():
    fig, axs = _panels(3)
    spacer = fig.add_subplot(1, 4, 4)
    spacer.set_visible(False)
    _labels(axs)

    rep = audit.audit_figure(fig, expect_panel_labels=True)

    assert rep.metrics["n_axes"] == 4
    assert rep.metrics["panel_axes"] == 3
    assert _panel_warns(rep) == []
