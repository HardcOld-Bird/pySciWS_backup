"""计算会话管理：结果持久化、日志、产物导出到 data/。

每个计算会话对应 data/research/<n>_<name>/theory/<slug>/ 下的一组目录：
- results/  数值结果（.npz, .csv）
- plots/    探索性图片（.png, .html）
- notes.md  Agent 计算日志（时间线记录每步操作和发现）

用法::

    from pysci.skills.theoretical_computation.tools.session import ensure_session, save_results

    session_dir = ensure_session("gain_ep", "ep_band_analysis")
    save_results(session_dir, {"eigenvalues": eig_vals, "params": params})
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings


def ensure_session(research: str, slug: str) -> Path:
    """确保计算会话的数据目录存在并返回路径。

        Args:
        research: 研究线名称（不含数字前缀），如 ``"gain_ep"``。
        slug: 计算会话标识，如 ``"ep_band_analysis"``。

    Returns:
        会话数据目录路径（``data/research/<n>_<name>/theory/<slug>/``）。
    """
    session_dir = settings.research_theory_data_dir(research) / slug
    (session_dir / "results").mkdir(parents=True, exist_ok=True)
    (session_dir / "plots").mkdir(parents=True, exist_ok=True)

    # 初始化 notes.md（如果不存在）
    notes_path = session_dir / "notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            f"# {research}/{slug} 计算日志\n\n"
            f"创建时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n---\n\n",
            encoding="utf-8",
        )

    return session_dir


def save_results(session_dir: Path, data_dict: dict[str, Any], name: str = "results") -> Path:
    """持久化数值结果到 session 的 results/ 目录。

    Args:
        session_dir: 会话数据目录。
        data_dict: 键值对，值应为 numpy 数组或可序列化对象。
        name: 文件名前缀（不含扩展名），默认 ``"results"``。

    Returns:
        保存的文件路径。
    """
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{name}.npz"

    # 分离 numpy 数组和其他对象
    np_dict = {}
    other_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            np_dict[k] = v
        elif isinstance(v, (int, float, complex, bool)):
            np_dict[k] = np.array(v)
        else:
            other_dict[k] = v

    # 保存 numpy 数组
    if np_dict:
        np.savez(str(out_path), **np_dict)

    # 非数组对象用 allow_pickle 追加保存
    if other_dict:
        pickle_path = results_dir / f"{name}_meta.npz"
        np.savez(str(pickle_path), allow_pickle=True, **other_dict)

    print(f"[session] 结果已保存: {out_path}")
    return out_path


def save_plot(session_dir: Path, fig: Any, name: str, dpi: int | None = None) -> Path:
    """保存 matplotlib Figure 到 session 的 plots/ 目录。

    Args:
        session_dir: 会话数据目录。
        fig: matplotlib Figure 对象。
        name: 文件名（不含扩展名）。
        dpi: 分辨率，默认使用配置值。

    Returns:
        保存的 PNG 文件路径。
    """
    plots_dir = session_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if dpi is None:
        dpi = settings.default_plot_dpi

    out_path = plots_dir / f"{name}.png"
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[session] 图已保存: {out_path}")
    return out_path


def write_log(session_dir: Path, entry: str) -> None:
    """追加计算日志条目到 notes.md。

    Args:
        session_dir: 会话数据目录。
        entry: 日志内容（会自动添加时间戳前缀）。
    """
    notes_path = session_dir / "notes.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    with notes_path.open("a", encoding="utf-8") as f:
        f.write(f"## {timestamp}\n\n{entry}\n\n---\n\n")


def list_sessions(research: str) -> list[dict[str, Any]]:
    """列出某研究线下所有计算会话。

    Args:
        research: 研究线名称。

    Returns:
        会话信息列表，每项含 name/path/n_results/n_plots。
    """
    data_dir = settings.research_theory_data_dir(research)
    if not data_dir.is_dir():
        return []

    sessions = []
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if d.name.startswith(".") or d.name == "参考资料":
            continue
        results_dir = d / "results"
        plots_dir = d / "plots"
        sessions.append({
            "name": d.name,
            "path": d,
            "n_results": len(list(results_dir.glob("*"))) if results_dir.is_dir() else 0,
            "n_plots": len(list(plots_dir.glob("*.png"))) if plots_dir.is_dir() else 0,
        })

    return sessions
