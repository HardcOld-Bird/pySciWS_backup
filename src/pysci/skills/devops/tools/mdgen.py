"""机械生成/改写 markdown 后的钩子终态治理（backlog 20261010-191922-devops）。

**问题**：`orchestration/skills/**` 下 .md 若由脚本批量产出，首次提交必被 pre-commit
的 `end-of-file-fixer` / `trailing-whitespace` 改写而失败——钩子打印 `Fixing <file>`
后中止提交，改动留在工作区（`AM`），必须再 `git add` + 提交一次。更隐蔽的是：脚本
作者若**在生成后立即**用逐字节/哈希断言内容等价性、随后才提交，则钩子改写让那条断言
当场失效——报告的是一个不再为真的证据。

**做法**（零算法漂移）：本工具**不重实现**钩子逻辑，而是**委托 pre-commit 自身**——
按顺序调用 `python -m pre_commit run <hook-id> --files <paths>` 跑官方两支「改写型」
钩子（`trailing-whitespace` 后 `end-of-file-fixer`；顺序同 `.pre-commit-config.yaml`），
迭代到本轮无文件被改写为止。逐文件返回 `pre_hash → post_hash` 与 pre/post 尺寸，
供生成脚本在**钩子终态字节**上做内容等价性验证。

只跑这两支钩子，不跑全套：mdgen-check 只治理 EOF/尾空白，**不越权**触发 ruff format
或 harness-budget——那些仍归提交时的 pre-commit 全套管。

**用法**（生成脚本落盘之后、`git add` 之前）：

```python
from pysci.skills.devops.tools import mdgen
from pathlib import Path
for r in mdgen.check_files([Path(p) for p in written], cwd=repo_root):
    if r.changed:
        print(f"钩子改写：{r.path} {r.pre_size}B→{r.post_size}B")
    # 用 r.post_hash 做内容等价性断言（不是 pre_hash）
```

或 CLI：`pysci-dev mdgen-check orchestration/skills/devops/references/*.md`
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

#: 与 `.pre-commit-config.yaml` §1 Official 的两支改写钩子对齐；顺序即执行顺序——
#: trailing-whitespace 先削行末空白，end-of-file-fixer 再收敛文件末尾（否则末尾纯空格
#: 空行会被 eoff 判成「一个非空行」留下）。追加钩子须同步本表与 dev.py 帮助文。
REWRITE_HOOKS: tuple[str, ...] = ("trailing-whitespace", "end-of-file-fixer")

#: 两支钩子互相触发（EOF 收敛后可能露出新的行末空白等）在真实仓库中未见；上限 3 轮
#: 纯为防御——若真到 3 轮仍不收敛，说明有钩子互相打架，宁可抛错也不静默放行陈旧字节。
MAX_ROUNDS = 3


@dataclass(frozen=True)
class FileResult:
    """单文件在钩子治理前后的字节快照。"""

    path: Path
    pre_hash: str
    post_hash: str
    pre_size: int
    post_size: int

    @property
    def changed(self) -> bool:
        """钩子是否改写过本文件（pre_hash ≠ post_hash 即改写）。"""
        return self.pre_hash != self.post_hash

    @property
    def delta(self) -> int:
        """字节增量（钩子终态减去生成态；通常为负或零）。"""
        return self.post_size - self.pre_size


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_hook(hook_id: str, files: list[Path], *, cwd: Path) -> int:
    """跑一支 pre-commit 钩子；返回 rc。rc=0 无改写；rc=1 有改写；其他为真错。

    pre-commit 需要 cwd 落在 git 仓库内并可见 ``.pre-commit-config.yaml``——主仓/worktree
    都满足（worktree 的 ``.git`` 是 file 指向主仓，pre-commit 会解析到主仓根）。
    """
    argv = [
        sys.executable,
        "-m",
        "pre_commit",
        "run",
        hook_id,
        "--files",
        *(str(p) for p in files),
    ]
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode not in (0, 1):
        # 真错——把 pre-commit 的 stdout/stderr 直接原样吐给调用者，方便排障
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
    return proc.returncode


def check_files(
    paths: Iterable[Path], *, cwd: Path, dry_run: bool = False
) -> list[FileResult]:
    """把 ``paths`` 中真实存在的文件送到钩子终态，逐文件返回 pre/post 快照。

    Args:
        paths: 待治理文件；不存在的路径静默跳过（生成脚本常带模板路径，缺失即视为
            该路径未产出——不该由本工具报错）。
        cwd: 运行 pre-commit 的目录（须落在 git 仓库内且可见 ``.pre-commit-config.yaml``）。
        dry_run: 只报告「钩子会改写哪些文件」，跑完把字节还原。缺省 False（真改写）。

    Returns:
        逐文件的 :class:`FileResult` 列表（顺序与去重后的 paths 一致）。

    Raises:
        RuntimeError: pre-commit 返回 rc 不属于 {0, 1}（钩子/环境本身故障），或
            ``MAX_ROUNDS`` 轮仍未收敛——此时钩子状态不确定，静默继续会让「终态哈希」
            失去意义，故硬失败。
    """
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        if not p.is_file():
            continue
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    if not uniq:
        return []

    pre: dict[Path, tuple[str, int]] = {}
    originals: dict[Path, bytes] = {}
    for p in uniq:
        pre[p] = (_sha256(p), p.stat().st_size)
        if dry_run:
            originals[p] = p.read_bytes()

    for _round in range(MAX_ROUNDS):
        any_rewrite = False
        for hook in REWRITE_HOOKS:
            rc = _run_hook(hook, uniq, cwd=cwd)
            if rc == 1:
                any_rewrite = True
            elif rc != 0:
                raise RuntimeError(
                    f"pre-commit 钩子 {hook!r} 意外退出码 rc={rc}（应为 0=无改写 / "
                    f"1=已改写）；钩子状态不确定，拒绝以当前字节报告终态哈希。"
                )
        if not any_rewrite:
            break
    else:
        raise RuntimeError(
            f"pre-commit 改写 {MAX_ROUNDS} 轮仍未收敛；可能是钩子互相打架，请人工排查。"
        )

    results: list[FileResult] = []
    for p in uniq:
        post_hash, post_size = _sha256(p), p.stat().st_size
        results.append(
            FileResult(
                path=p,
                pre_hash=pre[p][0],
                post_hash=post_hash,
                pre_size=pre[p][1],
                post_size=post_size,
            )
        )
        if dry_run:
            # 还原生成态字节，让 `--dry-run` 真做到「只报告不改动」
            p.write_bytes(originals[p])
    return results


def format_results(results: list[FileResult]) -> str:
    """把逐文件结果渲染成人类可读多行；每行末列 = 钩子终态 sha256（供事后校验）。"""
    if not results:
        return "mdgen-check：无在盘文件可校验（路径全部缺失或已跳过）"
    lines: list[str] = []
    changed_n = 0
    for r in results:
        flag = "FIXED" if r.changed else "OK   "
        if r.changed:
            changed_n += 1
        delta = f"（Δ{r.delta:+d}）" if r.changed else ""
        lines.append(
            f"  [{flag}] {r.path}  {r.pre_size}B→{r.post_size}B{delta}"
            f"  post_sha256={r.post_hash[:16]}"
        )
    lines.append(
        f"合计 {len(results)} 文件，钩子改写 {changed_n} 个"
        + (
            ""
            if changed_n == 0
            else "；请把内容等价性断言的基准换成上面的 post_sha256 再 git add"
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口；返回 rc。``--dry-run`` 只报告不改动。"""
    parser = argparse.ArgumentParser(
        prog="pysci-dev mdgen-check",
        description=(
            "机械生成/改写 markdown 后，把 pre-commit（trailing-whitespace + "
            "end-of-file-fixer）对文件的改写**提前做完**，并逐文件打印 pre/post "
            "sha256——让「钩子终态哈希」成为内容等价性验证的唯一基准，"
            "首次 add/commit 即通过。"
        ),
    )
    parser.add_argument("paths", nargs="+", help="待治理文件（路径不存在则跳过）")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只报告钩子会改写哪些文件，跑完还原字节（预览用）",
    )
    args = parser.parse_args(argv)

    try:
        results = check_files(
            [Path(p) for p in args.paths],
            cwd=Path.cwd(),
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        print(f"mdgen-check 失败：{exc}", file=sys.stderr)
        return 2
    print(format_results(results))
    if args.dry_run:
        n = sum(1 for r in results if r.changed)
        if n:
            print(
                f"\n[dry-run] {n} 个文件若不治理会被钩子改写；"
                f"去掉 --dry-run 落钩子终态。"
            )
    return 0
