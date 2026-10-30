"""notes —— paper-note frontmatter 的读写、规范化与合并（叶子模块）。

本模块是 literature_research 各组件共用的「笔记格式层」，收敛了原先散落在
``research.py`` / ``citation_verify.py`` / ``cache_manager.py`` / ``browser_fetch.py`` /
``arxiv_client.py`` / ``openalex_client.py`` 里的 5 份手写 frontmatter 解析器与
2 份手写 YAML 序列化器。

**依赖约束（务必遵守）**：只依赖 stdlib + ``yaml`` + :mod:`.config`。
绝不 import 同层的任何客户端模块（``openalex_client`` / ``arxiv_client`` /
``citation_verify`` / ``cache_manager`` / ``research`` …），否则会形成循环依赖——
它们全都要 import 本模块。

设计要点
--------
1. **统一到 PyYAML**。``pyyaml`` 是项目运行时依赖（``pyproject.toml``），scientific_plotting
   已在用。原手写解析器有一个实打实的缺陷：它跳过所有以 ``-`` 开头的行，导致 block-style
   的 ``authors`` / ``topics`` 等列表字段全部丢失（连带 ``cache_manager.referenced_paths()``
   读不出 ``extracted_md_path``，使 ``prune --keep-referenced`` 的保护完全失效）。
2. **``dump_frontmatter`` 幂等**。已知字段一律按 :data:`FIELD_ORDER` 输出、未知字段按插入序
   追加在后，因此同一输入永远产生同一字节序列——``index --fix`` 的 diff 才可读，测试才可依赖。
3. **合并只填空，绝不覆盖**。:func:`merge_frontmatter` 只填 ``None`` / ``""`` / ``[]`` / ``{}``
   或缺失的键。用户手填的 ``my_rating`` / ``status`` / ``related_to_my_work`` 永不被机器值顶掉。
4. **正文神圣**。本模块只提供 :func:`split_note`（拆出 body）与 :func:`render_note`（原样拼回），
   调用方有责任保证自动写入路径下 body 逐字节不变；唯一的例外是显式 ``--overwrite``。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import yaml

from .config import settings

# ---------------------------------------------------------------------------
# 规范字段序（与 templates/paper_note.md 的 frontmatter 一致）
# ---------------------------------------------------------------------------
#: paper-note frontmatter 的规范字段顺序。
#:
#: ``dump_frontmatter`` 先按本序输出**存在**的已知字段，再把未知字段按插入序追加在后。
#: 新增 frontmatter 字段时应同步更新本常量与 ``templates/paper_note.md``，两者的顺序
#: 保持一致，``index --fix`` 的规范化 diff 才是纯粹的值变化而非字段搬家。
FIELD_ORDER: tuple[str, ...] = (
    # --- 书目元数据 ---
    "title",
    "short_title",
    "authors",
    "first_author_last_name",
    "corresponding_author",
    "year",
    "publication_date",
    "journal",
    "journal_ref",
    "publisher",
    "volume",
    "issue",
    "pages",
    "doi",
    "arxiv_id",
    "openalex_id",
    "wos_id",
    "zotero_key",
    "zotero_uri",
    "local_pdf_path",
    "extracted_md_path",
    "extracted_html_path",
    "oa_url",
    "oa_status",
    # --- 质量与影响力指标 ---
    "cited_by_count",
    "cited_by_count_normalized",
    "jif",
    "jif_5yr",
    "jcr_quartile",
    "scimago_quartile",
    "citescore",
    "esi_highly_cited",
    "esi_hot_paper",
    "journal_h_index",
    "journal_tier",
    "journal_tier_basis",
    "listed_in",
    # --- AI 分类标签 ---
    "topics",
    "methods",
    "systems",
    "related_to_my_work",
    "related_to_my_work_reason",
    # --- 状态 ---
    "status",
    "my_rating",
    "added_date",
    "last_reviewed",
    "review_count",
    # --- 语义检索关键词 ---
    "keywords_auto",
)


# ===========================================================================
# 期刊档次派生（纯函数，无网络）
# ===========================================================================
#: OpenAlex ``sources.listed_in`` 里**带专家评议档次**的名单，及其档次 → tier 的映射。
#:
#: 为什么用这三套而不是引用类指标：JUFO（芬兰）、Norway（挪威）、KI-JL（瑞典卡罗林斯卡）
#: 都由**学科专家小组**按领域内评议定档，不受学科引用密度影响。实测差异极大——
#: ``J. Acoust. Soc. Am.`` 的 OpenAlex ``2yr_mean_citedness`` 只有 0.82（声学研究引用
#: 半衰期长、单篇引用数低），但 JUFO 把它判为 3 级（最高档），与 Nature / PRL 同档。
#: 对物理声学这类低引用密度领域，这个分级比 JIF 更贴近领域共识。
#:
#: 名单里的 ``cwts-core`` / ``erih-plus`` / ``medline`` / ``doaj`` / ``doyens`` 是**二元**
#: 收录标记（在/不在），不含档次信息，因此不参与派生。
TIER_LISTS: dict[str, dict[int, str]] = {
    "jufo": {3: "top", 2: "leading", 1: "basic"},
    "norway": {2: "top", 1: "basic"},
    "ki-jl": {3: "top", 2: "leading", 1: "basic"},
}

#: tier 的高低序，用于在多套名单之间取最高档。
TIER_RANK: dict[str, int] = {"top": 3, "leading": 2, "basic": 1}

#: :data:`TIER_RANK` 的反查表。
_RANK_TO_TIER: dict[int, str] = {v: k for k, v in TIER_RANK.items()}

#: 名单声明序——只为让 ``basis`` 的输出顺序稳定（与 ``listed_in`` 的原始顺序无关）。
_TIER_LIST_ORDER: tuple[str, ...] = tuple(TIER_LISTS)

#: ``listed_in`` 条目的形态：``<名单名>-<档次>``，如 ``jufo-3`` / ``ki-jl-2``。
#:
#: 名单名用惰性量词，使 ``ki-jl-2`` 能被拆成 ``ki-jl`` + ``2`` 而不是 ``ki`` + ``jl-2``。
_TIER_ENTRY_RE = re.compile(r"^(?P<list>[a-z][a-z0-9-]*?)-(?P<level>\d+)$")


def derive_journal_tier(listed_in: Any) -> tuple[str, list[str]]:
    """从 OpenAlex ``listed_in`` 派生期刊档次。

    取 :data:`TIER_LISTS` 三套名单里的**最高档**作为结论；``basis`` 列出所有达到该档的
    条目，使结论可审计——单看一个 ``top`` 无从知道它是芬兰的评议还是挪威的评议。

    Args:
        listed_in: OpenAlex ``sources.listed_in``，形如
            ``["cwts-core", "jufo-3", "ki-jl-2", "norway-2"]``。容错：非可迭代对象、
            含非字符串元素、未知名单名、越界档次一律忽略而不抛异常。

    Returns:
        ``(tier, basis)``。``tier`` ∈ ``{"top", "leading", "basic", ""}``；``""`` 表示
        **无从判断**（三套名单都没收录），与「判定为低档」语义不同，故不用 ``"basic"``
        兜底。``basis`` 在无结论时为 ``[]``。
    """
    if listed_in is None or isinstance(listed_in, str):
        return "", []
    try:
        items = list(listed_in)
    except TypeError:  # 不可迭代
        return "", []

    recognized: list[tuple[int, str, str]] = []  # (名单声明序, 条目原文, 该条目的档次)
    for raw in items:
        if not isinstance(raw, str):
            continue
        m = _TIER_ENTRY_RE.match(raw.strip().lower())
        if not m:
            continue
        name = m.group("list")
        levels = TIER_LISTS.get(name)
        if not levels:
            continue
        tier = levels.get(int(m.group("level")))
        if tier:
            recognized.append((_TIER_LIST_ORDER.index(name), m.group(0), tier))

    if not recognized:
        return "", []

    best_rank = max(TIER_RANK[t] for _, _, t in recognized)
    basis = [
        token
        for _, token, _ in sorted(
            (r for r in recognized if TIER_RANK[r[2]] == best_rank),
            key=lambda r: (r[0], r[1]),
        )
    ]
    return _RANK_TO_TIER[best_rank], basis


# ===========================================================================
# 文本 / 格式化工具
# ===========================================================================
def strip_accents(s: str) -> str:
    """去声调（NFKD 分解后剔除组合记号），便于跨源作者名/标题比对。

    ``'Büttner'`` → ``'Buttner'``（**转写**而非删除；删除会得到 ``'Bttner'``）。
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", s or "")
        if not unicodedata.combining(c)
    )


def normalize_last_name(name: str | None) -> str:
    """从各种作者名形态提取「姓」并归一化（**转写**声调 / 小写 / 去标点）。

    兼容 ``'Zhu, Zheng'``（逗号前为姓）、``'Zheng Zhu'``（末词为姓）、
    ``'Zheng'``（单词名）三种形态；空值返回 ``''``。

    这是全项目唯一的权威实现：原先 ``arxiv_client`` / ``openalex_client`` 各有一份
    ``_guess_last_name``，它们用 ``re.sub(r"[^a-zA-Z]", "", ...)`` **删除**声调
    （``Büttner`` → ``bttner``），与本函数的**转写**语义相反，导致同一位作者在不同源
    派生出不同的姓。现已统一到本函数。
    """
    if not name:
        return ""
    s = strip_accents(str(name)).strip()
    if not s:
        return ""
    if "," in s:  # 'Last, First' → 取逗号前
        s = s.split(",", 1)[0]
    else:  # 'First M. Last' → 取末词
        parts = [p for p in re.split(r"\s+", s) if p]
        s = parts[-1] if parts else s
    s = re.sub(r"[^a-zA-Z]", "", s).lower()
    return s


def slugify(text: str, max_len: int = 48) -> str:
    """把任意标题转成文件名安全的 slug（保留中文，标点/空白折叠为 ``-``）。"""
    if not text:
        return "untitled"
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return (text[:max_len].rstrip("-")) or "untitled"


def fmt_size(nbytes: float) -> str:
    """把字节数格式化为可读字符串（B/KB/MB/GB/TB）。

    原 ``research._fmt_bytes`` 与 ``cache_manager._fmt_size`` 是逐字节相同的两份实现，
    现统一到此。
    """
    n = float(nbytes or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ===========================================================================
# frontmatter I/O
# ===========================================================================
#: 匹配 markdown 开头的 ``--- ... ---`` frontmatter 块，group(1)=YAML 文本，group(2)=正文。
#:
#: 两处细节是「正文逐字节不变」这条不变量的关键：
#:
#: 1. 定界行用 ``[ \t]*`` 而非 ``\s*``——``\s`` 含 ``\n``，贪婪匹配会**吃掉正文开头的空行**
#:    （``2023_fang`` 的 ``---`` 与 ``# 标题`` 之间就有一个空行），写回时该空行会消失。
#: 2. 闭合 ``---`` 后只吃掉**一个**换行（``\n?``），其余（含空行）全部留在 group(2) 里。
#:    :func:`render_note` 恰好补回这一个换行，因此 ``split_note`` → ``render_note`` 往返字节等价。
_FM_BLOCK_RE = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)$", re.DOTALL)


def split_note(text: str) -> tuple[dict[str, Any], str]:
    """把一份 note 拆成 ``(frontmatter dict, body)``。

    无 frontmatter（或 YAML 解析失败）时返回 ``({}, 原文)``——**不抛异常**，
    因为调用方（``index --fix`` / ``referenced_paths``）要能优雅跳过坏文件。
    body 原样返回，不做任何规范化，以便调用方逐字节写回。
    """
    text = text or ""
    m = _FM_BLOCK_RE.match(text)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}, text
    if not isinstance(fm, dict):
        return {}, text
    return fm, m.group(2)


def load_frontmatter(text: str) -> dict[str, Any]:
    """只取一份 note 的 frontmatter dict（:func:`split_note` 的便捷封装）。

    与旧的极简解析器不同，本函数用 ``yaml.safe_load``，因此 **block-style 与 flow-style
    的列表都能正确读出**（旧实现跳过所有以 ``-`` 开头的行，``authors`` 之类会全部丢失）。
    """
    return split_note(text)[0]


def _order_fields(fm: dict[str, Any]) -> dict[str, Any]:
    """按 :data:`FIELD_ORDER` 重排已知字段，未知字段按插入序追加在后。"""
    ordered: dict[str, Any] = {}
    for key in FIELD_ORDER:
        if key in fm:
            ordered[key] = fm[key]
    for key, value in fm.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


class _NoteDumper(yaml.SafeDumper):
    """让 block-style 序列相对父键缩进两格。

    PyYAML 默认把块序列项发在与父键**同一列**（``authors:\\n- A``），而本项目既有笔记
    与 ``templates/paper_note.md`` 用的都是缩进式（``authors:\\n  - A``）。不纠正的话
    ``index --fix`` 会在每一篇笔记上制造一批纯粹的缩进 diff 噪声。
    """

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow, False)


#: 行宽上限。故意取得很大以**禁止长标量折行**：一个字段一行才能保持 git diff 可读、
#: ``Select-String`` 可搜。PyYAML 默认（或 ``width=100``）会把超过宽度的标题折成多行，
#: 往返后值不变但文件形状全变，与 WP-C 的「diff 只应包含值变化」验证标准相背。
_DUMP_WIDTH = 4096


def dump_frontmatter(fm: dict[str, Any]) -> str:
    """把 frontmatter dict 序列化为 YAML 文本（block style，末尾带换行）。

    固定参数：``allow_unicode=True``（中文不转义）、``sort_keys=False``（按
    :data:`FIELD_ORDER` 而非字母序）、``default_flow_style=False``（列表展开为
    block style，与既有笔记一致）、大行宽（不折行，一字段一行）。

    同一输入永远产生同一字节序列（幂等）。空 dict 返回 ``""``。
    """
    if not fm:
        return ""
    return yaml.dump(
        _order_fields(dict(fm)),
        Dumper=_NoteDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=_DUMP_WIDTH,
    )


def render_note(fm: dict[str, Any], body: str) -> str:
    """把 frontmatter 与 body 拼成完整 note 文本。

    ``body`` **原样**写入，不做任何规范化——这是「正文在任何自动写入路径下逐字节不变」
    这条不变量的执行点。
    """
    return f"---\n{dump_frontmatter(fm)}---\n{body}"


def _author_token(authors: Any) -> str:
    """从 ``authors`` 列表取首位作者的姓名末段（供 :func:`note_filename` 兜底）。

    ``normalize_last_name`` 对纯非拉丁字符姓名（如中文 ``'朱某某'``）返回 ``''``，
    若不兜底会让文件名退化成 ``{year}_unknown_{slug}.md``。此处直接取原始末段，
    交由 :func:`slugify` 保留中文。
    """
    if not isinstance(authors, (list, tuple)) or not authors:
        return ""
    a0 = authors[0]
    if isinstance(a0, dict):
        a0 = a0.get("name") or a0.get("display_name") or ""
    a0 = str(a0 or "").strip()
    if not a0:
        return ""
    if "," in a0:
        return a0.split(",", 1)[0].strip()
    parts = [p for p in re.split(r"\s+", a0) if p]
    return parts[-1] if parts else a0


def note_filename(fm: dict[str, Any]) -> str:
    """``papers/`` 命名规范：``{year}_{firstauthor_lastname}_{slug}.md``。"""
    year = fm.get("year") or "nd"
    last = str(fm.get("first_author_last_name") or "").strip()
    if not last:
        last = _author_token(fm.get("authors"))
    last = slugify(last or "unknown", 24)
    slug = slugify(str(fm.get("short_title") or fm.get("title") or "paper"), 40)
    return f"{year}_{last}_{slug}.md"


def template_defaults(name: str = "paper_note.md") -> dict[str, Any]:
    """读取 ``templates/<name>`` 的 frontmatter 作为字段默认值表。

    直接解析模板文件而非硬编码一份副本，保证默认值永不与模板漂移。
    模板缺失或解析失败时返回 ``{}``（静默降级，调用方按「无默认值」处理）。
    """
    path = settings.module_dir / "templates" / name
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    fm, _ = split_note(text)
    return fm


def template_body(name: str = "paper_note.md") -> str:
    """读取 ``templates/<name>`` 的正文骨架（frontmatter 之后的部分）。缺失返回 ``""``。"""
    path = settings.module_dir / "templates" / name
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    return split_note(text)[1]


#: 报告补齐键时的排序权重：先按 :data:`FIELD_ORDER`，未知键靠后并按字典序。
def _field_rank(key: str) -> tuple[int, str]:
    try:
        return (FIELD_ORDER.index(key), key)
    except ValueError:
        return (len(FIELD_ORDER), key)


def normalize_frontmatter(
    fm: dict[str, Any], *, template: str = "paper_note.md"
) -> tuple[dict[str, Any], list[str]]:
    """规范化 frontmatter：补齐模板定义的缺失键，再按 :data:`FIELD_ORDER` 排序。

    与 :func:`merge_frontmatter` **不可互换**，两者补的东西不一样：

    * ``merge_frontmatter`` 合并的是**另一个数据源的实际值**，只填空键，并且跳过
      空的 incoming 值（把 ``None`` 塞进一个缺失的键只是噪声）；
    * 本函数补的是**键的存在性**——即使模板默认值是空的也补。因为
      ``templates/paper_note.md`` 定义了 paper note 的完整字段契约：缺键的笔记会让
      ``fm["x"]`` 抛 ``KeyError``、让下游按字段取值的代码（``INDEX.md`` 的列、
      ``cache_manager.REFERENCED_FIELDS``、未来的期刊指标层）静默地拿不到东西。

    **只增不改**：``fm`` 里已有的键（包括值为 ``None`` / ``""`` / ``[]`` 的）一律保留原样，
    用户手填的 ``my_rating`` / ``status`` / ``related_to_my_work`` 永不被顶掉。

    Returns:
        ``(normalized, added_keys)``。``added_keys`` 按 :data:`FIELD_ORDER` 排序，
        使 ``index --fix`` 的报告稳定可读；为空且字段序已一致 ⇒ 调用方无需重写文件。
    """
    normalized = dict(fm or {})
    added: list[str] = []
    for key, value in template_defaults(template).items():
        if key not in normalized:
            normalized[key] = value
            added.append(key)
    added.sort(key=_field_rank)
    return _order_fields(normalized), added


# ===========================================================================
# 合并
# ===========================================================================
def _is_blank(v: Any) -> bool:
    """判定一个 frontmatter 值是否「空」（可被机器值填补）。

    只有 ``None`` / ``""`` / ``[]`` / ``{}`` 算空。注意 ``0`` 与 ``False`` **不是**空值
    ——它们是用户或数据源给出的真实取值，覆盖掉就是数据丢失。
    """
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, (list, dict, tuple, set)):
        return len(v) == 0
    return False


def merge_frontmatter(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """把 ``incoming`` 合并进 ``existing``，**只填空键，绝不覆盖非空值**。

    Returns:
        ``(merged, changed_keys)``。``merged`` 是新 dict（不修改入参）；
        ``changed_keys`` 是真正发生了值变化的键，按 ``incoming`` 的遍历序排列，
        供调用方写入 Changelog。``changed_keys`` 为空 ⇒ 调用方应完全不触碰文件
        （保留 mtime），避免无谓的写入。

    ``incoming`` 里的空值不会写进 ``merged``：把 ``None`` 填进一个缺失的键只是制造
    噪音，字段补全由 :func:`template_defaults` + ``index --fix`` 负责。
    """
    merged = dict(existing or {})
    changed: list[str] = []
    for key, value in (incoming or {}).items():
        if _is_blank(value):
            continue
        if _is_blank(merged.get(key)):
            merged[key] = value
            changed.append(key)
    return merged, changed


#: 匹配 ``## Changelog`` 段标题（正文里的二级标题），含带编号的 ``## 8. Changelog``。
#:
#: 标题后的空白只允许 ``[ \t]*`` 而**不是** ``\s*``：``\s`` 含 ``\n``，会把标题自己的
#: 换行吃进匹配里，使 ``m.end()`` 落到下一行开头，装配时凭空多出一个空行。
#:
#: 编号前缀是可选的：``templates/paper_note.md`` 与 ``shortlist.md`` 用裸标题，而
#: ``templates/review_note.md`` 的 Changelog 是第 8 节。不认编号会让综述笔记每次
#: ``review sync`` 都在文末新建一个重复的 ``## Changelog`` 段，而原有的第 8 节永远为空。
_CHANGELOG_HEAD_RE = re.compile(
    r"^##[ \t]*(?:\d+[\.、\)][ \t]*)?Changelog[ \t]*$", re.MULTILINE
)
#: 匹配 Changelog 段的结束边界：下一个同级或更高级标题（``#`` / ``##``）。
#: ``###`` 之类的三级标题属于 Changelog 段内部，不作边界。
_NEXT_HEADING_RE = re.compile(r"^#{1,2}[ \t]+\S", re.MULTILINE)


def append_changelog(body: str, line: str) -> str:
    """在正文的 ``## Changelog`` 段末追加一行；无该段则在文末新建一段。

    只**追加**，绝不重写既有行——Changelog 是人工与机器共写的审计轨迹。
    body 的其余部分逐字节保留。``line`` 为空时原样返回 body。
    """
    body = body or ""
    line = str(line).rstrip("\n")
    if not line:
        return body

    m = _CHANGELOG_HEAD_RE.search(body)
    if m is None:
        prefix = body if (not body or body.endswith("\n")) else body + "\n"
        return f"{prefix}\n## Changelog\n\n{line}\n"

    start = m.end()  # 紧跟标题文字之后，标题自己的换行仍在 section 里
    rest = body[start:]
    nxt = _NEXT_HEADING_RE.search(rest)
    end = start + nxt.start() if nxt else len(body)

    section = body[start:end]
    core = section.strip("\n")  # 段内实际内容（可能为空）
    # 尾随换行必须原样保留：Changelog 不是最后一段时，它承担着与下一标题的空行分隔。
    tail = section[len(section.rstrip("\n")) :] or "\n"
    gap = "\n\n"  # 标题与首条记录之间恒定一个空行
    new_section = f"{gap}{core}\n{line}{tail}" if core.strip() else f"{gap}{line}{tail}"
    return body[:start] + new_section + body[end:]
