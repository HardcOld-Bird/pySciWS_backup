"""统一配置加载模块。

从项目根 `.env` 读取凭据与设置，并暴露：
- 各服务的 API key / endpoint
- 缓存目录路径（自动创建）
- HTTP 会话工厂（统一 UA、超时、重试）

用法::

    from skills_src.literature_research.tools.config import settings, http_session

    settings.openalex_email
    settings.zotero_user_id
    with http_session() as s:
        r = s.get("https://api.openalex.org/works", params={...})
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
# 本文件位于 <project_root>/skills_src/literature_research/tools/config.py
_THIS_FILE = Path(__file__).resolve()
# parents: [0]=tools  [1]=literature_research  [2]=skills_src  [3]=<project_root>
PROJECT_ROOT: Path = _THIS_FILE.parents[3]
MODULE_DIR: Path = _THIS_FILE.parents[1]  # literature_research 模块根（papers/templates/cache 的父目录）
CACHE_DIR_ENV_DEFAULT: str = "skills_src/literature_research/cache"


def _resolve_cache_dir(raw: str | None) -> Path:
    """将 .env 中的 CACHE_DIR 解析为绝对路径。"""
    if not raw:
        raw = CACHE_DIR_ENV_DEFAULT
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


# ---------------------------------------------------------------------------
# .env 加载
# ---------------------------------------------------------------------------
def _load_dotenv_if_available() -> None:
    """尝试使用 python-dotenv 加载 .env；若未安装则回退到手写解析。

    我们不把 python-dotenv 列为硬依赖，以便脚本在最小环境也能启动。
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        # 允许通过环境变量覆盖
        env_path_override = os.environ.get("PYSCIWS_ENV_FILE")
        if env_path_override:
            env_path = Path(env_path_override)
        else:
            return

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    # 手写回退解析（支持 KEY=VALUE，忽略 # 注释与空行）
    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # 去除行内注释（保守处理：仅当 # 前有空格时才认为是注释）
            if " #" in value:
                value = value.split(" #", 1)[0].strip()
            os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Settings 数据类
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """集中管理所有配置项。字段值全部为 str 或 None，避免类型转换意外。"""

    # --- 路径 ---
    project_root: Path
    module_dir: Path
    cache_dir: Path
    cache_api_responses: Path
    cache_pdfs: Path
    cache_extracted: Path
    cache_html_fulltext: Path
    zotero_data_dir: Path | None

    # --- OpenAlex ---
    openalex_email: str | None

    # --- arXiv ---
    arxiv_user_agent: str | None

    # --- Web of Science ---
    wos_api_key: str | None
    wos_api_secret: str | None
    wos_api_base_url: str

    # --- Zotero ---
    zotero_user_id: str | None
    zotero_api_key: str | None
    zotero_local_api_base: str
    zotero_web_api_base: str

    # --- 其他数据源 ---
    semantic_scholar_api_key: str | None
    elsevier_api_key: str | None

    # --- PDF 抽取 ---
    pdf_extract_backend: str
    mineru_token: str | None

    # --- HTTP ---
    http_timeout: int
    http_max_retries: int
    http_user_agent: str

    # --- 缓存治理 ---
    cache_b_max_age_days: int           # Tier B(api_responses) 清理阈值（天）
    cache_autoclean_interval_days: int  # 自动清理钩子最小间隔（天）
    cache_soft_limit_mb: int            # Tier A 软上限（MB）：stats 告警 + prune 默认目标

    # --- 原始 env 快照（便于调试） ---
    _raw_env: dict[str, str] = field(default_factory=dict, repr=False)

    # ---------- 便捷判断 ----------
    @property
    def wos_ready(self) -> bool:
        """WoS Starter API 是否已配置完毕。

        Starter API 只需 API Key（无需 Secret）。
        若未来升级到完整版 API，再额外要求 Secret。
        """
        return bool(self.wos_api_key)

    @property
    def zotero_web_ready(self) -> bool:
        """Zotero Web API 是否可用（需要 user_id + api_key）。"""
        return bool(self.zotero_user_id and self.zotero_api_key)

    def summary(self) -> str:
        """人类可读的配置摘要，用于日志。敏感字段做脱敏。"""
        def mask(v: str | None) -> str:
            if not v:
                return "(unset)"
            if len(v) <= 6:
                return "***"
            return f"{v[:3]}***{v[-3:]}"

        lines = [
            "=== pySciWS literature_research configuration ===",
            f"project_root        : {self.project_root}",
            f"cache_dir           : {self.cache_dir}",
            f"zotero_data_dir     : {self.zotero_data_dir or '(unset)'}",
            "",
            f"openalex_email      : {self.openalex_email or '(unset)'}",
            f"arxiv_user_agent    : {self.arxiv_user_agent or '(unset)'}",
            "",
            f"wos_api_key         : {mask(self.wos_api_key)}",
            f"wos_api_secret      : {mask(self.wos_api_secret)}",
            f"wos_ready           : {self.wos_ready}",
            "",
            f"zotero_user_id      : {self.zotero_user_id or '(unset)'}",
            f"zotero_api_key      : {mask(self.zotero_api_key)}",
            f"zotero_web_ready    : {self.zotero_web_ready}",
            "",
            f"semantic_scholar    : {mask(self.semantic_scholar_api_key)}",
            f"elsevier            : {mask(self.elsevier_api_key)}",
            "",
            f"pdf_extract_backend : {self.pdf_extract_backend}",
            f"mineru_token        : {mask(self.mineru_token)}",
            f"http_timeout        : {self.http_timeout}s",
            f"http_max_retries    : {self.http_max_retries}",
            "",
            f"cache_b_max_age     : {self.cache_b_max_age_days}d",
            f"cache_autoclean     : {self.cache_autoclean_interval_days}d",
            f"cache_soft_limit    : {self.cache_soft_limit_mb} MB",
            "=========================================",
        ]
        return "\n".join(lines)


def _get_env(key: str, default: str | None = None) -> str | None:
    """从环境变量读取，空串视为未设置。"""
    v = os.environ.get(key)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def _get_env_int(key: str, default: int) -> int:
    v = _get_env(key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        print(f"[config] WARNING: {key}={v!r} is not an int, using default {default}", file=sys.stderr)
        return default


def build_settings() -> Settings:
    """加载 .env 并构造 Settings 对象。缓存目录会自动创建。"""
    _load_dotenv_if_available()

    cache_dir = _resolve_cache_dir(_get_env("CACHE_DIR"))
    cache_api = cache_dir / "api_responses"
    cache_pdfs = cache_dir / "pdfs"
    cache_extracted = cache_dir / "extracted"
    cache_html = cache_dir / "html_fulltext"
    for d in (cache_dir, cache_api, cache_pdfs, cache_extracted, cache_html):
        d.mkdir(parents=True, exist_ok=True)

    zotero_data_raw = _get_env("ZOTERO_DATA_DIR")
    zotero_data_dir = Path(zotero_data_raw).expanduser() if zotero_data_raw else None

    return Settings(
        project_root=PROJECT_ROOT,
        module_dir=MODULE_DIR,
        cache_dir=cache_dir,
        cache_api_responses=cache_api,
        cache_pdfs=cache_pdfs,
        cache_extracted=cache_extracted,
        cache_html_fulltext=cache_html,
        zotero_data_dir=zotero_data_dir,

        openalex_email=_get_env("OPENALEX_EMAIL"),
        arxiv_user_agent=_get_env("ARXIV_USER_AGENT"),

        wos_api_key=_get_env("WOS_API_KEY"),
        wos_api_secret=_get_env("WOS_API_SECRET"),
        wos_api_base_url=_get_env("WOS_API_BASE_URL", "https://api.clarivate.com/apis/wos-starter/v1") or "",

        zotero_user_id=_get_env("ZOTERO_USER_ID"),
        zotero_api_key=_get_env("ZOTERO_API_KEY"),
        zotero_local_api_base=_get_env("ZOTERO_LOCAL_API_BASE", "http://127.0.0.1:23119/api") or "",
        zotero_web_api_base=_get_env("ZOTERO_WEB_API_BASE", "https://api.zotero.org") or "",

        semantic_scholar_api_key=_get_env("SEMANTIC_SCHOLAR_API_KEY"),
        elsevier_api_key=_get_env("ELSEVIER_API_KEY"),

        pdf_extract_backend=(_get_env("PDF_EXTRACT_BACKEND", "auto") or "auto").lower(),
        mineru_token=_get_env("MINERU_TOKEN"),

        http_timeout=_get_env_int("HTTP_TIMEOUT_SECONDS", 30),
        http_max_retries=_get_env_int("HTTP_MAX_RETRIES", 3),
        http_user_agent=_get_env(
            "HTTP_USER_AGENT",
            "pySciWS/0.1 (research-tool; contact via .env)",
        ) or "pySciWS/0.1",

        cache_b_max_age_days=_get_env_int("CACHE_B_MAX_AGE_DAYS", 7),
        cache_autoclean_interval_days=_get_env_int("CACHE_AUTOCLEAN_INTERVAL_DAYS", 7),
        cache_soft_limit_mb=_get_env_int("CACHE_SOFT_LIMIT_MB", 2048),

        _raw_env={k: v for k, v in os.environ.items() if k.startswith(("WOS_", "ZOTERO_", "OPENALEX_", "ARXIV_", "SEMANTIC_", "ELSEVIER_", "PDF_", "HTTP_", "CACHE_", "MINERU_"))},
    )


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
settings: Settings = build_settings()


# ---------------------------------------------------------------------------
# HTTP 会话工厂
# ---------------------------------------------------------------------------
def http_session(retries: int | None = None, *, retry_on_status: bool = True) -> Any:
    """构造一个带自动重试与统一 UA 的 requests.Session。

    使用方式::

        with http_session() as s:
            r = s.get(url, params=...)
            r.raise_for_status()

    参数：
        retries: 传输层重试次数（默认 settings.http_max_retries）。
        retry_on_status: 是否将 429/5xx 作为传输层重试状态。
            - True（默认）：urllib3 自动退避重试；重试耗尽后会抛异常。
            - False：status_forcelist 置空，429/5xx 作为普通响应返回（不抛异常），
              由调用方自行读取 status_code / Retry-After 并控制退避（如 S2 客户端）。

    若未安装 requests，将抛出 ImportError 提示。
    """
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as e:
        raise ImportError(
            "requests is required for http_session(). "
            "Install with: uv sync"
        ) from e

    s = requests.Session()
    retry_count = retries if retries is not None else settings.http_max_retries
    retry = Retry(
        total=retry_count,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504) if retry_on_status else (),
        allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": settings.http_user_agent,
        "Accept": "application/json",
    })
    return s


# ---------------------------------------------------------------------------
# CLI: 打印当前配置摘要
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(settings.summary())
