"""MinerU 云端 PDF→Markdown 的薄 HTTP 封装（本机不跑本地提取，用户裁决）。

ADR（自研理由，2026-11-17）：本机硬件不足，PDF→MD 走 MinerU 云端 API（选型报告 §2 风险注记）；
MinerU 官方只提供裸 REST（提交批次→预签名 URL 上传→轮询→下载结果 zip），社区无薄客户端。
本模块仅封装该协议（stdlib urllib，零第三方依赖），不做缓存/重试编排。协议按官方 Open API v4
（``mineru.net/doc/docs/index_en`` 批量上传-解析流程实测，2026-11-17）；升级前先跑
``tests/skills/literature_research/test_mineru_cloud.py`` 的协议单测。

真实 Open API v4 批量协议（已实测核对）：

- 提交 ``POST /api/v4/file-urls/batch``，body ``{"files":[{"name","data_id"}],"model_version":...}``
  → ``{"data":{"batch_id","file_urls":[...]}}``；``data_id`` 限字母数字``_- .``、≤128 字符；
- 上传 ``PUT <file_url>`` 原始字节（预签名 URL，不带 Authorization、不带 Content-Type）；
- 轮询 ``GET /api/v4/extract-results/batch/{batch_id}`` → ``{"data":{"batch_id","extract_result":[
  {"file_name","state","full_zip_url",...}]}}``；``state`` 在**每个文件项内**（done/pending/running/
  converting/failed），结果 zip 链接是 **``full_zip_url``**。

响应解析对 ``extract_result``/``extract_results``、``full_zip_url``/``md_url`` 及 state 位置（文件项内
或顶层）保持容错——官方文档与各二手源字段名略有出入，且真实配额只够一次提取，故最小容错以保首跑成功
（非"废弃路径回退"，而是第三方响应字段归一）。

用法::

    from pysci.skills.literature_research.mineru_cloud import extract_markdown
    md = extract_markdown(Path("paper.pdf"))  # 读 .env 的 MINERU_TOKEN
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

from pysci.paths import PROJECT_ROOT

#: 默认 API 根（环境变量 ``MINERU_API_BASE`` 可覆盖）。
DEFAULT_API_BASE = "https://mineru.net"
_SUBMIT = "/api/v4/file-urls/batch"
_RESULT = "/api/v4/extract-results/batch/"


class MinerUError(RuntimeError):
    """认证/协议/配额失败，或提取结果不合格。"""


def _token() -> str:
    """MINERU_TOKEN：环境变量优先，其次 ``<PROJECT_ROOT>/.env``（密钥不入库）。"""
    token = os.environ.get("MINERU_TOKEN", "").strip()
    if not token:
        dotenv = PROJECT_ROOT / ".env"
        if dotenv.is_file():
            for line in dotenv.read_text(encoding="utf-8").splitlines():
                if line.startswith("MINERU_TOKEN="):
                    token = line.partition("=")[2].strip()
    if not token:
        raise MinerUError("MINERU_TOKEN 缺失（环境变量或 <PROJECT_ROOT>/.env）")
    return token


def _request(method: str, url: str, data: bytes | None = None, token: str = "", content_type: str | None = None) -> bytes:
    """单次 HTTP 请求（单测 monkeypatch 点）；网络/非 2xx 失败抛 MinerUError。

    ``content_type``：JSON API 调用需 ``application/json``（官方示例固定带此头，缺失会让服务端
    退回表单绑定、报 ``type mismatch for field "files"``）；预签名 PUT 上传/下载不传此头。
    """
    headers = {"User-Agent": "pySciWS/0.1 (research-tool)"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if content_type is not None:
        # 显式设头（含空串）：抑制 urllib 对带 body 请求自动注入的
        # ``application/x-www-form-urlencoded``。预签名 PUT 上传须发空 Content-Type
        # （OSS 签名按无 Content-Type 计算，注入表单类型会导致 403 SignatureDoesNotMatch）。
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read()
    except OSError as exc:
        raise MinerUError(f"MinerU 请求失败：{url}：{exc}") from exc


def _http_json(method: str, url: str, payload: dict | None = None, token: str = "") -> dict:
    """JSON 请求封装；响应含 ``data`` 包装时解包返回，否则原样返回。"""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    body = json.loads(_request(method, url, data, token, content_type="application/json").decode("utf-8"))
    return body.get("data", body) if isinstance(body, dict) else {}


def submit_batch(pdf_paths: list[Path], *, api_base: str | None = None) -> dict:
    """提交一批 PDF：申请预签名上传 URL 并逐个 PUT 上传。

    Returns:
        ``{"batch_id": str, "uploads": [(path, url), ...]}``。
    """
    base = api_base or os.environ.get("MINERU_API_BASE") or DEFAULT_API_BASE
    payload = {
        "files": [{"name": p.name, "data_id": p.stem} for p in pdf_paths],
        # v4 请求体固定带 model_version（官方示例均含；vlm 为推荐模型），可用 MINERU_MODEL_VERSION 覆盖。
        "model_version": os.environ.get("MINERU_MODEL_VERSION", "vlm"),
    }
    data = _http_json("POST", base + _SUBMIT, payload, _token())
    urls = data.get("file_urls") or []
    if len(urls) != len(pdf_paths):
        raise MinerUError(f"file_urls 数量与 PDF 数不符：{len(urls)} != {len(pdf_paths)}")
    uploads = list(zip(pdf_paths, urls, strict=True))
    for path, url in uploads:
        _request("PUT", url, path.read_bytes(), content_type="")  # 空 Content-Type，见 _request 注释
    return {"batch_id": str(data.get("batch_id", "")), "uploads": uploads}


def fetch_batch(batch_id: str, *, api_base: str | None = None) -> dict:
    """查询批次提取结果（``state``=queued/running/done/failed + ``extract_results``）。"""
    base = api_base or os.environ.get("MINERU_API_BASE") or DEFAULT_API_BASE
    return _http_json("GET", base + _RESULT + batch_id, token=_token())


def _first_result(data: dict) -> dict:
    """取批次首个文件的结果项（真实键 ``extract_result``；兼容 ``extract_results``）。"""
    results = data.get("extract_result") or data.get("extract_results") or []
    return results[0] if isinstance(results, list) and results else {}


def _item_state(item: dict, data: dict) -> str:
    """文件项内 state（真实形态）优先，退化到顶层 ``data.state``。"""
    return str(item.get("state") or data.get("state") or "").strip().lower()


def _result_url(item: dict) -> str:
    """结果 zip 下载链接（真实字段 ``full_zip_url``；兼容 ``md_url``）。"""
    return str(item.get("full_zip_url") or item.get("md_url") or "")


def extract_markdown(
    pdf_path: Path,
    *,
    api_base: str | None = None,
    poll_interval: float = 10.0,
    timeout: float = 900.0,
) -> str:
    """单文件全链：提交→上传→轮询→下载结果 zip→返回 ``full.md`` 文本。

    Raises:
        MinerUError: 提取 failed、结果缺 md_url、zip 内无 .md、或轮询超时。
    """
    batch = submit_batch([pdf_path], api_base=api_base)
    deadline = time.monotonic() + timeout
    while True:
        data = fetch_batch(batch["batch_id"], api_base=api_base)
        state = _item_state(_first_result(data), data)
        if state == "done":
            break
        if state == "failed":
            item = _first_result(data)
            err = str(item.get("err_msg") or data.get("err_msg") or "").strip()
            raise MinerUError(f"MinerU 提取失败：{pdf_path.name} {err}".strip())
        if time.monotonic() > deadline:
            raise MinerUError(f"MinerU 轮询超时（{timeout}s）：{pdf_path.name}（last state={state}）")
        time.sleep(poll_interval)
    md_url = _result_url(_first_result(data))
    if not md_url:
        raise MinerUError(f"提取结果缺下载链接（full_zip_url）：{pdf_path.name}")
    with zipfile.ZipFile(io.BytesIO(_request("GET", md_url))) as zf:
        names = [n for n in zf.namelist() if n.endswith(".md")]
        if not names:
            raise MinerUError(f"结果 zip 内无 .md：{pdf_path.name}")
        name = next((n for n in names if n.endswith("full.md")), names[0])
        return zf.read(name).decode("utf-8")
