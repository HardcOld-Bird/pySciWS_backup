"""mineru_cloud 协议单测：全部 mock ``_request``，零真实网络（外部 API 不进机械层）。"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from pysci.skills.literature_research import mineru_cloud
from pysci.skills.literature_research.mineru_cloud import MinerUError


def _zip_bytes(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def test_token_prefers_env(monkeypatch):
    monkeypatch.setenv("MINERU_TOKEN", "tok-env")
    assert mineru_cloud._token() == "tok-env"


def test_token_falls_back_to_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MINERU_TOKEN=tok-file\nOTHER=1\n", encoding="utf-8")
    monkeypatch.setattr(mineru_cloud, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("MINERU_TOKEN", raising=False)
    assert mineru_cloud._token() == "tok-file"


def test_token_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(mineru_cloud, "PROJECT_ROOT", tmp_path)  # tmp 下无 .env
    monkeypatch.delenv("MINERU_TOKEN", raising=False)
    with pytest.raises(MinerUError, match="MINERU_TOKEN 缺失"):
        mineru_cloud._token()


def test_extract_markdown_full_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERU_TOKEN", "tok")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    calls: list[str] = []
    polls = 0

    def fake_request(method, url, data=None, token="", content_type=None):
        nonlocal polls
        calls.append(f"{method} {url}")
        if url.endswith("/file-urls/batch"):
            assert method == "POST" and token == "tok"
            assert content_type == "application/json"  # JSON API 调用固定头
            payload = json.loads(data)
            assert payload["files"] == [{"name": "paper.pdf", "data_id": "paper"}]
            assert payload["model_version"]  # v4 请求体固定带 model_version
            body = {"code": 0, "data": {"batch_id": "bt1", "file_urls": ["https://up/1"]}}
            return json.dumps(body).encode()
        if url == "https://up/1":
            assert method == "PUT" and data.startswith(b"%PDF")
            assert content_type == ""  # 预签名上传须空 Content-Type（抑制 urllib 注入）
            return b""
        if url.endswith("/extract-results/batch/bt1"):
            polls += 1
            state = "done" if polls >= 2 else "running"  # 第 1 次轮询未完成，第 2 次完成
            # 真实形态：state 在文件项内，结果链接字段为 full_zip_url（非 md_url）
            results = [{"file_name": "paper.md", "state": state, "full_zip_url": "https://dl/res.zip"}]
            body = {"code": 0, "data": {"batch_id": "bt1", "extract_result": results}}
            return json.dumps(body).encode()
        if url == "https://dl/res.zip":
            return _zip_bytes("paper/full.md", "# title\n\nbody")
        raise AssertionError(f"意外请求：{method} {url}")

    monkeypatch.setattr(mineru_cloud, "_request", fake_request)
    md = mineru_cloud.extract_markdown(pdf, poll_interval=0)
    assert md == "# title\n\nbody"
    assert [c.split(" ")[0] for c in calls] == ["POST", "PUT", "GET", "GET", "GET"]


def test_extract_markdown_failed_state_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERU_TOKEN", "tok")
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF")

    def fake_request(method, url, data=None, token="", content_type=None):
        if url.endswith("/file-urls/batch"):
            return json.dumps({"data": {"batch_id": "bt1", "file_urls": ["https://up/1"]}}).encode()
        if url.startswith("https://up/"):
            return b""
        return json.dumps(
            {"data": {"batch_id": "bt1", "extract_result": [{"file_name": "p.md", "state": "failed", "err_msg": "boom"}]}}
        ).encode()

    monkeypatch.setattr(mineru_cloud, "_request", fake_request)
    with pytest.raises(MinerUError, match="提取失败"):
        mineru_cloud.extract_markdown(pdf, poll_interval=0)


def test_submit_batch_url_count_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERU_TOKEN", "tok")
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(
        mineru_cloud,
        "_request",
        lambda method, url, data=None, token="", content_type=None: json.dumps(
            {"data": {"batch_id": "b", "file_urls": []}}
        ).encode(),
    )
    with pytest.raises(MinerUError, match="file_urls 数量"):
        mineru_cloud.submit_batch([pdf])


def test_extract_markdown_missing_md_url_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERU_TOKEN", "tok")
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF")

    def fake_request(method, url, data=None, token="", content_type=None):
        if url.endswith("/file-urls/batch"):
            return json.dumps({"data": {"batch_id": "bt1", "file_urls": ["https://up/1"]}}).encode()
        if url.startswith("https://up/"):
            return b""
        return json.dumps(
            {"data": {"batch_id": "bt1", "extract_result": [{"file_name": "p.md", "state": "done", "full_zip_url": ""}]}}
        ).encode()

    monkeypatch.setattr(mineru_cloud, "_request", fake_request)
    with pytest.raises(MinerUError, match="下载链接"):
        mineru_cloud.extract_markdown(pdf, poll_interval=0)
