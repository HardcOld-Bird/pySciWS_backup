"""双云客户端的机械断言：请求拼装（双端点）/ 401 优雅降级 / 响应归一 / 落盘。

全部 mock（假 session），不发网络；真实出图冒烟见 smoke_real.py（hardware，key 更新后跑）。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from pysci.skills.image_generation import guardrails, providers


# ---------------------------------------------------------------------------
# mock 基件
# ---------------------------------------------------------------------------
class FakeResponse:
    """requests.Response 的最小替身。"""

    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """记录 POST 调用并返回预置响应。"""

    def __init__(self, resp: FakeResponse):
        self.resp = resp
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return self.resp


def _png_b64() -> str:
    """一张 2x2 RGBA PNG 的 base64（b64_json 响应用）。"""
    buf = io.BytesIO()
    Image.new("RGBA", (2, 2), (255, 0, 0, 128)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png_file(tmp_path: Path, name: str = "ref.png") -> Path:
    p = tmp_path / name
    Image.new("RGB", (4, 4), (10, 20, 30)).save(p, format="PNG")
    return p


# ---------------------------------------------------------------------------
# 请求拼装（双端点）
# ---------------------------------------------------------------------------
def test_ark_payload_t2i_contract():
    """方舟文生图：watermark 显式 false、默认 b64_json、size 透传、模型直传。"""
    payload = providers.build_payload(
        "ark", "  a red circle  ", model="doubao-seedream-5-5", size="2K"
    )
    assert payload["model"] == "doubao-seedream-5-5"
    assert payload["prompt"] == "a red circle"
    assert payload["watermark"] is False
    assert payload["response_format"] == "b64_json"
    assert payload["size"] == "2K"
    assert "image" not in payload


def test_ark_payload_transparent_png():
    """Seedream 5.5 透明底直出：background=transparent + output_format=png。"""
    payload = providers.build_payload(
        "ark", "sticker", model="doubao-seedream-5-5", transparent=True
    )
    assert payload["background"] == "transparent"
    assert payload["output_format"] == "png"


def test_ark_payload_sequential_group():
    """方舟组图：sequential_image_generation=auto + max_images（护栏判定）。"""
    payload = providers.build_payload(
        "ark", "story board", model="doubao-seedream-5-0-260128", sequential=True, n=3
    )
    assert payload["sequential_image_generation"] == "auto"
    assert payload["max_images"] == 3


def test_ark_payload_i2i_data_uri(tmp_path):
    """图生图：本地图 → data URI 数组进 image 字段。"""
    ref = _png_file(tmp_path)
    payload = providers.build_payload("ark", "make it blue", images=[ref])
    assert isinstance(payload["image"], list)
    assert payload["image"][0].startswith("data:image/png;base64,")


def test_siliconflow_payload_contract(tmp_path):
    """硅基文生图：image_size/batch_size/seed 映射（seed 兼容档全档生效）。"""
    payload = providers.build_payload(
        "siliconflow",
        "an island",
        model="Qwen/Qwen-Image",
        size="1328x1328",
        n=4,
        seed=42,
    )
    assert payload["model"] == "Qwen/Qwen-Image"
    assert payload["watermark"] is False
    assert payload["image_size"] == "1328x1328"
    assert payload["batch_size"] == 4
    assert payload["seed"] == 42
    assert "response_format" not in payload  # schema 无此字段（仅 url，1h 失效）


def test_siliconflow_image_key_mapping(tmp_path):
    """硅基图生图字段：Kontext-max/pro 用 input_image，Qwen-Image-Edit 用 image。"""
    ref = _png_file(tmp_path)
    p_edit = providers.build_payload(
        "siliconflow", "edit", model="Qwen/Qwen-Image-Edit", images=[ref]
    )
    assert p_edit["image"].startswith("data:image/png;base64,")
    p_kontext = providers.build_payload(
        "siliconflow",
        "edit",
        model="black-forest-labs/FLUX.1-Kontext-max",
        images=[ref],
    )
    assert p_kontext["input_image"].startswith("data:image/png;base64,")


def test_siliconflow_rejects_b64_response_format():
    """硅基无 response_format=b64_json：显式报错并指引立即下载（护栏契约）。"""
    with pytest.raises(providers.ProviderError, match="立即下载"):
        providers.build_payload(
            "siliconflow", "x", model="Qwen/Qwen-Image", response_format="b64_json"
        )


def test_payload_guardrails_seed_and_budget():
    """护栏接线：seed 越界报错；张数被成本上限 clamp。"""
    with pytest.raises(guardrails.GuardrailError):
        providers.build_payload("ark", "x", seed=70000)
    payload = providers.build_payload("siliconflow", "x", n=10)
    assert payload["batch_size"] == guardrails.max_images_cap()


def test_empty_prompt_rejected():
    with pytest.raises(providers.ProviderError):
        providers.build_payload("ark", "   ")


def test_unknown_provider_rejected():
    with pytest.raises(providers.ProviderError):
        providers.build_payload("comfyui", "x")


# ---------------------------------------------------------------------------
# 生成调用（mock session）
# ---------------------------------------------------------------------------
def test_generate_ark_request_assembly(monkeypatch):
    """generate() 拼出方舟端点 + Bearer 头 + 请求体（mock，不发网络）。"""
    monkeypatch.setenv("ARK_API_KEY", "test-key-ark")
    sess = FakeSession(
        FakeResponse(200, {"created": 1, "data": [{"b64_json": _png_b64()}],
                           "usage": {"generated_images": 1}})
    )
    resp = providers.generate(
        "ark", "a circle", model="doubao-seedream-5-5", session=sess, size="1K"
    )
    call = sess.calls[0]
    assert call["url"] == "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    assert call["headers"]["Authorization"] == "Bearer test-key-ark"
    assert call["json"]["watermark"] is False
    assert resp.provider == "ark"
    assert len(resp.ok_images) == 1
    assert resp.ok_images[0].to_bytes().startswith(b"\x89PNG")


def test_generate_siliconflow_response_normalized(monkeypatch):
    """硅基响应 {images:[{url}], seed} 归一成统一结果。"""
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key-sf")
    sess = FakeSession(
        FakeResponse(200, {"images": [{"url": "https://sf.example/a.png"}], "seed": 7})
    )
    resp = providers.generate(
        "siliconflow", "island", model="Qwen/Qwen-Image", session=sess
    )
    assert resp.seed == 7
    assert resp.images[0].url == "https://sf.example/a.png"
    assert resp.model == "Qwen/Qwen-Image"


def test_generate_401_graceful_error(monkeypatch):
    """key 过期/无效（401）→ 优雅抛 ProviderError（带处置提示），不裸崩。"""
    monkeypatch.setenv("ARK_API_KEY", "expired-key")
    sess = FakeSession(FakeResponse(401, {"error": {"code": "InvalidApiKey",
                                                    "message": "invalid api key"}}))
    with pytest.raises(providers.ProviderError) as excinfo:
        providers.generate("ark", "x", session=sess)
    err = excinfo.value
    assert err.status_code == 401
    assert "ARK_API_KEY" in err.hint()


def test_generate_401_plain_string_body(monkeypatch):
    """硅基 401 返回纯字符串体（"Invalid token"）也走优雅报错。"""
    monkeypatch.setenv("SILICONFLOW_API_KEY", "bad")
    sess = FakeSession(FakeResponse(401, "Invalid token", text="Invalid token"))
    with pytest.raises(providers.ProviderError) as excinfo:
        providers.generate("siliconflow", "x", session=sess)
    assert excinfo.value.status_code == 401
    assert "SILICONFLOW_API_KEY" in excinfo.value.hint()


def test_generate_missing_key_is_graceful(monkeypatch):
    """key 未配置：401 口径优雅报错，不发请求。"""
    monkeypatch.setenv("ARK_API_KEY", "")
    sess = FakeSession(FakeResponse(200, {}))
    with pytest.raises(providers.ProviderError) as excinfo:
        providers.generate("ark", "x", session=sess)
    assert excinfo.value.status_code == 401
    assert sess.calls == []


def test_per_image_error_not_fatal(monkeypatch):
    """单张失败走 data[i].error：整请求不失败，失败张进 failed_images。"""
    monkeypatch.setenv("ARK_API_KEY", "k")
    sess = FakeSession(
        FakeResponse(200, {
            "data": [
                {"b64_json": _png_b64()},
                {"error": {"code": "X", "message": "one image failed"}},
            ],
            "usage": {"generated_images": 1},
        })
    )
    resp = providers.generate("ark", "x", session=sess)
    assert len(resp.ok_images) == 1
    assert len(resp.failed_images) == 1
    assert resp.failed_images[0].error["message"] == "one image failed"


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def test_save_images_roundtrip(tmp_path, monkeypatch):
    """b64 产出落盘：后缀按真实字节判定（PNG→.png），不覆盖既有文件。"""
    monkeypatch.setenv("ARK_API_KEY", "k")
    resp = providers.GenerationResult(
        provider="ark",
        images=[providers.GeneratedImage(0, b64_json=_png_b64())],
    )
    paths = providers.save_images(resp, tmp_path, stem="gen")
    assert paths[0].suffix == ".png"
    assert paths[0].read_bytes().startswith(b"\x89PNG")
    paths2 = providers.save_images(resp, tmp_path, stem="gen")
    assert paths2[0].name == "gen_1.png"  # 防覆盖


def test_sniff_suffix_by_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG")
    assert providers.sniff_suffix(buf.getvalue()) == ".png"
    assert providers.sniff_suffix(b"\xff\xd8\xff\xe0junk") == ".jpg"
    assert providers.sniff_suffix(b"") == ".png"  # fallback


def test_base_url_env_override(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_BASE_URL", "http://sf.test/v1")
    assert providers.base_url("siliconflow") == "http://sf.test/v1"
