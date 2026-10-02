"""ark_client 单测：请求体护栏 / 输入编码 / 响应解析 / 落盘 / 错误提示。

**全部离线**——网络层用假 session 顶替（``generate(session=...)``），
既不花钱也不依赖 ARK_API_KEY。真实出图属 hardware，见 ``test_ark_live``（若存在）。
"""

from __future__ import annotations

import base64
import io

import pytest
import requests
from PIL import Image

from pysci.skills.ai_drawing.tools import ark_client as ark

# Model ID 以 2026-10 `GET /models` 实测为准：5.0 主档 ID **没有** lite 后缀。
V50 = "doubao-seedream-5-0-260128"
FLASH = "doubao-seedream-5-0-flash-260915"
PRO = "doubao-seedream-5-0-pro-260628"
V40 = "doubao-seedream-4-0-250828"


def _bytes(fmt: str = "PNG", color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format=fmt)
    return buf.getvalue()


def _b64(fmt: str = "PNG") -> str:
    return base64.b64encode(_bytes(fmt)).decode("ascii")


# ---------------------------------------------------------------------------
# 输入编码
# ---------------------------------------------------------------------------
def test_encode_image_makes_lowercase_data_uri(tmp_path):
    p = tmp_path / "ref.png"
    p.write_bytes(_bytes("PNG"))
    uri = ark.encode_image(p)
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == _bytes("PNG")


def test_encode_image_jpeg_mime(tmp_path):
    p = tmp_path / "ref.jpg"
    p.write_bytes(_bytes("JPEG"))
    assert ark.encode_image(p).startswith("data:image/jpeg;base64,")


def test_encode_image_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ark.encode_image(tmp_path / "nope.png")


def test_normalize_images_none_is_empty():
    assert ark.normalize_images(None, model=V40) == []


def test_normalize_images_accepts_single_string(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_bytes())
    out = ark.normalize_images(str(p), model=V40)  # 单值自动包成列表
    assert len(out) == 1 and out[0].startswith("data:image/png")


def test_normalize_images_passes_url_and_data_uri_through():
    src = ["https://x/a.png", "data:image/png;base64,AAAA", "  "]
    out = ark.normalize_images(src, model=V40)
    assert out == ["https://x/a.png", "data:image/png;base64,AAAA"]  # 空串被丢弃


def test_normalize_images_enforces_pro_limit():
    urls = [f"https://x/{i}.png" for i in range(ark.REF_LIMIT_PRO + 1)]
    with pytest.raises(ark.ArkError, match="上限 10"):
        ark.normalize_images(urls, model=PRO)


def test_normalize_images_enforces_default_limit():
    urls = [f"https://x/{i}.png" for i in range(ark.REF_LIMIT_DEFAULT + 1)]
    with pytest.raises(ark.ArkError, match="上限 14"):
        ark.normalize_images(urls, model=V40)
    # 恰好 14 张合法
    assert len(ark.normalize_images(urls[: ark.REF_LIMIT_DEFAULT], model=V40)) == 14


# ---------------------------------------------------------------------------
# 参数护栏
# ---------------------------------------------------------------------------
def test_check_seed_boundaries():
    ark.check_seed(None)
    ark.check_seed(ark.SEED_RANGE[0])
    ark.check_seed(ark.SEED_RANGE[1])
    for bad in (-1, ark.SEED_RANGE[1] + 1, "5", 1.5):
        with pytest.raises(ark.ArkError):
            ark.check_seed(bad)


def test_check_output_format_png_only_on_main_50():
    """png 只放行 5.0 **主档**；pro/flash/4.x 一律拦下，且报错给出 --live / --extra 两条出路。"""
    ark.check_output_format(None, model=V40)
    ark.check_output_format("jpeg", model=V40)
    ark.check_output_format("png", model=V50)
    for bad in (V40, PRO, FLASH):
        with pytest.raises(ark.ArkError, match="models --live"):
            ark.check_output_format("png", model=bad)


def test_check_group_rejects_pro():
    """5.0-pro 只能出单图，组图要在本地拦下而不是等云端报错。"""
    ark.check_group(model=PRO, sequential=False, max_images=4, n_refs=0)
    with pytest.raises(ark.ArkError, match="不支持组图"):
        ark.check_group(model=PRO, sequential=True, max_images=2, n_refs=0)


def test_check_group_accepts_v40():
    ark.check_group(model=V40, sequential=True, max_images=4, n_refs=2)
    ark.check_group(model=V50, sequential=True, max_images=4, n_refs=2)


def test_check_group_rejects_flash():
    """flash 与 pro 同档只能出单图；真实 ID 不含 'lite'，靠档位标记 + pro/flash 排除判定。"""
    with pytest.raises(ark.ArkError, match="不支持组图"):
        ark.check_group(model=FLASH, sequential=True, max_images=2, n_refs=0)


def test_check_group_total_cap():
    """参考图数 + 产出数 ≤ 15（方舟硬上限）。"""
    with pytest.raises(ark.ArkError, match="超过方舟上限 15"):
        ark.check_group(model=V40, sequential=True, max_images=10, n_refs=8)


def test_sniff_suffix_by_real_bytes():
    assert ark.sniff_suffix(_bytes("PNG")) == ".png"
    assert ark.sniff_suffix(_bytes("JPEG")) == ".jpg"
    assert ark.sniff_suffix(b"not an image", fallback=".bin") == ".bin"


def test_decode_b64_tolerates_data_uri_prefix():
    assert ark._decode_b64(f"data:image/png;base64,{_b64()}") == _bytes("PNG")
    assert ark._decode_b64(f"  {_b64()}  ") == _bytes("PNG")
    with pytest.raises(ark.ArkError):
        ark._decode_b64("!!!not base64!!!")


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------
def test_build_payload_minimal():
    p = ark.build_payload("a red circle", model=V40)
    assert p["model"] == V40
    assert p["prompt"] == "a red circle"
    # 方舟 watermark 默认 true → 必须显式出现在请求体里
    assert p["watermark"] is False
    assert p["response_format"] == "b64_json"
    # 未给的可选字段不该出现（避免把 None 发给云端）
    for k in (
        "seed",
        "size",
        "width",
        "height",
        "image",
        "output_format",
        "sequential_image_generation",
        "max_images",
        "tools",
    ):
        assert k not in p, k


def test_build_payload_strips_prompt():
    assert ark.build_payload("  hi  ", model=V40)["prompt"] == "hi"


def test_build_payload_rejects_blank_prompt():
    with pytest.raises(ark.ArkError, match="prompt 不能为空"):
        ark.build_payload("   ", model=V40)


def test_build_payload_watermark_true_when_asked():
    assert ark.build_payload("p", model=V40, watermark=True)["watermark"] is True


def test_build_payload_size_and_dims():
    p = ark.build_payload("p", model=V40, size="2K")
    assert p["size"] == "2K"
    p = ark.build_payload("p", model=V40, width=1024, height=768)
    assert (p["width"], p["height"]) == (1024, 768)
    assert "size" not in p


def test_build_payload_seed_included_only_when_given():
    assert "seed" not in ark.build_payload("p", model=V40)
    assert ark.build_payload("p", model=V40, seed=7)["seed"] == 7


def test_build_payload_images_key(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_bytes())
    payload = ark.build_payload("p", model=V40, images=[str(p), "https://x/b.png"])
    assert len(payload["image"]) == 2
    assert payload["image"][0].startswith("data:image/png;base64,")
    assert payload["image"][1] == "https://x/b.png"


def test_build_payload_group_fields():
    p = ark.build_payload("p", model=V40, sequential=True, max_images=4)
    assert p["sequential_image_generation"] == "auto"
    assert p["max_images"] == 4


def test_build_payload_group_count_clamped_by_guard():
    """非组图时张数被成本护栏 max_images 夹取（settings 默认 4）。"""
    p = ark.build_payload("p", model=V40, sequential=True, n=99)
    assert p["max_images"] <= ark.ABS_MAX_IMAGES


def test_build_payload_tools_and_extra():
    p = ark.build_payload("p", model=V50, tools=[{"type": "web_search"}])
    assert p["tools"] == [{"type": "web_search"}]
    # extra 用于方舟上新参数，可覆盖既有字段（逃生舱）
    p = ark.build_payload("p", model=V50, extra={"thinking": True, "watermark": True})
    assert p["thinking"] is True
    assert p["watermark"] is True


def test_build_payload_propagates_guard_errors():
    with pytest.raises(ark.ArkError):
        ark.build_payload("p", model=V40, seed=99999)
    with pytest.raises(ark.ArkError):
        ark.build_payload("p", model=V40, output_format="png")


# ---------------------------------------------------------------------------
# 响应解析
# ---------------------------------------------------------------------------
def test_parse_response_mixed_ok_and_failed():
    body = {
        "model": V40,
        "created": 1700000000,
        "data": [
            {"b64_json": _b64(), "size": "4x4"},
            {"error": {"code": "InternalError", "message": "boom"}},
            {"url": "https://x/c.png", "size": "4x4"},
            "not-a-dict",  # 异常元素应被跳过而非崩掉
        ],
        "usage": {"generated_images": 2, "output_tokens": 20, "total_tokens": 20},
    }
    r = ark._parse_response(body, request={"model": V40})
    assert r.model == V40 and r.created == 1700000000
    assert len(r.images) == 3
    assert [im.index for im in r.ok_images] == [0, 2]
    assert len(r.failed_images) == 1
    assert r.generated_images == 2


def test_generated_images_falls_back_to_ok_count():
    body = {"data": [{"b64_json": _b64()}]}  # usage 缺失
    r = ark._parse_response(body, request={})
    assert r.generated_images == 1


def test_model_falls_back_to_request():
    r = ark._parse_response({"data": []}, request={"model": V40})
    assert r.model == V40


def test_ark_image_to_bytes_prefers_b64():
    im = ark.ArkImage(index=0, b64_json=_b64())
    assert im.ok and im.to_bytes() == _bytes("PNG")


def test_ark_image_to_bytes_without_source_raises():
    im = ark.ArkImage(index=3)
    assert not im.ok
    with pytest.raises(ark.ArkError, match="第 3 张"):
        im.to_bytes()


def test_ark_image_error_marks_not_ok():
    im = ark.ArkImage(index=0, b64_json=_b64(), error={"code": "X"})
    assert im.ok is False


# ---------------------------------------------------------------------------
# 错误提示
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "code", "needle"),
    [
        (401, None, "ARK_API_KEY"),
        (403, None, "ARK_API_KEY"),
        (404, "ModelNotFound", "开通管理"),
        (400, "InvalidModel", "开通管理"),
        (429, "QuotaExceeded", "额度"),
        (503, "ServiceUnavailable", "方舟服务侧异常"),
    ],
)
def test_ark_error_hint(status, code, needle):
    assert needle in ark.ArkError("boom", status_code=status, code=code).hint()


def test_ark_error_hint_empty_when_unrecognized():
    assert ark.ArkError("something odd", status_code=418).hint() == ""


# ---------------------------------------------------------------------------
# generate（假 session，离线）
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text
        self.content = b""

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class _FakeSession:
    def __init__(self, resp=None, exc=None):
        self.resp = resp
        self.exc = exc
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        if self.exc is not None:
            raise self.exc
        return self.resp

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return self.resp


def test_generate_without_key_raises():
    with pytest.raises(ark.ArkError, match="ARK_API_KEY"):
        ark.generate("p", api_key="")


def test_generate_posts_to_images_endpoint():
    sess = _FakeSession(_FakeResp(body={"data": [{"b64_json": _b64()}]}))
    ark.generate(
        "p",
        api_key="sk-test",
        base_url="https://ark.example.com/api/v3/",
        session=sess,
        model=V40,
    )
    call = sess.calls[0]
    assert call["url"] == "https://ark.example.com/api/v3/images/generations"
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["json"]["model"] == V40
    assert call["timeout"] > 0


def test_generate_parses_ok_response():
    body = {
        "model": V40,
        "created": 1700000000,
        "data": [{"b64_json": _b64(), "size": "4x4"}],
        "usage": {"generated_images": 1, "total_tokens": 10},
    }
    r = ark.generate(
        "p", api_key="k", session=_FakeSession(_FakeResp(body=body)), model=V40
    )
    assert r.model == V40 and r.created == 1700000000
    assert len(r.ok_images) == 1 and r.generated_images == 1
    assert r.request["prompt"] == "p"


def test_generate_http_error_carries_hint():
    sess = _FakeSession(
        _FakeResp(
            status_code=401,
            body={"error": {"code": "InvalidApiKey", "message": "bad key"}},
        )
    )
    with pytest.raises(ark.ArkError) as ei:
        ark.generate("p", api_key="k", session=sess, model=V40)
    err = ei.value
    assert err.status_code == 401 and err.code == "InvalidApiKey"
    assert "ARK_API_KEY" in str(err)  # hint 已拼进 args，打印时直接可见


def test_generate_http_error_without_json_body():
    sess = _FakeSession(
        _FakeResp(status_code=502, body=None, text="<html>bad gateway</html>")
    )
    with pytest.raises(ark.ArkError) as ei:
        ark.generate("p", api_key="k", session=sess, model=V40)
    assert "bad gateway" in str(ei.value)


def test_generate_body_level_error_raises():
    """HTTP 200 但 body 里带 error（方舟的一种失败形态）。"""
    sess = _FakeSession(
        _FakeResp(body={"error": {"code": "QuotaExceeded", "message": "no quota"}})
    )
    with pytest.raises(ark.ArkError) as ei:
        ark.generate("p", api_key="k", session=sess, model=V40)
    assert "额度" in str(ei.value)


def test_generate_non_json_200_raises():
    sess = _FakeSession(_FakeResp(body=None, text="ok"))
    with pytest.raises(ark.ArkError, match="非 JSON"):
        ark.generate("p", api_key="k", session=sess, model=V40)


def test_generate_network_error_wrapped():
    sess = _FakeSession(exc=requests.ConnectionError("dns fail"))
    with pytest.raises(ark.ArkError, match="网络层"):
        ark.generate("p", api_key="k", session=sess, model=V40)


def test_generate_validates_payload_before_network():
    """护栏在发请求前就拦下（不浪费一次网络往返，也不计费）。"""
    sess = _FakeSession(_FakeResp(body={"data": []}))
    with pytest.raises(ark.ArkError):
        ark.generate("p", api_key="k", session=sess, model=V40, seed=999999)
    assert sess.calls == []


def test_generate_snapshot_redacts_data_uri(tmp_path):
    """请求体里的 data URI 可能数 MB → 快照前截断，避免 runs/ 爆盘。"""
    p = tmp_path / "a.png"
    p.write_bytes(_bytes())
    sess = _FakeSession(_FakeResp(body={"data": [{"b64_json": _b64()}]}))
    r = ark.generate("p", api_key="k", session=sess, model=V40, images=[str(p)])
    assert "…<data-uri " in r.request["image"][0]
    assert len(r.request["image"][0]) < 200


# ---------------------------------------------------------------------------
# list_models（假 session，离线）
# ---------------------------------------------------------------------------
def test_list_models_without_key_raises():
    with pytest.raises(ark.ArkError, match="ARK_API_KEY"):
        ark.list_models(api_key="")


def test_list_models_gets_models_endpoint():
    sess = _FakeSession(_FakeResp(body={"data": [{"id": V50}, {"id": PRO}]}))
    rows = ark.list_models(
        api_key="sk-test", base_url="https://ark.example.com/api/v3", session=sess
    )
    assert sess.calls[0]["url"] == "https://ark.example.com/api/v3/models"
    assert sess.calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert [r["id"] for r in rows] == [V50, PRO]


def test_list_models_skips_non_dict_entries():
    sess = _FakeSession(_FakeResp(body={"data": [{"id": V50}, "junk"]}))
    assert ark.list_models(api_key="k", session=sess) == [{"id": V50}]


def test_list_models_bad_shape_raises():
    sess = _FakeSession(_FakeResp(body={"object": "list"}))
    with pytest.raises(ark.ArkError, match="形状异常"):
        ark.list_models(api_key="k", session=sess)


def test_list_models_http_error_carries_message():
    sess = _FakeSession(
        _FakeResp(
            status_code=401,
            body={"error": {"code": "InvalidApiKey", "message": "bad key"}},
        )
    )
    with pytest.raises(ark.ArkError, match="bad key"):
        ark.list_models(api_key="k", session=sess)


# ---------------------------------------------------------------------------
# 落盘 / 快照
# ---------------------------------------------------------------------------
def _resp(n=1, *, fmt="PNG", created=1700000000):
    b64 = _b64(fmt)
    return ark.ArkResponse(
        model=V40,
        created=created,
        images=[ark.ArkImage(index=i, b64_json=b64, size="4x4") for i in range(n)],
        usage={"generated_images": n},
        request={"model": V40, "prompt": "p"},
        raw={"model": V40, "created": created, "data": [{"b64_json": b64}] * n},
    )


def test_save_images_single_uses_plain_stem(tmp_path):
    out = ark.save_images(_resp(1), tmp_path, stem="hero")
    assert [p.name for p in out] == ["hero.png"]  # 后缀由真实字节判定
    assert out[0].read_bytes() == _bytes("PNG")


def test_save_images_multi_uses_indexed_stem(tmp_path):
    out = ark.save_images(_resp(3), tmp_path, stem="set")
    assert [p.name for p in out] == ["set_00.png", "set_01.png", "set_02.png"]


def test_save_images_jpeg_bytes_get_jpg_suffix(tmp_path):
    """方舟 4.x 恒输出 jpeg → 不按 output_format 猜后缀，一律按字节判定。"""
    out = ark.save_images(_resp(1, fmt="JPEG"), tmp_path, stem="x")
    assert out[0].suffix == ".jpg"


def test_save_images_skips_failed(tmp_path):
    r = _resp(2)
    r.images[1] = ark.ArkImage(index=1, error={"code": "X", "message": "bad"})
    out = ark.save_images(r, tmp_path, stem="part")
    assert len(out) == 1


def test_save_images_never_overwrites(tmp_path):
    first = ark.save_images(_resp(1), tmp_path, stem="dup")
    second = ark.save_images(_resp(1), tmp_path, stem="dup")
    assert first[0] != second[0]
    assert second[0].name == "dup_1.png"


def test_save_images_creates_dest_dir(tmp_path):
    out = ark.save_images(_resp(1), tmp_path / "deep" / "dir", stem="x")
    assert out[0].is_file()


def test_save_snapshot_strips_b64_and_records_failures(tmp_path):
    r = _resp(1)
    r.images.append(
        ark.ArkImage(index=1, error={"code": "InternalError", "message": "boom"})
    )
    p = ark.save_snapshot(r, tmp_path, stem="snap")
    assert p.name == "snap.json"
    import json

    doc = json.loads(p.read_text(encoding="utf-8"))
    assert set(doc) == {"request", "response", "usage", "failed", "saved_at"}
    assert "stripped" in doc["response"]["data"][0]["b64_json"]
    assert doc["failed"] == [
        {"index": 1, "error": {"code": "InternalError", "message": "boom"}}
    ]
    assert doc["request"]["prompt"] == "p"


def test_redact_payload_keeps_urls():
    payload = {
        "image": ["https://x/a.png", "data:image/png;base64," + "A" * 500],
        "prompt": "p",
    }
    out = ark._redact_payload(payload, keep=16)
    assert out["image"][0] == "https://x/a.png"
    assert out["image"][1].startswith("data:image/png;b")
    assert "chars>" in out["image"][1]
    assert payload["image"][1] not in out["image"][1]  # 原对象未被就地改动


def test_redact_payload_without_images():
    assert ark._redact_payload({"prompt": "p"}) == {"prompt": "p"}
