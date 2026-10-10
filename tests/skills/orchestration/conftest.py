"""编排测试的共享夹具：把编排状态路径重定向到 tmp，隔离真实项目 state。

被测模块（drain/workflow/dispatch）在函数调用时读取各自的模块级 Path 常量，故
``monkeypatch.setattr`` 这些常量即可把 backlog/lock/runs/suggestions/replies 全部
落到 tmp_path，测试互不污染、不触碰 ``orchestration/state/``。
"""

from __future__ import annotations

import json

import pytest

from pysci.skills.orchestration.tools import dispatch, drain, ledger, registry, workflow

# 2026-10-10 额度事故原文（会话 jsonl 实测）
QUOTA_TEXT = (
    "You've reached your credit usage limit. Please upgrade your "
    "subscription plan to get more resources. Report Issue (input /feedback)"
)


@pytest.fixture
def iso_state(tmp_path, monkeypatch):
    """重定向编排状态路径到 ``tmp_path/state``，返回该 state 目录。"""
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(workflow, "BACKLOG_PATH", state / "backlog.json")
    monkeypatch.setattr(workflow, "SUGGESTIONS_DIR", state / "suggestions")
    monkeypatch.setattr(workflow, "REPLIES_DIR", state / "replies")
    monkeypatch.setattr(dispatch, "REPLIES_DIR", state / "replies")
    monkeypatch.setattr(dispatch, "LINT_RULES_PATH", state / "taskbook-lint.json")
    monkeypatch.setattr(drain, "LOCK_PATH", state / "devops.lock")
    monkeypatch.setattr(drain, "DEVOPS_RUNS_DIR", state / "devops-runs")
    monkeypatch.setattr(drain, "CANCEL_PATH", state / "devops.cancel")
    return state


def seed_backlog(state, items):
    """写入 backlog.json（items 为条目字典列表）。"""
    (state / "backlog.json").write_text(
        json.dumps({"version": 1, "items": items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_backlog(state):
    """读回 backlog.json 的 items 列表。"""
    return json.loads((state / "backlog.json").read_text(encoding="utf-8"))["items"]


def seed_suggestion(state, member="devops", sid=None):
    """在 suggestions/ 落一条待审批建议文件，返回其 id（文件名 stem）。"""
    d = state / "suggestions"
    d.mkdir(parents=True, exist_ok=True)
    sid = sid or f"20261010-120000-{member}"
    (d / f"{sid}.md").write_text(
        f"# 改进建议（{member}，待审批）\n\n建议正文首行。\n", encoding="utf-8"
    )
    return sid


# ---------------------------------------------------------------------------
# do_dispatch 全隔离夹具 + 假 headless（quota/fallback 系列共用）
# ---------------------------------------------------------------------------
@pytest.fixture
def iso_dispatch(tmp_path, monkeypatch):
    """隔离 do_dispatch：tmp registry/ledger + 假 pod + 假 exe。

    默认 registry models 含 ``flash``/``flash_fallback``；用 :func:`set_models`
    改档位映射。成员 ``quotamember`` 默认 model_tier=flash。返回 state 目录。
    """
    state = tmp_path / "state"
    state.mkdir()
    pod = tmp_path / "pods" / "quotamember"
    (pod / "inbox").mkdir(parents=True)
    reg_file = state / "registry.json"
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "exe": None,
                "models": {"max": "", "flash": "", "flash_fallback": ""},
                "members": {
                    "quotamember": {
                        "pod": str(pod),
                        "model_tier": "flash",
                        "max_turns": 3,
                        "timeout_s": 5,
                        "sessions": [],
                    }
                },
                "checks": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    exe = tmp_path / "fake-exe.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("PYSCI_ORCH_EXE", str(exe))
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg_file)
    # Registry.save 的 path 默认值在类定义时绑定，patch 类属性改不了 __init__ 默认——
    # 直接 no-op save，防止测试写真实 registry.json
    monkeypatch.setattr(registry.Registry, "save", lambda self: None)
    monkeypatch.setattr(ledger, "LEDGER_PATH", state / "ledger.jsonl")
    monkeypatch.setattr(dispatch, "REPLIES_DIR", state / "replies")
    monkeypatch.setattr(dispatch, "SUGGESTIONS_DIR", state / "suggestions")
    monkeypatch.setattr(dispatch, "LINT_RULES_PATH", state / "taskbook-lint.json")
    return state


def set_models(state, **models):
    """改写隔离 registry.json 的 models 块（do_dispatch 调用时 Registry.load 重读）。"""
    reg_file = state / "registry.json"
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    data["models"] = models
    reg_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def fake_headless(monkeypatch, responder):
    """monkeypatch dispatch.run_headless；responder(call_index, cmd) -> (rc, out, err)。

    返回 calls 列表（每项为该次 argv），供断言所用模型（``-m`` 后的值）。
    """
    calls = []

    def run(cmd, cwd=None, env=None, timeout_s=None):
        calls.append(cmd)
        return responder(len(calls), cmd)

    monkeypatch.setattr(dispatch, "run_headless", run)
    return calls


def envelope(result, *, is_error=True, stop="stop_sequence", rc=1):
    """构造 run_headless 的 (rc, stdout, err) 三元组（stdout 为 envelope JSON）。"""
    return (
        rc,
        json.dumps(
            {
                "result": result,
                "is_error": is_error,
                "stop_reason": stop,
                "num_turns": 1,
                "duration_ms": 13000,
                "session_id": "aaaa1111",
            }
        ),
        "",
    )


def cmd_model(cmd):
    """从 argv 提取 ``-m`` 后的模型串（无 -m 时返回 ''，代表走用户级默认）。"""
    return cmd[cmd.index("-m") + 1] if "-m" in cmd else ""
