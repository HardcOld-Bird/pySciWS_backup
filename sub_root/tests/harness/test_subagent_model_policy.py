"""Harness contract: subagent dynamic model routing plugin is wired correctly.

Covers the dsh-subagent-model-policy port (plan: sub_root/others/plans/subagent-model-policy.md):
the plugin bundle is mounted on the web profile, the authorization row keeps the
three allowed routes, tool-subagent patches use the DSH 0.2.2 schema keys, and
the plugin's own node test suite stays green.
"""

from __future__ import annotations

import re
import subprocess

from pysci.paths import PROJECT_ROOT
from pysci.toolchain import node_exe

MODULES = PROJECT_ROOT / "deps_management" / "deps" / "node" / "modules"
PLUGIN = MODULES / "dsh_plugins" / "subagent-model-policy"
PROFILE = MODULES / "dsh_home" / "profiles" / "web"
PROFILE_PACKAGE = PROFILE / "package.json"
PROFILE_PATCH = PROFILE / "cordis.patch.yml"
POLICY_YAML = PROJECT_ROOT / "subagent-model-policy.yaml"

ALLOWED_ROUTES = (
    ("xiaomi-token-plan-cn", "mimo-v2.6-flash"),
    ("xiaomi-token-plan-cn", "mimo-v2.6-pro"),
    ("deepseek-account", "deepseek-flash"),
)


def _read(path) -> str:
    return path.read_text(encoding="utf-8")


def test_plugin_package_complete():
    for name in (
        "package.json",
        "index.js",
        "config.js",
        "policy.js",
        "calendar.js",
        "cordis.patch.yml",
        "data/chinese-days.json",
        "test/policy.test.mjs",
        "test/guard.test.mjs",
        "test/config.test.mjs",
    ):
        assert (PLUGIN / name).is_file(), f"missing {name}"


def test_bundle_wired_in_profile():
    text = _read(PROFILE_PACKAGE)
    assert "dsh-subagent-model-policy" in text
    assert "link:" in text
    # The bundle must be listed for the loader to compose its cordis patch.
    bundles = text.split('"bundles"', 1)[1]
    assert '"dsh-subagent-model-policy"' in bundles


def test_authorization_row_holds_three_routes():
    text = _read(PROFILE_PATCH)
    assert "id: subagent-model-selection-settings" in text
    assert "enabled: true" in text
    for provider, model in ALLOWED_ROUTES:
        assert f"provider: {provider}" in text
        assert f"model: {model}" in text


def test_tool_subagent_patches_use_new_schema_keys():
    text = _read(PROFILE_PATCH)
    # DSH 0.2.2 renamed modelSelectionSettings -> modelSelection / modelDiscovery;
    # the old key is not in the schema and must never reappear.
    assert "modelSelectionSettings" not in text
    assert "modelSelection: true" in text
    assert "modelDiscovery: true" in text


def test_plugin_patch_only_inserts_policy_row():
    text = _read(PLUGIN / "cordis.patch.yml")
    assert "id: subagent-model-policy" in text
    # Authorization stays a single source of truth on the profile layer.
    # Comment lines may name the row to document the contract; only YAML
    # content counts.
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "subagent-model-selection-settings" not in body


def test_policy_yaml_defaults():
    text = _read(POLICY_YAML)
    plan = re.search(r"^plan:\s*(\S+)", text, flags=re.MULTILINE)
    assert plan is not None
    assert plan.group(1) in {"mimo", "qwen", "deepseek"}
    assert "peak: [[9, 12], [14, 18]]" in text
    for provider, model in ALLOWED_ROUTES:
        assert model in text
        assert provider in text


def test_guard_scope_route_pair_and_discovery_exemption():
    # v1.1 (2026-11-16): route selection requires a full provider+model pair —
    # a bare `provider` parameter with unrelated semantics (e.g.
    # cordis_inspect_query's Inspect Provider ID) is not a route. Read-only
    # discovery tools are exempt by stable upstream name. Behavior details
    # live in the plugin node suite; these identifiers lock the contract.
    text = _read(PLUGIN / "index.js")
    assert "declaresRoutePair" in text
    assert "READ_ONLY_DISCOVERY_TOOLS" in text
    assert "'list_subagent_models'" in text


def test_denial_text_contract_v12():
    # v1.2 (2026-11-18, user decision): the implicit-inheritance denial stops
    # echoing the parent route — the main session usually runs the pro tier
    # while subagents default to flash, so echoing it read as a recommendation
    # and anchored models onto the expensive tier. The fork denial keeps the
    # parent route because there it is the rejected inheritance target.
    # Allowed listings always put flash-tier routes before pro/max-tier ones
    # (`flashFirst`), and every route denial carries the tier-selection
    # teaching line (`TIER_GUIDANCE`). Identifiers lock the contract; wording
    # behavior lives in the plugin node suite.
    index = _read(PLUGIN / "index.js")
    assert "TIER_GUIDANCE" in index
    for phrase in ("优先使用 flash", "拿不准，就选 flash", "事后必须向用户显式声明"):
        assert phrase in index
    needs = index.split("function needsExplicitRoute", 1)[1].split("function ", 1)[0]
    assert "父会话" not in needs
    fork = index.split("function forkNotAllowed", 1)[1]
    assert "只能继承父路由" in fork
    assert "flashFirst" in index
    assert "flashFirst" in _read(PLUGIN / "policy.js")


def test_plugin_node_suite_green():
    # Run each test file directly: `node --test` spawns child processes that
    # confinement blocks (piped stdio EPERM), a plain node run does not.
    for name in ("policy.test.mjs", "guard.test.mjs", "config.test.mjs"):
        result = subprocess.run(
            [node_exe(), str(PLUGIN / "test" / name)],
            cwd=PLUGIN,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr
