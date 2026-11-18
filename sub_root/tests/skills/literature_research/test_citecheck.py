"""citecheck 固定样例断言：引用键抽取与 PASS/WARN/FAIL/NOT_FOUND 分级合同。"""

from __future__ import annotations

from pysci.skills.literature_research import citecheck

BODY = (
    "As shown \\citep[p.~3]{doe2020} and \\cite{roe2021,doe2020}, "
    "the claim holds [@smith2022]. Also see @smith2022 and mail a@b.com."
)
BIB = "@article{doe2020,\n  title={A}\n}\n@inproceedings{roe2021,\n  title={B}\n}\n@misc{smith2022,\n  title={C}\n}\n@misc{orphan1999,\n  title={D}\n}"


def test_extract_citation_keys_mixed_notations_dedup_ordered():
    keys = citecheck.extract_citation_keys(BODY)
    assert keys == ("doe2020", "roe2021", "smith2022")


def test_extract_citation_keys_ignores_email_and_tail_punct():
    keys = citecheck.extract_citation_keys("contact a@b.com. Then cite @key2020.")
    assert keys == ("key2020",)


def test_extract_bib_keys():
    assert citecheck.extract_bib_keys(BIB) == ("doe2020", "roe2021", "smith2022", "orphan1999")


def test_grade_pass_when_aligned():
    report = citecheck.reconcile(("a", "b"), ("b", "a"))
    assert report.grade == "PASS"
    assert not report.missing and not report.orphans


def test_grade_warn_on_orphan_only():
    report = citecheck.reconcile(("a",), ("a", "extra"))
    assert report.grade == "WARN"
    assert report.orphans == ("extra",)


def test_grade_fail_on_missing_wins_over_warn():
    report = citecheck.reconcile(("a", "gone"), ("a", "extra"))
    assert report.grade == "FAIL"
    assert report.missing == ("gone",)
    assert report.orphans == ("extra",)


def test_grade_not_found_when_both_empty():
    assert citecheck.reconcile((), ()).grade == "NOT_FOUND"


def test_reconcile_note_and_render_audit_marks():
    # BIB 多出 orphan1999（正文未引用）→ WARN；删掉后全对齐 → PASS
    report = citecheck.reconcile_note(BODY, BIB)
    assert report.grade == "WARN"
    report_ok = citecheck.reconcile_note(BODY, BIB.replace("@misc{orphan1999,\n  title={D}\n}", ""))
    assert report_ok.grade == "PASS"
    rendered = report_ok.render()
    assert rendered.splitlines()[0].startswith("[✓ PASS]")
