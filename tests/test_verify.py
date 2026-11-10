from tblab.verify import format_table, run_checks


def test_all_builtin_checks_pass():
    rows = run_checks()
    assert len(rows) >= 15
    failed = [r["check"] for r in rows if not r["passed"]]
    assert not failed, failed
    table = format_table(rows)
    assert f"{len(rows)}/{len(rows)} checks passed" in table
