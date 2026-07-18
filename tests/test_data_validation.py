from mini_tpo.data_loading import load_config, read_raw_data
from mini_tpo.data_validation import validation_summary


def test_validation_summary_has_required_checks():
    cfg = load_config()
    report = validation_summary(read_raw_data(cfg), cfg)
    assert {"check", "column", "affected_rows"}.issubset(report.columns)
    assert (report["check"] == "required_column").any()

