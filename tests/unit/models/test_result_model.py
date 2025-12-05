from datetime import datetime, timedelta

from kosmos.models.result import ExecutionMetadata, StatisticalTestResult


def test_statisticaltestresult_infers_flags_from_pvalue():
    """If significance flags aren't provided, they should be inferred from p_value."""
    st = StatisticalTestResult(
        test_type="t-test",
        test_name="two-sample t-test",
        statistic=2.5,
        p_value=0.012,
        effect_size=0.65,
        effect_size_type="Cohen's d",
        confidence_interval={"lower": 0.2, "upper": 1.1},
        sample_size=100,
        degrees_of_freedom=98,
        is_primary=True,
    )

    # p=0.012 -> significant at 0.05 but not at 0.01
    assert st.significant_0_05 is True
    assert st.significant_0_01 is False
    assert st.significant_0_001 is False
    assert st.significance_label == "*"


def test_executionmetadata_defaults():
    """ExecutionMetadata should default python_version and platform when omitted."""
    now = datetime.utcnow()
    meta = ExecutionMetadata(
        start_time=now,
        end_time=now + timedelta(seconds=1.0),
        duration_seconds=1.0,
        experiment_id="exp-123",
    )

    # Defaults should be present and non-empty
    assert isinstance(meta.python_version, str) and len(meta.python_version) > 0
    assert isinstance(meta.platform, str) and len(meta.platform) > 0
    # protocol_id was made optional; if omitted it should be None
    assert meta.protocol_id is None
