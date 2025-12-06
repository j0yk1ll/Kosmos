"""
Integration tests for complete analysis pipeline (Phase 6).

Tests end-to-end flow: ExperimentResult → Analysis → Visualization → Summary.
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from kosmos.agents.data_analyst import DataAnalystAgent, ResultInterpretation
from kosmos.analysis.plotly_viz import PlotlyVisualizer
from kosmos.analysis.statistics import DescriptiveStats, StatisticalReporter
from kosmos.analysis.summarizer import ResultSummarizer, ResultSummary
from kosmos.analysis.visualization import PublicationVisualizer
from kosmos.models.hypothesis import Hypothesis
from kosmos.models.result import (
    ExecutionMetadata,
    ExperimentResult,
    ResultStatus,
    StatisticalTestResult,
    VariableResult,
)


# Fixtures


@pytest.fixture
def sample_experiment_result():
    """Create sample experiment result."""
    return ExperimentResult(
        id="result-001",
        experiment_id="exp-001",
        hypothesis_id="hyp-001",
        protocol_id="proto-001",
        status=ResultStatus.SUCCESS,
        primary_test="Two-sample T-test",
        primary_p_value=0.012,
        primary_effect_size=0.65,
        primary_ci_lower=0.2,
        primary_ci_upper=1.1,
        supports_hypothesis=True,
        statistical_tests=[
            StatisticalTestResult(
                test_type="t-test",
                test_name="Two-sample T-test",
                statistic=2.54,
                p_value=0.012,
                effect_size=0.65,
                effect_size_type="Cohen's d",
                confidence_interval={"lower": 0.2, "upper": 1.1},
                sample_size=100,
                degrees_of_freedom=98,
                significance_label="*",
                is_primary=True,
                significant_0_05=True,  # p=0.012 < 0.05
                significant_0_01=False,  # p=0.012 > 0.01
                significant_0_001=False,  # p=0.012 > 0.001
            )
        ],
        variable_results=[
            VariableResult(
                variable_name="treatment",
                variable_type="independent",
                mean=10.5,
                median=10.3,
                std=2.1,
                min=6.2,
                max=15.8,
                q1=9.1,
                q3=11.9,
                n_samples=50,
                n_missing=0,
            ),
            VariableResult(
                variable_name="control",
                variable_type="independent",
                mean=8.8,
                median=8.5,
                std=2.3,
                min=4.5,
                max=13.2,
                q1=7.2,
                q3=10.1,
                n_samples=50,
                n_missing=0,
            ),
        ],
        metadata=ExecutionMetadata(
            experiment_id="exp-001",
            protocol_id="proto-001",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=5.3,
            random_seed=42,
            python_version="3.11",
            platform="linux",
        ),
        raw_data={"mean_diff": 1.7},
        generated_files=[],
        version=1,
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_hypothesis():
    """Create sample hypothesis."""
    return Hypothesis(
        id="hyp-001",
        research_question="Does treatment X increase outcome Y compared to control?",
        statement="Treatment X increases outcome Y compared to control",
        rationale="Prior studies suggest mechanism via pathway Z operates through documented biological pathways",
        domain="biology",
        testability_score=0.9,
        novelty_score=0.7,
        variables=["treatment", "control", "outcome_Y"],
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# End-to-End Pipeline Tests


class TestCompleteAnalysisPipeline:
    """Tests for complete analysis pipeline."""

    def test_full_pipeline_result_to_interpretation(
        self, sample_experiment_result, sample_hypothesis
    ):
        """Test Result → DataAnalystAgent → Interpretation."""
        # Mock DSPy response
        mock_response = Mock()
        mock_response.hypothesis_supported = "true"
        mock_response.confidence = "0.85"
        mock_response.summary = "Test summary"
        mock_response.key_findings = '["Finding 1"]'
        mock_response.significance_interpretation = "Significant"
        mock_response.biological_significance = "Meaningful"
        mock_response.comparison_to_prior_work = "Similar"
        mock_response.potential_confounds = '["Confound 1"]'
        mock_response.follow_up_experiments = '["Experiment 1"]'
        mock_response.overall_assessment = "Good"

        # Create agent with DSPy mocking
        with patch("kosmos.agents.data_analyst.dspy.LM") as mock_lm:
            mock_llm = Mock()
            mock_lm.return_value = mock_llm

            # Pass llm_config so self.llm is not None
            agent = DataAnalystAgent(llm_config={"model": "test", "api_key": "test"})

            with patch("kosmos.agents.data_analyst.dspy.context"):
                with patch("kosmos.agents.data_analyst.dspy.Predict") as mock_predict:
                    mock_predictor = Mock()
                    mock_predictor.return_value = mock_response
                    mock_predict.return_value = mock_predictor

                    # Interpret results
                    interpretation = agent.interpret_results(
                        result=sample_experiment_result, hypothesis=sample_hypothesis
                    )

        assert isinstance(interpretation, ResultInterpretation)
        assert interpretation.experiment_id == "exp-001"
        assert interpretation.hypothesis_supported is True
        assert len(interpretation.key_findings) > 0

    def test_full_pipeline_result_to_visualization(self, sample_experiment_result, temp_output_dir):
        """Test Result → PublicationVisualizer → Plots."""
        viz = PublicationVisualizer()

        # Auto-select plots
        suggested_plots = viz.select_plot_types(sample_experiment_result)

        assert len(suggested_plots) > 0
        assert any(p["type"] == "box_plot_with_points" for p in suggested_plots)

        # Generate a plot
        np.random.seed(42)
        data = {
            "treatment": np.random.normal(10.5, 2.1, 50),
            "control": np.random.normal(8.8, 2.3, 50),
        }

        output_path = os.path.join(temp_output_dir, "test_plot.png")
        viz.box_plot_with_points(
            data=data, title="Treatment vs Control", y_label="Outcome", output_path=output_path
        )

        assert os.path.exists(output_path)

    def test_full_pipeline_result_to_summary(self, sample_experiment_result, sample_hypothesis):
        """Test Result → ResultSummarizer → Summary."""
        # Mock DSPy response
        mock_response = Mock()
        mock_response.summary = "The experiment tested whether Treatment X increases outcome Y compared to control. Results show a statistically significant increase with medium-large effect size."
        mock_response.key_findings = """1. Treatment group significantly higher than control (p=0.012)
2. Effect size Cohen's d=0.65 (medium-large)
3. Results support the hypothesis"""
        mock_response.hypothesis_assessment = (
            "Results support the hypothesis that Treatment X increases outcome Y."
        )
        mock_response.limitations = """- Sample size moderate (n=100)
- Single-center study"""
        mock_response.future_work = """1. Replicate in larger cohort
2. Investigate mechanism
3. Test dose-response"""

        # Create summarizer with DSPy mocking
        with patch("kosmos.analysis.summarizer.dspy.LM") as mock_lm:
            mock_llm = Mock()
            mock_lm.return_value = mock_llm

            # Pass llm_config
            summarizer = ResultSummarizer(llm_config={"model": "test", "api_key": "test"})

            with patch("kosmos.analysis.summarizer.dspy.context"):
                with patch("kosmos.analysis.summarizer.dspy.Predict") as mock_predict:
                    mock_predictor = Mock()
                    mock_predictor.return_value = mock_response
                    mock_predict.return_value = mock_predictor

                    # Generate summary
                    summary = summarizer.generate_summary(
                        result=sample_experiment_result, hypothesis=sample_hypothesis
                    )

        assert isinstance(summary, ResultSummary)
        assert summary.experiment_id == "exp-001"
        assert len(summary.summary) > 0
        assert len(summary.key_findings) > 0
        assert len(summary.future_work) > 0

    def test_complete_pipeline_integration(
        self, sample_experiment_result, sample_hypothesis, temp_output_dir
    ):
        """Test complete pipeline: Result → All Analysis Components."""
        # 1. Interpret results with DSPy mocking
        mock_analysis_response = Mock()
        mock_analysis_response.hypothesis_supported = "true"
        mock_analysis_response.confidence = "0.85"
        mock_analysis_response.summary = "Test"
        mock_analysis_response.key_findings = '["F1"]'
        mock_analysis_response.significance_interpretation = "Sig"
        mock_analysis_response.biological_significance = "Bio"
        mock_analysis_response.comparison_to_prior_work = "Comp"
        mock_analysis_response.potential_confounds = '["C1"]'
        mock_analysis_response.follow_up_experiments = '["E1"]'
        mock_analysis_response.overall_assessment = "Good"

        with patch("kosmos.agents.data_analyst.dspy.LM") as mock_lm:
            mock_llm = Mock()
            mock_lm.return_value = mock_llm

            # Pass llm_config so self.llm is not None
            agent = DataAnalystAgent(llm_config={"model": "test", "api_key": "test"})

            with patch("kosmos.agents.data_analyst.dspy.context"):
                with patch("kosmos.agents.data_analyst.dspy.Predict") as mock_predict:
                    mock_predictor = Mock()
                    mock_predictor.return_value = mock_analysis_response
                    mock_predict.return_value = mock_predictor

                    interpretation = agent.interpret_results(
                        result=sample_experiment_result, hypothesis=sample_hypothesis
                    )

        # 2. Generate visualizations
        viz = PublicationVisualizer()
        np.random.seed(42)
        data = {
            "treatment": np.random.normal(10.5, 2.1, 50),
            "control": np.random.normal(8.8, 2.3, 50),
        }

        plot_path = os.path.join(temp_output_dir, "analysis_plot.png")
        viz.box_plot_with_points(data=data, output_path=plot_path)

        # 3. Generate summary
        mock_summary_response = Mock()
        mock_summary_response.summary = "Test summary"
        mock_summary_response.key_findings = "1. Finding 1"
        mock_summary_response.hypothesis_assessment = "Supported"
        mock_summary_response.limitations = "- Limit 1"
        mock_summary_response.future_work = "1. Work 1"

        with patch("kosmos.analysis.summarizer.dspy.LM") as mock_lm2:
            mock_llm2 = Mock()
            mock_lm2.return_value = mock_llm2

            # Pass llm_config
            summarizer = ResultSummarizer(llm_config={"model": "test", "api_key": "test"})

            with patch("kosmos.analysis.summarizer.dspy.context"):
                with patch("kosmos.analysis.summarizer.dspy.Predict") as mock_predict:
                    mock_predictor = Mock()
                    mock_predictor.return_value = mock_summary_response
                    mock_predict.return_value = mock_predictor

                    summary = summarizer.generate_summary(
                        result=sample_experiment_result, hypothesis=sample_hypothesis
                    )

        # Verify all components completed successfully
        assert interpretation is not None
        assert os.path.exists(plot_path)
        assert summary is not None


# Statistical Analysis Tests


class TestStatisticalAnalysis:
    """Tests for statistical analysis components."""

    def test_descriptive_statistics(self):
        """Test descriptive statistics computation."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)

        stats = DescriptiveStats.compute_full_descriptive(data)

        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "skewness" in stats
        assert stats["n"] == 100
        assert 9 < stats["mean"] < 11  # Should be close to 10

    def test_statistical_reporter(self):
        """Test comprehensive statistical report generation."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "var1": np.random.normal(10, 2, 50),
                "var2": np.random.normal(15, 3, 50),
                "var3": np.random.normal(20, 4, 50),
            }
        )

        reporter = StatisticalReporter()
        report = reporter.generate_full_report(
            df, include_correlations=True, include_distributions=True
        )

        assert len(report) > 0
        assert "Descriptive Statistics" in report
        assert "Distribution Analysis" in report or "Correlation Analysis" in report


# Visualization Format Tests


class TestVisualizationFormats:
    """Tests for visualization output formats."""

    def test_matplotlib_and_plotly_compatibility(self, temp_output_dir):
        """Test both matplotlib and plotly visualizers work."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + np.random.randn(50)

        # Matplotlib version
        pub_viz = PublicationVisualizer()
        pub_path = os.path.join(temp_output_dir, "matplotlib.png")
        pub_viz.scatter_with_regression(x, y, "X", "Y", "Matplotlib", pub_path)

        assert os.path.exists(pub_path)

        # Plotly version
        try:
            plotly_viz = PlotlyVisualizer()
            fig = plotly_viz.interactive_scatter(x, y, "X", "Y", "Plotly")

            html_path = os.path.join(temp_output_dir, "plotly.html")
            plotly_viz.save_html(fig, html_path)

            assert os.path.exists(html_path)
        except ImportError:
            pytest.skip("Plotly not installed")


# Anomaly and Pattern Detection Tests


class TestDetectionPipeline:
    """Tests for anomaly and pattern detection."""

    def test_anomaly_detection_in_pipeline(self):
        """Test anomaly detection on problematic results."""
        with patch("kosmos.agents.data_analyst.dspy.LM") as mock_lm:
            mock_llm = Mock()
            mock_lm.return_value = mock_llm

            agent = DataAnalystAgent(llm_config={"model": "test", "api_key": "test"})

        # Create result with anomaly (significant p-value, tiny effect)
        anomalous_result = ExperimentResult(
            id="result-anom",
            experiment_id="exp-anom",
            hypothesis_id="hyp-anom",
            protocol_id="proto-anom",
            status=ResultStatus.SUCCESS,
            primary_test="T-test",
            primary_p_value=0.001,  # Very significant
            primary_effect_size=0.05,  # Tiny effect
            supports_hypothesis=True,
            statistical_tests=[
                StatisticalTestResult(
                    test_type="T-test",
                    test_name="T-test",
                    statistic=3.5,
                    p_value=0.001,
                    effect_size=0.05,
                    effect_size_type="Cohen's d",
                    confidence_interval={"lower": 0.01, "upper": 0.09},
                    sample_size=100,
                    degrees_of_freedom=98,
                    significance_label="***",
                    is_primary=True,
                    significant_0_05=True,
                    significant_0_01=True,
                    significant_0_001=True,
                )
            ],
            variable_results=[],
            metadata=ExecutionMetadata(
                experiment_id="exp-anom",
                protocol_id="proto-anom",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_seconds=1.0,
                random_seed=42,
                python_version="3.11",
                platform="linux",
            ),
            created_at=datetime.utcnow(),
        )

        anomalies = agent.detect_anomalies(anomalous_result)

        assert len(anomalies) > 0
        assert any("tiny effect size" in a.lower() for a in anomalies)

    def test_pattern_detection_across_results(self):
        """Test pattern detection across multiple results."""
        with patch("kosmos.agents.data_analyst.dspy.LM") as mock_lm:
            mock_llm = Mock()
            mock_lm.return_value = mock_llm

            agent = DataAnalystAgent(llm_config={"model": "test", "api_key": "test"})

        # Create results with consistent positive effects
        results = [
            ExperimentResult(
                id=f"result-{i}",
                experiment_id=f"exp-{i}",
                hypothesis_id="hyp-001",
                protocol_id="proto-001",
                status=ResultStatus.SUCCESS,
                primary_test="T-test",
                primary_p_value=0.01,
                primary_effect_size=0.5 + i * 0.1,  # Positive, increasing
                supports_hypothesis=True,
                statistical_tests=[
                    StatisticalTestResult(
                        test_type="T-test",
                        test_name="T-test",
                        statistic=2.5 + i * 0.2,
                        p_value=0.01,
                        effect_size=0.5 + i * 0.1,
                        effect_size_type="Cohen's d",
                        confidence_interval={"lower": 0.2 + i * 0.1, "upper": 0.8 + i * 0.1},
                        sample_size=100,
                        degrees_of_freedom=98,
                        significance_label="**",
                        is_primary=True,
                        significant_0_05=True,
                        significant_0_01=True,
                        significant_0_001=False,
                    )
                ],
                variable_results=[],
                metadata=ExecutionMetadata(
                    experiment_id=f"exp-{i}",
                    protocol_id="proto-001",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    duration_seconds=1.0,
                    random_seed=42,
                    python_version="3.11",
                    platform="linux",
                ),
                created_at=datetime.utcnow(),
            )
            for i in range(5)
        ]

        patterns = agent.detect_patterns_across_results(results)

        assert len(patterns) > 0
        # Should detect consistent positive effects or increasing trend
        assert any("positive" in p.lower() or "increasing" in p.lower() for p in patterns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
