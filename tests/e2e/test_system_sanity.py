"""
Comprehensive system sanity tests for Kosmos AI Scientist.

Tests all major components to validate the complete system works end-to-end.
These are component-level sanity tests designed to touch every major part
of the system and validate basic functionality.
"""

import os

import pytest


@pytest.mark.e2e
class TestComponentSanity:
    """Test each major component works independently."""

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="API key required",
    )
    def test_llm_provider_integration(self):
        """Test LLM provider can generate text (DSPy)."""
        import dspy

        from kosmos.config import get_config

        print("\n🤖 Testing LLM provider integration...")

        config = get_config()
        llm_config = config.llm.to_dspy_config()
        lm = dspy.LM(**llm_config)

        # Simple generation test using DSPy
        with dspy.context(lm=lm):
            response = lm("Say 'hello' in one word")

        assert response is not None
        assert len(response) > 0
        assert "hello" in response.lower()

        print("✅ LLM provider operational (DSPy)")
        print(f"   Response: {response}")

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="API key required",
    )
    def test_hypothesis_generator(self):
        """Test hypothesis generator creates valid hypotheses."""
        from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent

        print("\n💡 Testing hypothesis generator...")

        generator = HypothesisGeneratorAgent(config={"num_hypotheses": 2})

        # Generate hypotheses
        response = generator.generate_hypotheses(
            research_question="How does temperature affect enzyme activity?", domain="biology"
        )

        assert response is not None
        assert hasattr(response, "hypotheses")
        assert len(response.hypotheses) >= 1, "No hypotheses generated"

        # Verify first hypothesis structure
        hyp = response.hypotheses[0]
        assert hyp.statement is not None
        assert hyp.domain == "biology"
        assert hyp.research_question is not None

        print(f"✅ Generated {len(response.hypotheses)} hypothesis(es)")
        print(f"   First: {hyp.statement[:80]}...")

        if hasattr(hyp, "novelty_score"):
            print(f"   Novelty: {hyp.novelty_score}")
        if hasattr(hyp, "testability_score"):
            print(f"   Testability: {hyp.testability_score}")

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="API key required",
    )
    def test_experiment_designer(self):
        """Test experiment designer creates protocols."""
        import uuid

        from kosmos.agents.experiment_designer import ExperimentDesignerAgent
        from kosmos.models.hypothesis import Hypothesis

        print("\n🔬 Testing experiment designer...")

        hypothesis = Hypothesis(
            id=str(uuid.uuid4()),
            research_question="Does temperature affect chemical reaction rates?",
            statement="Higher temperatures increase chemical reaction rates due to increased molecular kinetic energy",
            rationale="According to kinetic molecular theory, higher temperatures result in faster molecular motion and more frequent collisions with sufficient activation energy",
            domain="chemistry",
            testability_score=0.8,
            novelty_score=0.6,
        )

        designer = ExperimentDesignerAgent()
        response = designer.design_experiment(hypothesis)

        assert response is not None, "ExperimentDesigner returned None"
        assert response.protocol is not None, "Response has no protocol"
        assert response.protocol.hypothesis_id == hypothesis.id, "Protocol hypothesis_id mismatch"
        assert response.protocol.experiment_type is not None, "Protocol has no experiment_type"
        assert len(response.protocol.steps) > 0, "Protocol has no steps"

        print("✅ Experiment designer operational")
        print(f"   Protocol: {response.protocol.name}")
        print(f"   Type: {response.protocol.experiment_type.value}")
        print(f"   Steps: {len(response.protocol.steps)}")

    def test_safety_validator(self):
        """Test safety validator blocks dangerous code."""
        from kosmos.safety.code_validator import CodeValidator

        print("\n🛡️  Testing safety validator...")

        validator = CodeValidator()

        # Test safe code
        safe_code = "import numpy as np\nresult = np.mean([1, 2, 3])"
        safe_result = validator.validate(safe_code)
        assert safe_result.passed is True
        print("✅ Safe code allowed")

        # Test dangerous code
        dangerous_code = "import os; os.system('rm -rf /')"
        dangerous_result = validator.validate(dangerous_code)
        assert dangerous_result.passed is False
        assert len(dangerous_result.violations) > 0
        print("✅ Dangerous code blocked")
        print(f"   Violations: {len(dangerous_result.violations)}")

    def test_code_executor(self):
        """Test code executor can run safe code."""
        from kosmos.execution.executor import CodeExecutor

        print("\n▶️  Testing code executor...")

        executor = CodeExecutor(use_sandbox=False)  # Direct execution for speed

        safe_code = """
import numpy as np
result = np.mean([10, 20, 30, 40, 50])
print(f"Mean: {result}")
"""

        exec_result = executor.execute(safe_code)

        assert exec_result.success is True
        assert exec_result.error is None
        assert "Mean: 30" in exec_result.stdout
        assert exec_result.execution_time > 0

        print("✅ Code executed successfully")
        print(f"   Time: {exec_result.execution_time:.3f}s")
        print(f"   Output: {exec_result.stdout.strip()}")

    def test_sandboxed_execution(self):
        """Test Docker sandbox execution."""
        from kosmos.execution.sandbox import DockerSandbox

        print("\n🐳 Testing Docker sandbox execution...")

        # Create sandbox with conservative limits for testing
        sandbox = DockerSandbox(
            cpu_limit=1.0, memory_limit="512m", timeout=60, enable_monitoring=True
        )

        try:
            # Execute simple code
            code = """
import numpy as np
data = [1, 2, 3, 4, 5]
result = np.mean(data)
print(f"Mean: {result}")
print(f"RESULT:{result}")
"""
            result = sandbox.execute(code)

            assert result.success is True, f"Execution failed: {result.error}"
            assert result.exit_code == 0, f"Non-zero exit code: {result.exit_code}"
            assert "Mean: 3.0" in result.stdout, f"Expected output not found: {result.stdout}"
            assert result.timeout_occurred is False, "Unexpected timeout"

            print("✅ Docker sandbox execution successful")
            print(f"   Output: {result.stdout.strip()[:100]}")
            print(f"   Execution time: {result.execution_time:.2f}s")
            if result.resource_stats:
                print(f"   Resources: {result.resource_stats}")
        finally:
            sandbox.cleanup()

    def test_statistical_analysis(self):
        """Test statistical analysis functions."""
        import numpy as np
        import pandas as pd

        from kosmos.execution.data_analysis import DataAnalyzer

        print("\n📊 Testing statistical analysis...")

        # Create test data with known statistical properties
        np.random.seed(42)
        control = np.random.normal(100, 15, 50)
        treatment = np.random.normal(120, 15, 50)  # ~20 point difference

        df = pd.DataFrame(
            {
                "group": ["control"] * 50 + ["treatment"] * 50,
                "score": np.concatenate([control, treatment]),
            }
        )

        # Test t-test
        result = DataAnalyzer.ttest_comparison(df, "group", "score", ("treatment", "control"))

        # Verify result structure
        assert "t_statistic" in result
        assert "p_value" in result
        assert "mean_difference" in result
        assert "significance_label" in result

        # Verify statistical correctness
        assert result["p_value"] < 0.05, "Expected significant difference"
        assert result["significance_label"] in ["*", "**", "***"], "Expected significant result"
        assert abs(result["mean_difference"]) > 15, "Expected meaningful difference"

        print("✅ Statistical analysis operational")
        print(f"   t-statistic: {result['t_statistic']:.3f}")
        print(f"   p-value: {result['p_value']:.6f}")
        print(f"   Mean diff: {result['mean_difference']:.2f}")
        print(f"   Significance: {result['significance_label']}")

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="API key required",
    )
    def test_data_analyst(self):
        """Test data analyst interprets results."""
        import platform as plat
        from datetime import datetime

        from kosmos.agents.data_analyst import DataAnalystAgent
        from kosmos.models.result import (
            ExecutionMetadata,
            ExperimentResult,
            ResultStatus,
            StatisticalTestResult,
        )

        print("\n🔍 Testing data analyst agent...")

        # Create sample experiment result with all required fields
        result = ExperimentResult(
            id="test-result-001",
            experiment_id="exp-001",
            hypothesis_id="hyp-001",
            protocol_id="proto-001",
            status=ResultStatus.SUCCESS,
            primary_test="Two-sample T-test",
            primary_p_value=0.003,
            primary_effect_size=1.2,
            supports_hypothesis=True,
            statistical_tests=[
                StatisticalTestResult(
                    test_type="t-test",
                    test_name="Two-sample T-test",
                    statistic=3.45,
                    p_value=0.003,
                    effect_size=1.2,
                    effect_size_type="Cohen's d",
                    significant_0_05=True,
                    significant_0_01=True,
                    significant_0_001=False,
                    significance_label="**",
                    sample_size=100,
                    is_primary=True,
                )
            ],
            metadata=ExecutionMetadata(
                experiment_id="exp-001",
                protocol_id="proto-001",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_seconds=5.0,
                python_version="3.11",
                platform=plat.system(),
            ),
        )

        agent = DataAnalystAgent()
        interpretation = agent.interpret_results(result=result)

        assert interpretation is not None
        assert hasattr(interpretation, "summary")
        assert len(interpretation.summary) > 0

        print("✅ Data analyst operational")
        print(f"   Hypothesis supported: {interpretation.hypothesis_supported}")
        print(f"   Confidence: {interpretation.confidence}")
        print(f"   Summary: {interpretation.summary[:100]}...")

    def test_database_persistence(self):
        """Test database persistence works."""
        import tempfile
        import uuid

        from kosmos.db import get_session, init_database
        from kosmos.db.models import Hypothesis, HypothesisStatus

        print("\n💾 Testing database persistence...")

        # Use a temporary database for testing
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        init_database(f"sqlite:///{db_path}")

        # Create a hypothesis with explicit ID
        hyp_id = str(uuid.uuid4())
        with get_session() as session:
            hypothesis = Hypothesis(
                id=hyp_id,
                research_question="Does X affect Y?",
                statement="X increases Y through mechanism Z",
                rationale="Prior research suggests a causal relationship between X and Y",
                domain="test_domain",
                status=HypothesisStatus.GENERATED,
                novelty_score=0.75,
                testability_score=0.80,
            )
            session.add(hypothesis)

        # Verify retrieval
        with get_session() as session:
            retrieved = session.query(Hypothesis).filter(Hypothesis.id == hyp_id).first()
            assert retrieved is not None, "Failed to retrieve hypothesis"
            assert retrieved.statement == "X increases Y through mechanism Z"
            assert retrieved.domain == "test_domain"
            assert retrieved.novelty_score == 0.75

        # Clean up
        import os

        os.unlink(db_path)

        print("✅ Database persistence operational")
        print("   Created, stored, and retrieved hypothesis")
        print(f"   ID: {hyp_id[:8]}...")

    @pytest.mark.skip(reason="Neo4j authentication not configured")
    def test_knowledge_graph(self):
        """Test knowledge graph operations."""
        pass


@pytest.mark.e2e
@pytest.mark.slow
class TestEndToEndMiniWorkflow:
    """Test simplified end-to-end workflow without complex agent coordination."""

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="API key required",
    )
    def test_mini_research_workflow(self):
        """Test simplified pipeline: question → hypothesis → execution."""
        from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent
        from kosmos.execution.executor import CodeExecutor

        print("\n🔄 Testing mini end-to-end workflow...")

        research_question = "Is there a correlation between study time and test scores?"

        # Step 1: Generate hypothesis
        print("\n  Step 1: Generate hypothesis...")
        generator = HypothesisGeneratorAgent(config={"num_hypotheses": 1})
        response = generator.generate_hypotheses(research_question, domain="social_science")
        assert len(response.hypotheses) > 0
        hypothesis = response.hypotheses[0]
        print(f"  ✅ Hypothesis: {hypothesis.statement[:60]}...")
        print(f"     Domain: {hypothesis.domain}")
        print(f"     Research question: {hypothesis.research_question[:50]}...")

        # Step 2: Execute simple code (validate executor works)
        print("\n  Step 2: Execute simple analysis code...")
        executor = CodeExecutor(use_sandbox=False)  # Fast direct execution

        # Simple mock analysis
        mock_code = """
import numpy as np
import pandas as pd

# Simulate study time vs test score data
study_hours = [1, 2, 3, 4, 5, 6, 7, 8]
test_scores = [55, 60, 65, 70, 75, 80, 85, 90]

# Calculate correlation
correlation = np.corrcoef(study_hours, test_scores)[0, 1]
print(f"Correlation: {correlation:.3f}")
print(f"Conclusion: {'Strong positive correlation' if correlation > 0.7 else 'Weak correlation'}")
"""

        exec_result = executor.execute(mock_code)
        assert exec_result.success is True
        print("  ✅ Execution successful")
        print(f"     Output: {exec_result.stdout.strip()}")

        # Verify end-to-end flow
        print("\n✅ COMPLETE MINI WORKFLOW VALIDATED")
        print("   Question → Hypothesis → Analysis → Results")
        print("   Core pipeline components integrated successfully!")
