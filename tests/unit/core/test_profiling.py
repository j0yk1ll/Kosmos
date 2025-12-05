"""
Unit tests for profiling system.

Tests ExecutionProfiler, profile context manager, and performance tracking.
"""

import time

import pytest

from kosmos.core.profiling import (
    ExecutionProfiler,
    ProfilingMode,
    format_profile_summary,
    profile_experiment_execution,
)


@pytest.mark.unit
class TestProfilingMode:
    """Test ProfilingMode enum."""

    def test_profiling_modes_exist(self):
        """Test that all profiling modes are defined."""
        assert ProfilingMode.LIGHT == "light"
        assert ProfilingMode.STANDARD == "standard"
        assert ProfilingMode.FULL == "full"


@pytest.mark.unit
class TestExecutionProfiler:
    """Test ExecutionProfiler class."""

    def test_initialization_light_mode(self):
        """Test profiler initialization with light mode."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)
        assert profiler.mode == ProfilingMode.LIGHT

    def test_initialization_standard_mode(self):
        """Test profiler initialization with standard mode."""
        profiler = ExecutionProfiler(mode=ProfilingMode.STANDARD)
        assert profiler.mode == ProfilingMode.STANDARD

    def test_initialization_full_mode(self):
        """Test profiler initialization with full mode."""
        profiler = ExecutionProfiler(mode=ProfilingMode.FULL)
        assert profiler.mode == ProfilingMode.FULL

    def test_initialization_with_threshold(self):
        """Test profiler initialization with custom threshold."""
        profiler = ExecutionProfiler(mode=ProfilingMode.STANDARD, bottleneck_threshold_percent=5.0)
        assert profiler.bottleneck_threshold == 5.0

    def test_context_manager_light_mode(self):
        """Test profiler as context manager."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)

        with profiler.profile_context():
            time.sleep(0.05)

        result = profiler.get_result()
        assert result is not None
        assert result.execution_time >= 0.05

    def test_context_manager_standard_mode(self):
        """Test profiler context manager with standard mode."""
        profiler = ExecutionProfiler(mode=ProfilingMode.STANDARD)

        with profiler.profile_context():
            time.sleep(0.05)

        result = profiler.get_result()
        assert result is not None
        assert result.execution_time >= 0.05
        assert result.profiling_mode == ProfilingMode.STANDARD

    def test_profile_function(self):
        """Test profiling a function."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)

        def test_func(x, y):
            time.sleep(0.05)
            return x + y

        result, profile = profiler.profile_function(test_func, 2, 3)

        assert result == 5
        assert profile is not None
        assert profile.execution_time >= 0.05

    def test_profile_function_with_kwargs(self):
        """Test profiling a function with keyword arguments."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)

        def test_func(x, factor=1):
            return x * factor

        result, profile = profiler.profile_function(test_func, 5, factor=3)

        assert result == 15
        assert profile is not None

    def test_get_result_before_profiling(self):
        """Test get_result returns None before profiling."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)
        assert profiler.get_result() is None

    def test_reset(self):
        """Test reset clears profiler state."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)

        with profiler.profile_context():
            time.sleep(0.01)

        assert profiler.get_result() is not None

        profiler.reset()
        assert profiler.get_result() is None


@pytest.mark.unit
class TestProfileResult:
    """Test ProfileResult data model."""

    def test_profile_result_fields(self):
        """Test ProfileResult has expected fields."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)

        with profiler.profile_context():
            time.sleep(0.01)

        result = profiler.get_result()

        assert hasattr(result, "execution_time")
        assert hasattr(result, "cpu_time")
        assert hasattr(result, "wall_time")
        assert hasattr(result, "memory_peak_mb")
        assert hasattr(result, "profiling_mode")

    def test_profile_result_timing_consistency(self):
        """Test that wall_time == execution_time."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)

        with profiler.profile_context():
            time.sleep(0.05)

        result = profiler.get_result()

        # execution_time and wall_time should be the same
        assert result.execution_time == result.wall_time


@pytest.mark.unit
class TestProfileExperiment:
    """Test profiling experiment code."""

    def test_profile_experiment(self):
        """Test profiling experiment code execution."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)

        code = "result = sum(range(1000))"

        profile = profiler.profile_experiment(experiment_id="test_001", code=code)

        assert profile is not None
        assert profile.execution_time >= 0

    def test_profile_experiment_with_local_vars(self):
        """Test profiling with local variables."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)

        code = "result = value * 2"
        local_vars = {"value": 42}

        profile = profiler.profile_experiment(
            experiment_id="test_002", code=code, local_vars=local_vars
        )

        assert profile is not None

    def test_profile_experiment_error_handling(self):
        """Test profiling handles code errors."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)

        code = "result = undefined_variable"

        with pytest.raises(NameError):
            profiler.profile_experiment(experiment_id="test_003", code=code)


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_profile_experiment_execution(self):
        """Test profile_experiment_execution function."""
        result = profile_experiment_execution(
            experiment_id="test_123",
            code="result = [i**2 for i in range(100)]",
            mode=ProfilingMode.LIGHT,
        )

        assert result is not None
        assert result.execution_time >= 0

    def test_format_profile_summary(self):
        """Test format_profile_summary function."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)

        with profiler.profile_context():
            time.sleep(0.01)

        result = profiler.get_result()
        summary = format_profile_summary(result)

        assert "PROFILE SUMMARY" in summary
        assert "Execution time" in summary
        assert "CPU time" in summary
        assert "Memory peak" in summary


@pytest.mark.unit
class TestMemoryTracking:
    """Test memory tracking functionality."""

    def test_memory_tracking_standard_mode(self):
        """Test memory tracking in standard mode."""
        profiler = ExecutionProfiler(mode=ProfilingMode.STANDARD, enable_memory_tracking=True)

        with profiler.profile_context():
            # Allocate some memory
            data = [0] * 100000
            _ = len(data)

        result = profiler.get_result()

        # Memory tracking should capture something
        assert result.memory_peak_mb >= 0

    def test_memory_tracking_disabled(self):
        """Test memory tracking can be disabled."""
        profiler = ExecutionProfiler(mode=ProfilingMode.STANDARD, enable_memory_tracking=False)

        with profiler.profile_context():
            data = [0] * 10000
            _ = len(data)

        result = profiler.get_result()
        # Should still complete successfully
        assert result is not None


@pytest.mark.unit
class TestBottleneckDetection:
    """Test bottleneck detection."""

    def test_bottleneck_detection_standard_mode(self):
        """Test bottleneck detection in standard mode."""
        profiler = ExecutionProfiler(
            mode=ProfilingMode.STANDARD, bottleneck_threshold_percent=1.0  # Low threshold for test
        )

        with profiler.profile_context():
            # Do some work to generate statistics
            total = 0
            for i in range(10000):
                total += i
            _ = total

        result = profiler.get_result()

        # Should have analyzed function calls
        assert result.profiling_mode == ProfilingMode.STANDARD


@pytest.mark.unit
class TestProfilerOverhead:
    """Test profiler overhead."""

    def test_light_mode_minimal_overhead(self):
        """Test light mode has minimal overhead."""
        profiler = ExecutionProfiler(mode=ProfilingMode.LIGHT)

        def simple_func():
            return sum(range(1000))

        # Time without profiling
        start = time.time()
        for _ in range(100):
            simple_func()
        base_time = time.time() - start

        # Time with profiling
        start = time.time()
        for _ in range(100):
            with profiler.profile_context():
                simple_func()
        profiled_time = time.time() - start

        # Profiled time should not be drastically more than base time
        # Allow 5x overhead as a generous bound
        assert profiled_time < base_time * 5
