"""
Kosmos Execution Module.

Provides sandboxed code execution capabilities for running generated
scientific code safely with resource limits and security isolation.

Components:
- DockerManager: Container lifecycle management with pooling
- JupyterClient: Code execution with output capture
- PackageResolver: Automatic dependency detection and installation
- ProductionExecutor: Main executor combining all components

Basic Usage:
    from kosmos.execution import ProductionExecutor, ProductionConfig

    async def run():
        executor = ProductionExecutor()
        await executor.initialize()

        result = await executor.execute_code('''
            import pandas as pd
            df = pd.DataFrame({'x': [1, 2, 3]})
            results = {'sum': df['x'].sum()}
        ''')

        print(result.return_value)  # {'sum': 6}
        await executor.cleanup()

Legacy Usage (existing sandbox):
    from kosmos.execution import DockerSandbox, CodeExecutor

    # Direct sandbox usage
    sandbox = DockerSandbox()
    result = sandbox.execute("print('Hello!')")

    # Code executor with optional sandbox
    executor = CodeExecutor(use_sandbox=True)
    result = executor.execute("x = 1 + 1")
"""

# Production executor components (new)
from .docker_manager import ContainerConfig, ContainerInstance, ContainerStatus, DockerManager
from .executor import (
    CodeExecutor,
    CodeValidator,
    RetryStrategy,
    execute_protocol_code,
)
from .jupyter_client import CellOutput, ExecutionResult, ExecutionStatus, JupyterClient
from .package_resolver import (
    IMPORT_TO_PIP,
    STDLIB_MODULES,
    PackageRequirement,
    PackageResolver,
    extract_imports_from_code,
    is_stdlib_module,
    resolve_package_name,
)
from .production_executor import ProductionConfig, ProductionExecutor, execute_code_safely

# Legacy components (existing)
from .sandbox import DockerSandbox, SandboxExecutionResult, execute_in_sandbox


# Re-export commonly used items at package level
__all__ = [
    # Production executor (recommended)
    "ProductionExecutor",
    "ProductionConfig",
    "execute_code_safely",
    # Docker management
    "DockerManager",
    "ContainerConfig",
    "ContainerInstance",
    "ContainerStatus",
    # Jupyter client
    "JupyterClient",
    "ExecutionResult",
    "ExecutionStatus",
    "CellOutput",
    # Package resolution
    "PackageResolver",
    "PackageRequirement",
    "extract_imports_from_code",
    "resolve_package_name",
    "is_stdlib_module",
    "IMPORT_TO_PIP",
    "STDLIB_MODULES",
    # Legacy (existing)
    "DockerSandbox",
    "SandboxExecutionResult",
    "execute_in_sandbox",
    "CodeExecutor",
    "CodeValidator",
    "RetryStrategy",
    "execute_protocol_code",
]
