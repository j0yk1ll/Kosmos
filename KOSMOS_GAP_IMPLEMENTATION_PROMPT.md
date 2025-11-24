# Kosmos Gap Implementation Guide

**Target Repository**: https://github.com/jimmc414/kosmos
**Purpose**: Implement solutions for 6 critical gaps identified in OPEN_QUESTIONS.md
**Based On**: OPENQUESTIONS_SOLUTION.md analysis and R&D implementation
**Total Work**: 16 new files + 1 modified file (~6,500 lines of code)

---

## Executive Summary

The Kosmos paper demonstrates impressive autonomous AI scientist capabilities but leaves 6 critical implementation gaps unspecified. This guide provides step-by-step instructions to fill these gaps using patterns from 4 external repositories.

### Gaps to Address

| Gap | Name | Severity | Status | Lines to Implement |
|-----|------|----------|--------|-------------------|
| **Gap 0** | Context Compression Architecture | FOUNDATIONAL | NEW | 403 |
| **Gap 1** | State Manager Architecture | CRITICAL | NEW | 410 |
| **Gap 2** | Task Generation Strategy | CRITICAL | NEW | 1,945 |
| **Gap 3** | Agent Integration & System Prompts | CRITICAL | ENHANCE | 322+ |
| **Gap 4** | Language & Tooling Constraints | HIGH | DESIGN | N/A |
| **Gap 5** | Discovery Evaluation & Filtering | MODERATE | NEW | 407 |

**Total New Code**: ~3,487 lines across 16 new files + 1 modified file

---

## 📚 Required Reading

**IMPORTANT: Before starting implementation, you MUST read these documents:**

### REQUIRED (2 documents):

1. **`OPENQUESTIONS_SOLUTION.md`** - Comprehensive gap analysis (90KB, ~2,800 lines)
   - **WHY** each solution works
   - **EVIDENCE** from R&D implementation
   - **DESIGN TRADE-OFFS** and architectural decisions
   - **PERFORMANCE METRICS** (20:1 compression, 80% approval rates, etc.)
   - **INTEGRATION ARCHITECTURE** showing how all components work together
   - **PATTERN SOURCES** with specific file references from sub-repos
   - **Includes all problem statements** from OPEN_QUESTIONS.md (self-contained)

2. **`KOSMOS_GAP_IMPLEMENTATION_PROMPT.md`** (this document) - Step-by-step implementation guide
   - **HOW** to implement each component
   - **EXACT FILE PATHS** and code templates
   - **GIT COMMANDS** for repository integration
   - **TESTING STRATEGY** and success criteria
   - **IMPLEMENTATION CHECKLIST** with phases

### OPTIONAL (1 document):

3. **`OPEN_QUESTIONS.md`** - Original gap identification (19KB, ~280 lines)
   - Problem statements before solutions
   - Analytical process (gap identification → solution)
   - **Recommended for first-time implementers** (better context)
   - **Can skip if experienced** (solution doc is self-contained)
   - **Useful for debugging** (pure problem statement reference)

### How to Use These Documents Together

**Think of these as:**
- **OPEN_QUESTIONS.md** (optional) = Problem identification
  - Read this FIRST if you're a first-time implementer
  - Shows pure problem statements before solutions
  - 15 minutes, adds valuable context

- **OPENQUESTIONS_SOLUTION.md** (required) = Engineering manual + research paper
  - Self-contained (includes problem statements from OPEN_QUESTIONS.md)
  - Read this to understand WHY each solution works
  - Reference during implementation for architectural context
  - Provides evidence, metrics, and design trade-offs

- **KOSMOS_GAP_IMPLEMENTATION_PROMPT.md** (required) = Assembly instructions + cookbook
  - Follow this step-by-step during implementation
  - Use as your task checklist
  - Provides exact code templates and commands

### Recommended Workflow

**Option A: First-Time Implementer** (includes optional reading):
```
1. READ: OPEN_QUESTIONS.md (15 min)
   ↓ Understand the problems we're solving

2. READ: OPENQUESTIONS_SOLUTION.md (30-45 min, Gaps 0-5 sections)
   ↓ Understand the architecture and why each solution works

3. FOLLOW: KOSMOS_GAP_IMPLEMENTATION_PROMPT.md (Part 1-8)
   ↓ Implement components step-by-step

4. REFERENCE: OPENQUESTIONS_SOLUTION.md when you need:
   - Deeper understanding of a pattern
   - Evidence of what performance to expect
   - Clarification on design decisions

5. VERIFY: Use testing strategy from both documents
   ↓ Ensure implementation matches expected outcomes
```

**Option B: Experienced Developer** (skip optional reading):
```
1. READ: OPENQUESTIONS_SOLUTION.md (30-45 min, Gaps 0-5 sections)
   ↓ Understand architecture (includes problem context)

2. FOLLOW: KOSMOS_GAP_IMPLEMENTATION_PROMPT.md (Part 1-8)
   ↓ Implement components step-by-step

3. REFERENCE: OPENQUESTIONS_SOLUTION.md during implementation
   ↓ Architectural context and design decisions

4. VERIFY: Use testing strategy
   ↓ Ensure implementation matches expected outcomes
```

**Analogy**: You can assemble IKEA furniture with just the instructions (implementation prompt), but having the engineering diagrams (solution doc) helps you understand what you're building, troubleshoot problems, and make informed modifications.

---

## ⚠️ CRITICAL: Implementation Mindset

**ULTRATHINK BEFORE YOU CODE**

This is NOT a simple copy-paste implementation. You are building the **core architecture** of an autonomous AI scientist system with:
- **6,500+ lines** of interconnected code
- **6 critical gaps** that must work together seamlessly
- **Complex dependencies** between components (compression → state → planning → execution → validation)
- **Design trade-offs** that impact system behavior
- **Performance requirements** (20:1 compression, 80% approval rates, 75% validation)

### Before Implementing Each Component:

1. **READ the relevant section** in OPENQUESTIONS_SOLUTION.md
   - Understand WHY this solution works
   - Review the evidence and metrics
   - Identify critical vs optional features

2. **THINK about integration points**
   - How does this component receive input?
   - What does it output and who consumes it?
   - What happens if this component fails?

3. **VERIFY your understanding**
   - Can you explain why 20:1 compression is necessary?
   - Do you understand why ScholarEval needs 8 dimensions?
   - Why is the novelty detector using 75% similarity threshold?

4. **IMPLEMENT with intention**
   - Every line of code has a purpose
   - Design decisions were made for specific reasons
   - Don't skip "optional" features without understanding trade-offs

### Red Flags That You Need to Think Deeper:

- ❌ "I'll just copy this template without understanding it"
- ❌ "This seems complicated, I'll simplify it"
- ❌ "The optional features don't seem important"
- ❌ "I don't need to read the solution doc, the code template is enough"
- ❌ "I'll skip testing until the end"

### Green Flags That You're On Track:

- ✅ "I understand why hierarchical compression is necessary"
- ✅ "I can explain the trade-offs between JSON artifacts and knowledge graphs"
- ✅ "I know why exploration/exploitation ratio changes by cycle"
- ✅ "I'm testing each component before integrating"
- ✅ "I can debug issues because I understand the architecture"

**If you're an AI assistant**: Use extended thinking/reasoning for each major decision. This is complex architectural work that requires careful consideration of design patterns, integration points, and system behavior.

**If you're a human developer**: Take your time. Rushing this implementation will result in a fragile system that doesn't work properly. The 8-10 day timeline assumes deep, thoughtful work - not surface-level coding.

---

## Part 1: Sub-Repository Integration

### Required Repositories

You will reference patterns from these repositories:

| Repository | Purpose | Integration Method | GitHub URL |
|------------|---------|-------------------|------------|
| **kosmos-claude-scientific-skills** | 566 skill files for domain expertise | **GIT SUBTREE** (needs file access) | https://github.com/jimmc414/kosmos-claude-scientific-skills |
| **kosmos-karpathy** | Orchestration patterns (Plan Creator/Reviewer) | REFERENCE ONLY | https://github.com/jimmc414/kosmos-karpathy |
| **kosmos-claude-skills-mcp** | Progressive disclosure, caching patterns | REFERENCE ONLY | https://github.com/jimmc414/kosmos-claude-skills-mcp |
| **kosmos-claude-scientific-writer** | ScholarEval validation framework | REFERENCE ONLY | https://github.com/jimmc414/kosmos-claude-scientific-writer |
| **kosmos-agentic-data-scientist** | Additional patterns (optional) | REFERENCE ONLY | https://github.com/jimmc414/kosmos-agentic-data-scientist |

### Step 1: Integrate Scientific Skills (REQUIRED)

The SkillLoader component needs direct access to 566 skill markdown files. Add as git subtree:

```bash
# From the ORIGINAL kosmos repository root
cd /mnt/c/python/kosmos

# Verify you're in the right place
pwd  # Should show: /mnt/c/python/kosmos

# Add scientific-skills as a subtree (recommended)
git subtree add --prefix kosmos-claude-scientific-skills \
  https://github.com/jimmc414/kosmos-claude-scientific-skills.git \
  main --squash

# Alternative: Add as submodule (if you prefer)
# git submodule add https://github.com/jimmc414/kosmos-claude-scientific-skills.git

# Verify the skills directory exists
ls kosmos-claude-scientific-skills/scientific-skills/
# Should show directories: aeon, alphafold-database, anndata, arboreto, etc. (566 total skills)
```

### Step 2: Reference Other Repositories

Clone these for reference while implementing (NOT as subtrees):

```bash
# Create a reference directory
mkdir -p kosmos-reference
cd kosmos-reference

# Clone reference repos
git clone https://github.com/jimmc414/kosmos-karpathy.git
git clone https://github.com/jimmc414/kosmos-claude-skills-mcp.git
git clone https://github.com/jimmc414/kosmos-claude-scientific-writer.git
git clone https://github.com/jimmc414/kosmos-agentic-data-scientist.git

# Return to kosmos repository root
cd ..
```

---

## Part 2: Implementation Plan by Gap

### Gap 0: Context Compression Architecture

**Problem**: Processing 1,500 papers + 42,000 lines of code exceeds LLM context windows
**Solution Source**: `kosmos-claude-skills-mcp` (progressive disclosure pattern)
**Status**: Fully addresses gap ✅

#### Files to Create

1. **`kosmos/compression/__init__.py`** (~36 lines)
```python
"""
Context Compression Module for Kosmos.

Implements hierarchical compression pipeline achieving 20:1 reduction
(100K+ tokens → 5K tokens) to fit within LLM context windows.

Pattern source: kosmos-claude-skills-mcp (progressive disclosure)
Gap addressed: Gap 0 (Context Compression Architecture)
"""

from .compressor import ContextCompressor, NotebookCompressor, LiteratureCompressor

__all__ = [
    "ContextCompressor",
    "NotebookCompressor",
    "LiteratureCompressor",
]
```

2. **`kosmos/compression/compressor.py`** (~403 lines)

**Key Classes**:
- `ContextCompressor`: Main orchestrator for multi-tier compression
- `NotebookCompressor`: Compresses 42K line notebooks → 2-line summary + stats (300:1)
- `LiteratureCompressor`: Compresses 1,500 papers → structured summaries (25:1)

**Core Methods**:
```python
class ContextCompressor:
    async def compress_cycle(self, cycle_results: List[Dict]) -> Dict:
        """Compress 10 task results into 1 cycle summary."""

    async def compress_notebook(self, notebook_path: str) -> Dict:
        """Compress Jupyter notebook to summary + statistics."""

    async def compress_literature(self, papers: List[Dict]) -> Dict:
        """Compress literature search results."""

class NotebookCompressor:
    def _extract_statistics(self, content: str) -> Dict:
        """Rule-based extraction: p-values, correlations, sample sizes."""

    async def _generate_summary(self, content: str, max_lines: int = 2) -> str:
        """LLM-based 2-line summary generation."""

class LiteratureCompressor:
    async def compress_papers(self, papers: List[Dict]) -> List[Dict]:
        """Convert papers to structured summaries."""
```

**Pattern Reference**: Study `kosmos-reference/kosmos-claude-skills-mcp/` for:
- Multi-tier caching strategy
- Lazy loading patterns
- Progressive disclosure hierarchy

**Compression Targets**:
- Task-level: 150K tokens → 500 tokens (300:1)
- Cycle-level: 10 tasks → 1 summary
- Overall: 100K+ → 5K tokens (20:1)

---

### Gap 1: State Manager Architecture

**Problem**: Paper calls State Manager the "core advancement" but never defines schema, storage, or query interface
**Solution Source**: `kosmos-karpathy` (artifact pattern) + custom knowledge graph design
**Status**: Fully addresses gap ✅

#### Files to Create

1. **`kosmos/world_model/artifacts.py`** (~410 lines)

**Key Class**: `ArtifactStateManager`

**4-Layer Architecture**:

**Layer 1: JSON Artifacts** (always active)
```python
class ArtifactStateManager:
    """
    Hybrid State Manager for Kosmos.

    Architecture:
    - Layer 1: JSON artifacts (human-readable debugging)
    - Layer 2: Knowledge graph (structural queries) [optional]
    - Layer 3: Vector store (semantic search) [optional]
    - Layer 4: Citation tracking (evidence chains)

    Pattern source: kosmos-karpathy (artifact-based communication)
    Gap addressed: Gap 1 (State Manager Architecture)
    """

    def __init__(self, artifacts_dir: str, world_model=None):
        self.artifacts_dir = Path(artifacts_dir)
        self.world_model = world_model  # Optional Neo4j
        self.vector_store = None  # Optional Pinecone/Weaviate

    async def save_finding_artifact(self, cycle: int, task: int, finding: Dict) -> Path:
        """Save finding as JSON artifact."""
        cycle_dir = self.artifacts_dir / f"cycle_{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = cycle_dir / f"task_{task}_finding.json"
        with open(artifact_path, "w") as f:
            json.dump(finding, f, indent=2)

        # Also index to graph if available
        if self.world_model:
            await self._index_finding_to_graph(finding)

        return artifact_path
```

**Layer 2: Knowledge Graph** (optional, for production)
```python
    async def _index_finding_to_graph(self, finding: Dict):
        """Index finding to Neo4j knowledge graph."""
        # Create Finding node
        await self.world_model.create_node(
            "Finding",
            finding_id=finding["id"],
            summary=finding["summary"],
            cycle=finding["cycle"],
            confidence=finding.get("scholar_eval", {}).get("overall_score")
        )

        # Link to Evidence
        await self.world_model.create_relationship(
            finding["id"],
            "DERIVES_FROM",
            finding["notebook_path"],
            {"evidence_type": "data_analysis"}
        )
```

**Core Methods**:
```python
    async def generate_cycle_summary(self, cycle: int) -> str:
        """Generate markdown summary for a cycle."""

    def get_cycle_context(self, cycle: int) -> Dict:
        """Get context for task generation (last 3 cycles)."""

    async def add_finding_with_conflict_check(self, finding: Dict):
        """Add finding, checking for contradictions."""
```

**Pattern Reference**: Study `kosmos-reference/kosmos-karpathy/` for:
- Artifact directory structure (`sandbox/cycle_N/`)
- JSON artifact format
- Human-readable output patterns

**Knowledge Graph Schema**:
```
Entities:
  - Finding (summary, statistics, confidence)
  - Hypothesis (statement, status: supported/refuted/unknown)
  - Evidence (type: notebook/paper, path)
  - Task (id, type, cycle)
  - Citation (paper_id, relevance)

Relationships:
  - SUPPORTS (Finding → Hypothesis, strength: 0-1)
  - REFUTES (Finding → Hypothesis, strength: 0-1)
  - DERIVES_FROM (Finding → Evidence)
  - CITES (Finding → Citation)
  - SUGGESTS (Finding → Hypothesis)
  - CONTRADICTS (Finding → Finding)
```

---

### Gap 2: Task Generation Strategy

**Problem**: How does the system strategically generate 10 prioritized research tasks per cycle? The "brain" is unspecified.
**Solution Source**: `kosmos-karpathy` (Plan Creator + Reviewer orchestration)
**Status**: Fully addresses gap ✅

#### Files to Create

1. **`kosmos/orchestration/__init__.py`** (~53 lines)
```python
"""
Orchestration Module for Kosmos.

Implements strategic task generation with:
- Plan Creator: Generates 10-task research plans
- Plan Reviewer: Multi-dimension quality control
- Delegation Manager: Parallel execution coordinator
- Novelty Detector: Redundancy prevention

Pattern source: kosmos-karpathy (orchestration patterns)
Gap addressed: Gap 2 (Task Generation Strategy)
"""

from .plan_creator import PlanCreatorAgent
from .plan_reviewer import PlanReviewerAgent
from .delegation import DelegationManager
from .novelty_detector import NoveltyDetector

__all__ = [
    "PlanCreatorAgent",
    "PlanReviewerAgent",
    "DelegationManager",
    "NoveltyDetector",
]
```

2. **`kosmos/orchestration/plan_creator.py`** (~329 lines)

**Key Class**: `PlanCreatorAgent`

**Core Responsibility**: Generate 10 strategic research tasks per cycle

**Key Methods**:
```python
class PlanCreatorAgent:
    """
    Plan Creator Agent for strategic research planning.

    Generates 10 prioritized tasks per cycle with exploration/exploitation balance.
    """

    def __init__(self, llm_client, skill_loader):
        self.client = llm_client
        self.skill_loader = skill_loader

    async def create_plan(self, cycle: int, context: Dict) -> Dict:
        """Create 10-task research plan."""

    def _get_exploration_ratio(self, cycle: int) -> float:
        """
        Early cycles: 70% exploration (find new directions)
        Middle: 50% balanced
        Late: 30% exploration (deepen findings)
        """
        if cycle <= 7:
            return 0.70  # Early: explore
        elif cycle <= 14:
            return 0.50  # Middle: balanced
        else:
            return 0.30  # Late: exploit

    def _enforce_task_diversity(self, tasks: List[Dict]) -> bool:
        """Ensure min 3 data_analysis tasks, 2+ different types."""
```

3. **`kosmos/orchestration/plan_reviewer.py`** (~306 lines)

**Key Class**: `PlanReviewerAgent`

**Core Responsibility**: Multi-dimension quality control before execution

**5-Dimension Scoring**:
```python
class PlanReviewerAgent:
    """
    Plan Reviewer Agent for quality control.

    Scores plans on 5 dimensions before approval.
    """

    async def review_plan(self, plan: Dict) -> Dict:
        """
        Review plan on 5 dimensions:
        1. Specificity (0-10): Concrete and executable?
        2. Relevance (0-10): Addresses research objective?
        3. Novelty (0-10): Avoids redundancy?
        4. Coverage (0-10): Covers important aspects?
        5. Feasibility (0-10): Achievable?

        Approval thresholds:
        - Average score ≥ 7.0/10
        - Minimum score ≥ 5.0/10
        """

    def _calculate_approval(self, scores: Dict) -> Dict:
        """Check if plan meets approval thresholds."""
```

4. **`kosmos/orchestration/delegation.py`** (~616 lines)

**Key Class**: `DelegationManager`

**Core Responsibility**: Parallel task execution with routing and retry logic

**Key Methods**:
```python
class DelegationManager:
    """
    Delegation Manager for parallel task execution.

    Features:
    - Parallel execution (max 3 concurrent tasks)
    - Task-type routing (data_analysis → DataAnalystAgent)
    - Retry logic (max 2 attempts per failed task)
    - Result aggregation
    """

    async def execute_plan(self, plan: Dict, cycle: int) -> List[Dict]:
        """Execute all tasks in plan with parallelism."""

    async def _execute_task(self, task: Dict) -> Dict:
        """Execute single task with routing and retry."""

    def _route_task(self, task: Dict):
        """Route task to appropriate agent."""
```

5. **`kosmos/orchestration/novelty_detector.py`** (~494 lines)

**Key Class**: `NoveltyDetector`

**Core Responsibility**: Prevent redundant analyses using vector similarity

**Key Methods**:
```python
class NoveltyDetector:
    """
    Novelty Detector using semantic similarity.

    Uses sentence-transformers to compute task similarity.
    Threshold: 75% similarity = redundant task
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.task_embeddings = []
        self.task_history = []

    def check_task_novelty(self, task: Dict) -> Dict:
        """
        Check if task is novel or redundant.

        Returns:
            {
                "is_novel": bool,
                "novelty_score": float (0-1),
                "similar_tasks": List[Dict]
            }
        """
        # Compute embedding for new task
        task_text = f"{task['type']} {task['description']}"
        task_embedding = self.model.encode(task_text)

        # Compute similarities
        if len(self.task_embeddings) > 0:
            similarities = np.dot(self.task_embeddings, task_embedding)
            max_similarity = max(similarities)
        else:
            max_similarity = 0.0

        is_novel = max_similarity < 0.75  # 75% threshold

        return {
            "is_novel": is_novel,
            "novelty_score": 1.0 - max_similarity,
            "similar_tasks": self._get_top_similar(similarities, 3)
        }
```

6. **`kosmos/orchestration/instructions.yaml`** (~300+ lines)

**System Prompts** for Plan Creator, Plan Reviewer, and agents

```yaml
plan_creator:
  system_prompt: |
    You are a strategic research planning agent for the Kosmos AI Scientist.

    Your role is to generate 10 prioritized research tasks per cycle that:
    - Build on previous findings
    - Balance exploration vs exploitation
    - Cover diverse approaches
    - Are specific and executable

    Current cycle: {cycle}
    Exploration ratio: {exploration_ratio}

    Context from State Manager:
    {context}

    Generate 10 tasks in this format:
    [JSON format specification]

plan_reviewer:
  system_prompt: |
    You are a critical reviewer of research plans.

    Score the following plan on 5 dimensions (0-10 each):
    1. Specificity
    2. Relevance
    3. Novelty
    4. Coverage
    5. Feasibility

    Provide scores and detailed feedback.
```

**Pattern Reference**: Study `kosmos-reference/kosmos-karpathy/` for:
- `karpathy/instructions.yaml` - Orchestration prompt patterns
- Plan Creator/Reviewer interaction flow
- Expert team coordination approach

---

### Gap 3: Agent Integration & System Prompts

**Problem**: How are agents enhanced with domain expertise? System prompts, output formats, error recovery unspecified.
**Solution Source**: `kosmos-claude-scientific-skills` (566 skill files)
**Status**: Fully addresses gap ✅

#### Files to Create

1. **`kosmos/agents/skill_loader.py`** (~322 lines)

**Key Class**: `SkillLoader`

**Core Responsibility**: Auto-load domain expertise based on task type

**Task-Skill Mapping**:
```python
class SkillLoader:
    """
    Scientific Skills Loader for Kosmos Agents.

    Integrates 566 skill markdown files from kosmos-claude-scientific-skills.
    Auto-loads relevant skills based on task requirements.

    Pattern source: kosmos-claude-scientific-skills (566 skill files)
    Gap addressed: Gap 3 (Agent Integration & System Prompts)
    """

    # Task type → Skills mapping
    SKILL_BUNDLES = {
        "single_cell_analysis": ["scanpy", "anndata", "scvi-tools", "cellxgene"],
        "genomics_analysis": ["biopython", "pysam", "gget", "pydeseq2"],
        "cheminformatics": ["rdkit", "datamol", "deepchem"],
        "drug_discovery": ["rdkit", "datamol", "deepchem", "diffdock"],
        "proteomics": ["pyopenms", "matchms", "biopython"],
        "clinical_research": ["clinicaltrials-database", "clinvar-database"],
        "machine_learning": ["pytorch-lightning", "transformers", "scikit-learn"],
        "literature_review": ["pubmed", "openalexdata", "biorxiv-database"],
        "data_visualization": ["matplotlib", "seaborn", "plotly"],
        "statistics": ["scipy", "statsmodels", "pingouin"],
        "pathway_analysis": ["gseapy", "enrichr"],
        "general": ["pandas", "numpy", "scipy"]
    }

    def __init__(self, skills_dir: Optional[Path] = None):
        """
        Initialize SkillLoader.

        Args:
            skills_dir: Path to scientific-skills directory
                        If None, looks for kosmos-claude-scientific-skills sibling
        """
        if skills_dir is None:
            # Look for subtree location
            project_root = Path(__file__).parent.parent.parent
            skills_dir = project_root / "kosmos-claude-scientific-skills" / "scientific-skills"

        self.skills_dir = Path(skills_dir)
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")

    def load_skills_for_task(
        self,
        task_type: Optional[str] = None,
        libraries: Optional[List[str]] = None
    ) -> str:
        """
        Load relevant skills based on task type and required libraries.

        Returns formatted markdown text for prompt injection.
        """
        skills_to_load = []

        # Load by task type
        if task_type and task_type in self.SKILL_BUNDLES:
            skills_to_load.extend(self.SKILL_BUNDLES[task_type])

        # Load specific libraries
        if libraries:
            skills_to_load.extend(libraries)

        # Remove duplicates
        skills_to_load = list(set(skills_to_load))

        # Load skill files
        skill_texts = []
        for skill_name in skills_to_load:
            skill_text = self._load_skill_file(skill_name)
            if skill_text:
                skill_texts.append(skill_text)

        return self._format_skills_for_prompt(skill_texts)
```

#### Files to Modify

1. **`kosmos/agents/data_analyst.py`** (enhance existing file)

**Changes**: Integrate SkillLoader to auto-inject domain expertise

```python
# Add import
from kosmos.agents.skill_loader import SkillLoader

class DataAnalystAgent:
    def __init__(self, llm_client):
        self.client = llm_client
        self.skill_loader = SkillLoader()  # NEW

    async def analyze(self, task: Dict, context: Dict) -> Dict:
        """Enhanced analysis with auto-loaded skills."""

        # Load relevant skills based on task
        skills_text = self._load_relevant_skills(task)  # NEW

        # Build enhanced prompt
        prompt = f"""
{self.base_instruction}

# Scientific Skills Available

{skills_text}

**Research Context**: {context['summary']}

**Task**: {task['description']}

Please analyze and return findings in this JSON format:
{{
    "summary": "2-line summary of key finding",
    "statistics": {{"p_value": 0.001, "effect_size": 2.3}},
    "methods": "Description of methods used",
    "interpretation": "What this means for research",
    "libraries_used": ["scanpy", "anndata"],
    "limitations": "Known limitations"
}}
"""

        response = await self.client.messages.create(...)
        return self._parse_response(response)

    def _load_relevant_skills(self, task: Dict) -> str:  # NEW
        """Load skills based on task type and libraries."""
        task_type = task.get("task_type")
        libraries = task.get("libraries", [])
        return self.skill_loader.load_skills_for_task(task_type, libraries)
```

**Pattern Reference**: Study `kosmos-claude-scientific-skills/scientific-skills/` for:
- Skill file format (SKILL.md structure)
- Common functions and best practices per library
- API documentation patterns

---

### Gap 4: Language & Tooling Constraints

**Problem**: Paper uses R packages but also says "Do all analysis in PYTHON". Which is it?
**Solution**: Python-first design decision with LLM-generated code
**Status**: Partially addresses (acceptable trade-off) ⚠️

#### Implementation Approach

**Design Decision** (No new files needed):
- Python-first code generation
- Library mappings: R → Python equivalents
  - `deseq2` → `pydeseq2`
  - `seurat` → `scanpy`
  - `susieR` → `rpy2 + susieR` (wrapper if needed)
- LLM generates scientifically valid Python code
- Domain skills include Python best practices

**What's NOT Implemented**:
- Actual sandboxed code execution (Docker/Jupyter)
- Multi-language kernel support
- Automatic package installation

**Why This Is Acceptable**:
- Focuses on orchestration architecture (Gap 2 is higher priority)
- LLM code generation sufficient for planning/validation
- Full sandbox execution is future enhancement, not blocker

**Future Work** (if needed):
```python
class SandboxedExecutor:
    """Execute code in isolated Jupyter kernel."""

    async def execute_code(self, code: str) -> Dict:
        # Run in Docker container
        # Install required packages
        # Execute with timeout
        # Return results + stdout/stderr
        pass
```

---

### Gap 5: Discovery Evaluation & Filtering

**Problem**: Paper shows 57.9% interpretation accuracy - significant weakness. How are discoveries validated?
**Solution Source**: `kosmos-claude-scientific-writer` (ScholarEval framework)
**Status**: Fully addresses gap ✅

#### Files to Create

1. **`kosmos/validation/__init__.py`** (~5 lines)
```python
"""
Validation Module for Kosmos.

Implements ScholarEval peer review framework.
"""

from .scholar_eval import ScholarEvalValidator, ScholarEvalScore

__all__ = ["ScholarEvalValidator", "ScholarEvalScore"]
```

2. **`kosmos/validation/scholar_eval.py`** (~407 lines)

**Key Classes**: `ScholarEvalValidator`, `ScholarEvalScore`

**8-Dimension Peer Review**:
```python
from dataclasses import dataclass

@dataclass
class ScholarEvalScore:
    """
    ScholarEval scoring result.

    8 dimensions rated 0-1:
    1. Novelty: How new/original is the finding?
    2. Rigor: Are methods scientifically sound?
    3. Clarity: Is finding clearly stated?
    4. Reproducibility: Can others replicate?
    5. Impact: Scientific/practical significance?
    6. Coherence: Fits existing knowledge?
    7. Limitations: Are limitations acknowledged?
    8. Ethics: Ethical considerations?
    """
    novelty: float
    rigor: float
    clarity: float
    reproducibility: float
    impact: float
    coherence: float
    limitations: float
    ethics: float
    overall_score: float
    passes_threshold: bool
    feedback: str

class ScholarEvalValidator:
    """
    ScholarEval Validation Framework for Kosmos.

    Provides peer-review style validation of scientific discoveries
    before inclusion in State Manager.

    Pattern source: kosmos-claude-scientific-writer (ScholarEval)
    Gap addressed: Gap 5 (Discovery Evaluation & Filtering)
    """

    # Weighted scoring formula
    WEIGHTS = {
        "rigor": 0.25,           # 25% - most important
        "impact": 0.20,          # 20%
        "novelty": 0.15,         # 15%
        "reproducibility": 0.15, # 15%
        "clarity": 0.10,         # 10%
        "coherence": 0.10,       # 10%
        "limitations": 0.03,     # 3%
        "ethics": 0.02           # 2%
    }

    # Approval thresholds
    OVERALL_THRESHOLD = 0.75  # 75%
    RIGOR_THRESHOLD = 0.70    # 70% minimum for rigor

    async def evaluate_finding(self, finding: Dict) -> ScholarEvalScore:
        """
        Evaluate finding on 8 dimensions.

        Returns ScholarEvalScore with weighted overall score.
        """

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(finding)

        # Query LLM for scoring
        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            temperature=0.3,  # Consistent evaluation
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse scores
        scores = self._parse_scores(response.content[0].text)

        # Calculate weighted overall score
        overall = self._calculate_overall_score(scores)

        # Check approval thresholds
        passes = (
            overall >= self.OVERALL_THRESHOLD
            and scores.get("rigor", 0) >= self.RIGOR_THRESHOLD
        )

        return ScholarEvalScore(
            novelty=scores["novelty"],
            rigor=scores["rigor"],
            clarity=scores["clarity"],
            reproducibility=scores["reproducibility"],
            impact=scores["impact"],
            coherence=scores["coherence"],
            limitations=scores["limitations"],
            ethics=scores["ethics"],
            overall_score=overall,
            passes_threshold=passes,
            feedback=scores.get("feedback", "")
        )
```

**Pattern Reference**: Study `kosmos-reference/kosmos-claude-scientific-writer/` for:
- ScholarEval scoring rubrics
- Peer review dimension definitions
- Quality threshold calibration

---

### Integration: Research Workflow

**Purpose**: Tie all components together into 20-cycle autonomous research loop

#### Files to Create

1. **`kosmos/workflow/__init__.py`** (~25 lines)
```python
"""
Research Workflow Module for Kosmos.

Implements end-to-end autonomous research loop.
"""

from .research_loop import ResearchWorkflow

__all__ = ["ResearchWorkflow"]
```

2. **`kosmos/workflow/research_loop.py`** (~687 lines)

**Key Class**: `ResearchWorkflow`

**Complete 20-Cycle Loop**:
```python
class ResearchWorkflow:
    """
    End-to-End Research Workflow for Kosmos.

    Integrates all components:
    - ContextCompressor (Gap 0)
    - ArtifactStateManager (Gap 1)
    - PlanCreator, PlanReviewer, DelegationManager, NoveltyDetector (Gap 2)
    - SkillLoader (Gap 3)
    - ScholarEvalValidator (Gap 5)
    """

    def __init__(self, config: Dict):
        # Initialize all components
        self.state_manager = ArtifactStateManager(config["artifacts_dir"])
        self.compressor = ContextCompressor()
        self.plan_creator = PlanCreatorAgent(llm_client, skill_loader)
        self.plan_reviewer = PlanReviewerAgent(llm_client)
        self.delegation_manager = DelegationManager(agents)
        self.novelty_detector = NoveltyDetector()
        self.scholar_eval = ScholarEvalValidator(llm_client)

    async def run(self, research_objective: str, num_cycles: int = 20):
        """
        Run autonomous research for N cycles.

        Each cycle:
        1. BUILD CONTEXT (State Manager) → ~5K tokens
        2. PLAN CREATION (Plan Creator) → 10 tasks
        3. NOVELTY CHECK (Novelty Detector) → Flag redundant
        4. PLAN REVIEW (Plan Reviewer) → 5-dimension scoring
        5. TASK EXECUTION (Delegation Manager) → Parallel
        6. DISCOVERY VALIDATION (ScholarEval) → 8-dimension
        7. STATE UPDATE (State Manager) → JSON + graph
        8. CONTEXT COMPRESSION (Compressor) → 20:1
        """

        for cycle in range(1, num_cycles + 1):
            logger.info(f"=== Cycle {cycle}/{num_cycles} ===")

            # 1. Get context from State Manager
            context = self.state_manager.get_cycle_context(cycle)
            compressed_context = await self.compressor.compress_cycle(context)

            # 2. Create plan
            plan = await self.plan_creator.create_plan(cycle, compressed_context)

            # 3. Check novelty
            for task in plan["tasks"]:
                novelty = self.novelty_detector.check_task_novelty(task)
                task["novelty_score"] = novelty["novelty_score"]

            # 4. Review plan
            review = await self.plan_reviewer.review_plan(plan)

            if not review["approved"]:
                logger.warning(f"Plan rejected: {review['feedback']}")
                # Revise plan based on feedback
                plan = await self.plan_creator.create_plan(
                    cycle, compressed_context, revision_feedback=review["feedback"]
                )
                review = await self.plan_reviewer.review_plan(plan)

            # 5. Execute tasks
            results = await self.delegation_manager.execute_plan(plan, cycle)

            # 6. Validate findings
            validated_findings = []
            for task_result in results:
                finding = self._extract_finding(task_result)

                # ScholarEval validation
                eval_score = await self.scholar_eval.evaluate_finding(finding)

                if eval_score.passes_threshold:
                    finding["scholar_eval"] = eval_score.__dict__
                    validated_findings.append(finding)

                    # 7. Store in State Manager
                    await self.state_manager.save_finding_artifact(
                        cycle, task_result["task_id"], finding
                    )
                else:
                    logger.warning(
                        f"Finding rejected: score={eval_score.overall_score:.2f}"
                    )

            # 8. Update novelty detector
            for task in plan["tasks"]:
                self.novelty_detector.add_task(task)

            # Generate cycle summary
            summary = await self.state_manager.generate_cycle_summary(cycle)
            logger.info(f"Cycle {cycle} complete: {len(validated_findings)} validated findings")
```

---

### Enhancement: Report Generation

**Purpose**: Generate publication-quality research reports

#### Files to Create

1. **`kosmos/reporting/__init__.py`** (~35 lines)
```python
"""
Reporting Module for Kosmos.

Generates publication-quality research reports from validated findings.
"""

from .report_synthesizer import ReportSynthesizer

__all__ = ["ReportSynthesizer"]
```

2. **`kosmos/reporting/report_synthesizer.py`** (~536 lines)

**Key Class**: `ReportSynthesizer`

**Pattern Source**: `kosmos-claude-scientific-writer`

```python
class ReportSynthesizer:
    """
    Report Synthesis using scientific-writer patterns.

    Generates publication-quality research reports with:
    - Professional formatting
    - Citation management
    - Figure integration
    - Multiple output formats

    Pattern source: kosmos-claude-scientific-writer
    Enhancement for final report generation
    """

    async def generate_research_report(
        self,
        findings: List[Dict],
        research_objective: str
    ) -> Dict:
        """
        Generate complete research report from validated findings.

        Returns:
            {
                "title": str,
                "abstract": str,
                "introduction": str,
                "methods": str,
                "results": str,
                "discussion": str,
                "conclusions": str,
                "references": List[str],
                "figures": List[Dict]
            }
        """
```

---

## Part 3: Implementation Checklist

### Phase 1: Foundation (Gaps 0, 1, 5, 3)

- [ ] **Gap 0: Context Compression**
  - [ ] Create `kosmos/compression/__init__.py` (~36 lines)
  - [ ] Create `kosmos/compression/compressor.py` (~403 lines)
  - [ ] Test: Compress sample notebook (42K → 500 tokens)

- [ ] **Gap 1: State Manager**
  - [ ] Create `kosmos/world_model/artifacts.py` (~410 lines)
  - [ ] Test: Save/retrieve JSON artifacts
  - [ ] Test: Generate cycle summary

- [ ] **Gap 5: Discovery Validation**
  - [ ] Create `kosmos/validation/__init__.py` (~5 lines)
  - [ ] Create `kosmos/validation/scholar_eval.py` (~407 lines)
  - [ ] Test: Validate sample finding (8 dimensions)

- [ ] **Gap 3: Agent Integration**
  - [ ] Add scientific-skills as subtree: `git subtree add --prefix kosmos-claude-scientific-skills https://github.com/jimmc414/kosmos-claude-scientific-skills.git main --squash`
  - [ ] Create `kosmos/agents/skill_loader.py` (~322 lines)
  - [ ] Modify `kosmos/agents/data_analyst.py` (+~100 lines)
  - [ ] Test: Load skills for "single_cell_analysis"

### Phase 2: Orchestration (Gap 2)

- [ ] **Gap 2: Task Generation Strategy**
  - [ ] Create `kosmos/orchestration/__init__.py` (~53 lines)
  - [ ] Create `kosmos/orchestration/plan_creator.py` (~329 lines)
  - [ ] Create `kosmos/orchestration/plan_reviewer.py` (~306 lines)
  - [ ] Create `kosmos/orchestration/delegation.py` (~616 lines)
  - [ ] Create `kosmos/orchestration/novelty_detector.py` (~494 lines)
  - [ ] Create `kosmos/orchestration/instructions.yaml` (~300+ lines)
  - [ ] Test: Create plan for cycle 1
  - [ ] Test: Review plan (5 dimensions)
  - [ ] Test: Check novelty (vector similarity)

### Phase 3: Integration

- [ ] **Research Workflow**
  - [ ] Create `kosmos/workflow/__init__.py` (~25 lines)
  - [ ] Create `kosmos/workflow/research_loop.py` (~687 lines)
  - [ ] Test: Run 5-cycle workflow

- [ ] **Report Generation** (Enhancement)
  - [ ] Create `kosmos/reporting/__init__.py` (~35 lines)
  - [ ] Create `kosmos/reporting/report_synthesizer.py` (~536 lines)
  - [ ] Test: Generate report from findings

### Phase 4: Testing & Documentation

- [ ] **Integration Testing**
  - [ ] Test end-to-end: 5-cycle research workflow
  - [ ] Verify all components integrate correctly
  - [ ] Measure compression ratios (target: 20:1)
  - [ ] Measure validation rates (expect: ~75%)

- [ ] **Documentation**
  - [ ] Update README with new architecture
  - [ ] Document gap solutions
  - [ ] Add usage examples

---

## Part 4: Testing Strategy

### Component Tests

```python
# Test Context Compression
from kosmos.compression import ContextCompressor

compressor = ContextCompressor()
compressed = await compressor.compress_notebook("sample_notebook.ipynb")
assert len(compressed["summary"]) <= 200  # 2-line summary
assert "statistics" in compressed
print(f"Compression ratio: {original_tokens / compressed_tokens}")

# Test State Manager
from kosmos.world_model import ArtifactStateManager

state_mgr = ArtifactStateManager("./artifacts")
finding = {
    "summary": "Identified 245 DEGs with p<0.001",
    "statistics": {"p_value": 0.0001, "n_genes": 245}
}
await state_mgr.save_finding_artifact(cycle=1, task=1, finding=finding)

# Test ScholarEval
from kosmos.validation import ScholarEvalValidator

validator = ScholarEvalValidator(llm_client)
score = await validator.evaluate_finding(finding)
assert score.overall_score >= 0 and score.overall_score <= 1
print(f"Overall: {score.overall_score:.2f}, Rigor: {score.rigor:.2f}")

# Test Skill Loader
from kosmos.agents import SkillLoader

loader = SkillLoader()
skills = loader.load_skills_for_task("single_cell_analysis", ["scanpy"])
assert "scanpy" in skills
assert "sc.pp.filter_cells" in skills  # Should have function examples

# Test Novelty Detector
from kosmos.orchestration import NoveltyDetector

detector = NoveltyDetector()
detector.add_task({"description": "Analyze gene expression in cancer"})
novelty = detector.check_task_novelty({"description": "Examine gene expression in tumors"})
assert novelty["novelty_score"] < 1.0  # Should be similar
```

### Integration Test

```python
# Test Full 5-Cycle Workflow
from kosmos.workflow import ResearchWorkflow

config = {
    "artifacts_dir": "./test_artifacts",
    "llm_provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022"
}

workflow = ResearchWorkflow(config)
results = await workflow.run(
    research_objective="Identify novel cancer biomarkers",
    num_cycles=5
)

# Verify results
assert results["cycles_completed"] == 5
assert results["validated_findings"] > 0
assert results["plan_approval_rate"] > 0.5
```

---

## Part 5: Dependencies

### Python Packages to Add

Add to `pyproject.toml` or `requirements.txt`:

```toml
# Context Compression (Gap 0)
sentence-transformers = ">=2.2.0"  # For novelty detection

# Orchestration (Gap 2)
pyyaml = ">=6.0"  # For instructions.yaml
numpy = ">=1.24.0"  # For vector operations

# Scientific Skills (Gap 3)
# (Skills reference these, but don't need to be installed)
# scanpy, biopython, rdkit, etc. - optional based on domain

# State Manager (Gap 1) - Optional
neo4j = ">=5.0.0"  # For knowledge graph (optional)
pinecone-client = ">=2.0.0"  # For vector store (optional)

# LLM Integration
anthropic = ">=0.18.0"  # For Claude API
```

---

## Part 6: Success Criteria

### Gap 0: Context Compression ✅
- [ ] Achieves 20:1 compression ratio (100K+ → 5K tokens)
- [ ] Maintains critical information (findings, statistics)
- [ ] Lazy loading functional for full content

### Gap 1: State Manager ✅
- [ ] JSON artifacts human-readable
- [ ] Cycle summaries generated correctly
- [ ] Context retrieval works for task generation
- [ ] Optional: Graph queries functional

### Gap 2: Task Generation ✅
- [ ] Plans approved ~80% on first submission
- [ ] Tasks complete successfully ~90%
- [ ] Novelty detector prevents redundancy (>75% similar)
- [ ] Exploration/exploitation ratio adapts by cycle

### Gap 3: Agent Integration ✅
- [ ] 120+ skills loadable by domain
- [ ] Skills auto-injected based on task type
- [ ] Consistent JSON output format
- [ ] Error recovery functional

### Gap 5: Discovery Validation ✅
- [ ] ~75% validation rate (expected for quality)
- [ ] 8-dimension scoring consistent
- [ ] Approval thresholds enforced (≥0.75 overall, ≥0.70 rigor)
- [ ] Actionable feedback provided

### Integration ✅
- [ ] 5-cycle workflow completes successfully
- [ ] All components integrate without errors
- [ ] Final report generation functional

---

## Part 7: Troubleshooting

### Issue: Skills directory not found

```bash
# Verify subtree was added correctly
ls kosmos-claude-scientific-skills/scientific-skills/
# Should show: aeon, alphafold-database, anndata, etc.

# If missing, re-add:
git subtree add --prefix kosmos-claude-scientific-skills \
  https://github.com/jimmc414/kosmos-claude-scientific-skills.git \
  main --squash
```

### Issue: Import errors

```python
# Verify __init__.py files are present
ls kosmos/compression/__init__.py
ls kosmos/orchestration/__init__.py
ls kosmos/validation/__init__.py
ls kosmos/workflow/__init__.py
ls kosmos/reporting/__init__.py
```

### Issue: LLM API errors

Check environment variables:
```bash
export ANTHROPIC_API_KEY="your-api-key"
export DEEPSEEK_API_KEY="your-api-key"  # If using DeepSeek
```

### Issue: Missing dependencies

```bash
pip install sentence-transformers pyyaml numpy anthropic
```

---

## Part 8: Next Steps After Implementation

Once all gaps are implemented:

1. **Run Integration Tests**
   ```bash
   pytest tests/e2e/test_research_workflow.py
   ```

2. **Run 5-Cycle Demo**
   ```bash
   python examples/04_end_to_end/research_workflow_demo.py
   ```

3. **Verify Metrics**
   - Compression ratio: ~20:1
   - Plan approval: ~80%
   - Task success: ~90%
   - Validation rate: ~75%

4. **Generate First Report**
   ```python
   from kosmos.reporting import ReportSynthesizer
   synthesizer = ReportSynthesizer()
   report = await synthesizer.generate_research_report(findings, objective)
   ```

5. **Scale to 20 Cycles**
   ```python
   workflow.run(objective, num_cycles=20)
   ```

---

## Summary

This guide provides complete instructions to implement solutions for all 6 gaps identified in the Kosmos paper:

- **Gap 0**: Context compression (403 lines)
- **Gap 1**: State Manager architecture (410 lines)
- **Gap 2**: Task generation strategy (1,945 lines)
- **Gap 3**: Agent integration (322+ lines)
- **Gap 4**: Language constraints (design decision)
- **Gap 5**: Discovery validation (407 lines)

**Total**: 16 new files + 1 modified file (~3,487 lines) across 3 implementation phases.

**Repository Integration**:
- **Required**: kosmos-claude-scientific-skills (git subtree for 566 skill files)
- **Reference**: kosmos-karpathy, kosmos-claude-skills-mcp, kosmos-claude-scientific-writer

**Timeline**:
- Phase 1 (Foundation): 2-3 days
- Phase 2 (Orchestration): 2-3 days
- Phase 3 (Integration): 1-2 days
- Testing & Documentation: 1-2 days

**Result**: Fully autonomous AI scientist system capable of 20-cycle research with validated discoveries and publication-quality reports.

---

## 📦 Implementation Package

**To successfully implement the Kosmos gaps, provide these documents:**

```
📁 Kosmos Gap Implementation Package/
├── KOSMOS_GAP_IMPLEMENTATION_PROMPT.md  ← REQUIRED: Step-by-step HOW
├── OPENQUESTIONS_SOLUTION.md             ← REQUIRED: Deep WHY + evidence
└── OPEN_QUESTIONS.md                     ← OPTIONAL: Problem identification
```

**What Each Document Provides:**

1. **KOSMOS_GAP_IMPLEMENTATION_PROMPT.md** (REQUIRED)
   - WHAT to build and HOW to build it
   - Exact code templates and commands
   - Step-by-step instructions

2. **OPENQUESTIONS_SOLUTION.md** (REQUIRED)
   - WHY each solution works
   - Architectural context and design trade-offs
   - Evidence and performance metrics
   - **Self-contained** (includes problem statements)

3. **OPEN_QUESTIONS.md** (OPTIONAL)
   - Pure problem identification
   - Shows analytical process (problem → solution)
   - **Recommended for first-time implementers**
   - **Can skip if experienced** (solution doc has problem context)

**Minimum Required**: Documents #1 and #2 are sufficient for implementation.

**Recommended for First-Timers**: All 3 documents for complete context.
