# PROMPT 2: Implement Kosmos Gaps

**⚠️ CRITICAL: Run this in `/mnt/c/python/kosmos` (the original kosmos repository)**
**NOT in `/mnt/c/python/kosmos-research` - that's just the R&D directory**
**Repository**: https://github.com/jimmc414/kosmos
**Purpose**: Implement 6 critical gaps to create autonomous AI scientist system
**Prerequisites**: PROMPT 1 completed + 3 MD files copied to repo root

---

## Prerequisites Checklist

Before proceeding, verify ALL of these are complete:

### ✅ Step 1: Repository Setup (PROMPT 1)
- [ ] Ran PROMPT_1_SETUP_REPOSITORIES.md
- [ ] `kosmos-claude-scientific-skills/` exists with 566 skills
- [ ] `kosmos-reference/` contains 4 reference repositories

### ✅ Step 2: Implementation Files Copied
- [ ] `KOSMOS_GAP_IMPLEMENTATION_PROMPT.md` is in repo root (REQUIRED)
- [ ] `OPENQUESTIONS_SOLUTION.md` is in repo root (REQUIRED)
- [ ] `OPEN_QUESTIONS.md` is in repo root (OPTIONAL - see below)

---

## Instructions for AI Assistant

You are implementing the 6 critical gaps identified in the Kosmos paper. This is **complex architectural work** requiring deep understanding and careful implementation.

### 🚨 MANDATORY: Verify Prerequisites First

Run this verification BEFORE implementing anything:

```bash
echo "========================================="
echo "Implementation Prerequisites Check"
echo "========================================="
echo ""

# Check we're in the CORRECT kosmos repo
current_dir=$(pwd)
if [ "$current_dir" != "/mnt/c/python/kosmos" ]; then
    echo "❌ ERROR: Wrong directory!"
    echo "   Current: $current_dir"
    echo "   Expected: /mnt/c/python/kosmos"
    echo "   This must run in the ORIGINAL kosmos repo, not kosmos-research"
    exit 1
fi

if [ ! -d "kosmos" ]; then
    echo "❌ ERROR: Not in kosmos repository root"
    echo "   Expected to see kosmos/ directory (Python package)"
    exit 1
fi
echo "✅ In correct kosmos repository root: /mnt/c/python/kosmos"

# Check scientific skills (REQUIRED)
echo ""
echo "Checking required repositories..."
if [ -d "kosmos-claude-scientific-skills/scientific-skills" ]; then
    skill_count=$(ls kosmos-claude-scientific-skills/scientific-skills/ | wc -l)
    echo "✅ kosmos-claude-scientific-skills: $skill_count skills"
    if [ $skill_count -lt 500 ]; then
        echo "⚠️  WARNING: Expected 560+ skills, found $skill_count"
    fi
else
    echo "❌ CRITICAL ERROR: kosmos-claude-scientific-skills NOT FOUND"
    echo "   This is REQUIRED for Gap 3 (SkillLoader)"
    echo "   Run PROMPT_1_SETUP_REPOSITORIES.md first"
    exit 1
fi

# Check reference repos
if [ ! -d "kosmos-reference" ]; then
    echo "⚠️  WARNING: kosmos-reference/ not found"
    echo "   Reference patterns will be harder to access"
    echo "   Recommended: Run PROMPT_1_SETUP_REPOSITORIES.md"
else
    echo "✅ Reference repositories available in kosmos-reference/"
fi

# Check implementation files
echo ""
echo "Checking implementation files..."
files_present=0

if [ -f "KOSMOS_GAP_IMPLEMENTATION_PROMPT.md" ]; then
    echo "✅ KOSMOS_GAP_IMPLEMENTATION_PROMPT.md present"
    ((files_present++))
else
    echo "❌ ERROR: KOSMOS_GAP_IMPLEMENTATION_PROMPT.md NOT FOUND"
fi

if [ -f "OPENQUESTIONS_SOLUTION.md" ]; then
    echo "✅ OPENQUESTIONS_SOLUTION.md present"
    ((files_present++))
else
    echo "❌ ERROR: OPENQUESTIONS_SOLUTION.md NOT FOUND"
fi

if [ -f "OPEN_QUESTIONS.md" ]; then
    echo "✅ OPEN_QUESTIONS.md present (optional - adds context)"
    ((files_present++))
else
    echo "ℹ️  OPEN_QUESTIONS.md not found (optional - not required)"
fi

echo ""
if [ $files_present -lt 2 ]; then
    echo "❌ CRITICAL ERROR: Missing REQUIRED implementation files"
    echo "   Required (2 files):"
    echo "     1. KOSMOS_GAP_IMPLEMENTATION_PROMPT.md"
    echo "     2. OPENQUESTIONS_SOLUTION.md"
    echo "   Optional (1 file):"
    echo "     3. OPEN_QUESTIONS.md (recommended for first-timers)"
    echo ""
    echo "   Action: Copy required files to repo root before proceeding"
    exit 1
fi

echo "========================================="
echo "✅ All prerequisites met - ready to implement"
echo "========================================="
```

**If any checks fail, STOP and resolve the issues before proceeding.**

---

## 📄 About OPEN_QUESTIONS.md (Optional File)

**Question**: Should I read OPEN_QUESTIONS.md?

**Short Answer**: **Optional** - OPENQUESTIONS_SOLUTION.md is self-contained.

**Detailed Explanation**:

### OPEN_QUESTIONS.md Contains:
- Identification of the 6 gaps (~280 lines, 15 min read)
- Problem statements: "What's missing from the paper?"
- Questions: "How is this done?"
- Analysis of why each gap blocks reproduction

### OPENQUESTIONS_SOLUTION.md Already Includes:
- **All problem statements** from OPEN_QUESTIONS.md (quoted in each gap section)
- **All missing information** listings
- **PLUS the solutions**, evidence, and implementation details

### When to Read OPEN_QUESTIONS.md:

**✅ READ IT if you are:**
- First-time implementer (better context)
- Want to understand the analytical process (problem → solution)
- Prefer seeing pure problem statements before solutions
- Have 15 extra minutes

**⏭️ SKIP IT if you are:**
- Experienced developer (solution doc is self-contained)
- Time-constrained (jump straight to solutions)
- Coming back for reference (solution doc has everything)

**🔍 REFERENCE IT later if:**
- Debugging and need pure problem statement
- Validating your solution addresses original requirement
- Writing documentation about gap identification

### Recommendation by Experience Level:

| Experience | OPEN_QUESTIONS.md | Reason |
|------------|-------------------|--------|
| **First implementation** | ✅ Read it first (15 min) | Better context, shows analytical process |
| **Experienced developer** | ⏭️ Optional | Solution doc is self-contained |
| **Debugging/Validation** | 🔍 Reference | Pure problem statements useful |

**Bottom line**: OPENQUESTIONS_SOLUTION.md includes all the problem context you need. OPEN_QUESTIONS.md adds value for first-timers but isn't required.

---

## 📚 MANDATORY: Read Documentation First

**DO NOT start implementing until you have READ AND UNDERSTOOD the documentation.**

### Step 1: Read Documentation (30-60 minutes total)

**REQUIRED**: Read OPENQUESTIONS_SOLUTION.md (30-45 minutes)
**OPTIONAL**: Read OPEN_QUESTIONS.md first (15 minutes) - see section above for guidance

This is the **architectural foundation**. You MUST understand WHY each solution works before implementing.

**Reading Order** (if reading both):
1. **OPEN_QUESTIONS.md** (15 min) - Problem identification
2. **OPENQUESTIONS_SOLUTION.md** (30-45 min) - Solutions and evidence

OR skip directly to:
1. **OPENQUESTIONS_SOLUTION.md** (30-45 min) - Self-contained with problem context

**Focus on these sections:**
1. **Executive Summary** - Overview of all 6 gaps
2. **Gap 0: Context Compression** - WHY 20:1 compression is necessary
3. **Gap 1: State Manager** - WHY hybrid architecture (JSON + graph)
4. **Gap 2: Task Generation** - WHY Plan Creator + Reviewer pattern
5. **Gap 3: Agent Integration** - WHY 566 skills are auto-loaded
6. **Gap 4: Language/Tooling** - WHY Python-first design decision
7. **Gap 5: Discovery Validation** - WHY 8-dimension scoring

**What you should understand after reading:**
- ✅ Why hierarchical compression enables 100K+ → 5K token reduction
- ✅ Why State Manager uses 4-layer architecture
- ✅ Why exploration/exploitation ratio changes by cycle (70% → 50% → 30%)
- ✅ Why novelty detector uses 75% similarity threshold
- ✅ Why ScholarEval needs 8 dimensions (not fewer)
- ✅ How all components integrate in the 20-cycle research loop

**If you cannot explain these WHY questions, read the document again.**

### Step 2: Review KOSMOS_GAP_IMPLEMENTATION_PROMPT.md (15-20 minutes)

This is your **step-by-step implementation guide**.

**Focus on these sections:**
- **Part 2: Implementation Plan by Gap** - Exact files to create
- **Part 3: Implementation Checklist** - Phase-by-phase approach
- **Part 4: Testing Strategy** - How to verify each component
- **Part 6: Success Criteria** - Performance targets to hit

---

## ⚠️ CRITICAL: ULTRATHINK Before You Code

**This is NOT a simple copy-paste implementation.**

You are building the **core architecture** of an autonomous AI scientist system with:
- **6,500+ lines** of interconnected code
- **6 critical gaps** that must work together seamlessly
- **Complex dependencies**: compression → state → planning → execution → validation
- **Design trade-offs** that impact system behavior
- **Performance requirements**: 20:1 compression, 80% approval, 75% validation

### Before Implementing EACH Component:

1. **READ** the relevant section in OPENQUESTIONS_SOLUTION.md
   - Understand WHY this solution works
   - Review evidence and metrics
   - Identify critical vs optional features

2. **THINK** about integration points
   - How does this component receive input?
   - What does it output and who consumes it?
   - What happens if this component fails?

3. **VERIFY** your understanding
   - Can you explain the design decisions?
   - Do you know what performance to expect?
   - Can you justify the trade-offs made?

4. **IMPLEMENT** with intention
   - Every line of code has a purpose
   - Don't skip "optional" features without understanding trade-offs
   - Test as you go (don't wait until the end)

### 🚩 Red Flags - STOP and Think Deeper:

- ❌ "I'll just copy this template without understanding it"
- ❌ "This seems complicated, I'll simplify it"
- ❌ "The optional features don't seem important"
- ❌ "I don't need to read the solution doc"
- ❌ "I'll skip testing until the end"

### ✅ Green Flags - You're On Track:

- ✅ "I understand why hierarchical compression is necessary"
- ✅ "I can explain the JSON artifacts vs knowledge graph trade-offs"
- ✅ "I know why exploration/exploitation changes by cycle"
- ✅ "I'm testing each component before integrating"
- ✅ "I can debug because I understand the architecture"

---

## 📋 Implementation Plan

Follow **KOSMOS_GAP_IMPLEMENTATION_PROMPT.md** for detailed instructions. Here's the high-level flow:

### Phase 1: Foundation (Days 1-3)
**Implement Gaps 0, 1, 5, 3 first - these are the foundation**

1. **Gap 0: Context Compression** (~403 lines)
   - Create `kosmos/compression/__init__.py`
   - Create `kosmos/compression/compressor.py`
   - Test: Compress sample notebook (42K → 500 tokens)
   - **Target**: 20:1 compression ratio

2. **Gap 1: State Manager** (~410 lines)
   - Create `kosmos/world_model/artifacts.py`
   - Test: Save/retrieve JSON artifacts
   - Test: Generate cycle summary
   - **Target**: Human-readable artifacts + optional graph

3. **Gap 5: Discovery Validation** (~407 lines)
   - Create `kosmos/validation/__init__.py`
   - Create `kosmos/validation/scholar_eval.py`
   - Test: Validate sample finding (8 dimensions)
   - **Target**: 75% validation rate

4. **Gap 3: Agent Integration** (~322 lines + modifications)
   - Create `kosmos/agents/skill_loader.py`
   - Modify `kosmos/agents/data_analyst.py`
   - Test: Load skills for "single_cell_analysis"
   - **Target**: 120+ skills auto-loadable

**Checkpoint 1**: All foundation components working independently

### Phase 2: Orchestration (Days 4-6)
**Implement Gap 2 - the "brain" of the system**

1. **Gap 2: Task Generation** (~1,945 lines)
   - Create `kosmos/orchestration/__init__.py`
   - Create `kosmos/orchestration/plan_creator.py`
   - Create `kosmos/orchestration/plan_reviewer.py`
   - Create `kosmos/orchestration/delegation.py`
   - Create `kosmos/orchestration/novelty_detector.py`
   - Create `kosmos/orchestration/instructions.yaml`
   - Test: Create plan for cycle 1
   - Test: Review plan (5 dimensions)
   - Test: Check novelty (vector similarity)
   - **Target**: 80% plan approval, 75% novelty threshold

**Checkpoint 2**: Orchestration components working, can generate valid plans

### Phase 3: Integration (Days 7-8)
**Wire everything together**

1. **Research Workflow** (~687 lines)
   - Create `kosmos/workflow/__init__.py`
   - Create `kosmos/workflow/research_loop.py`
   - Test: Run 5-cycle workflow end-to-end
   - **Target**: Complete 5 cycles with validated findings

2. **Report Generation** (~536 lines) - Optional enhancement
   - Create `kosmos/reporting/__init__.py`
   - Create `kosmos/reporting/report_synthesizer.py`
   - Test: Generate report from findings
   - **Target**: Publication-quality output

**Checkpoint 3**: Full system working end-to-end

### Phase 4: Validation (Days 9-10)
**Verify everything works**

1. **Integration Testing**
   - Run 5-cycle research workflow
   - Measure compression ratios (target: 20:1)
   - Measure plan approval rates (target: 80%)
   - Measure validation rates (target: 75%)

2. **Performance Tuning**
   - Optimize if performance below targets
   - Fix any integration issues
   - Document any deviations from spec

**Final Checkpoint**: System meets all success criteria

---

## 🔄 Implementation Loop for Each Component

For each file you create, follow this loop:

```
1. READ relevant section in OPENQUESTIONS_SOLUTION.md
   ↓ Understand WHY this component exists and how it works

2. REVIEW code template in KOSMOS_GAP_IMPLEMENTATION_PROMPT.md
   ↓ See WHAT needs to be implemented

3. THINK about integration
   ↓ How does this fit with other components?

4. IMPLEMENT the code
   ↓ Write with understanding, not just copying

5. TEST immediately
   ↓ Verify it works before moving to next component

6. INTEGRATE with existing code
   ↓ Make sure it connects properly

7. VERIFY performance
   ↓ Check against success criteria
```

**Do NOT batch implementation** - test each component as you build it.

---

## 🎯 Success Criteria

Your implementation is successful when ALL of these are true:

### Gap 0: Context Compression ✅
- [ ] Achieves 20:1 compression ratio (100K+ → 5K tokens)
- [ ] Maintains critical information (findings, statistics)
- [ ] Lazy loading functional for full content
- [ ] Integrates with State Manager

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
- [ ] All 5 orchestration components working together

### Gap 3: Agent Integration ✅
- [ ] 120+ skills loadable by domain
- [ ] Skills auto-injected based on task type
- [ ] Consistent JSON output format
- [ ] Error recovery functional
- [ ] DataAnalyst enhanced with SkillLoader

### Gap 4: Language/Tooling ⚠️
- [ ] Python-first approach documented
- [ ] Library mappings defined (R → Python)
- [ ] LLM code generation functional
- [ ] Note: Sandboxed execution is future work

### Gap 5: Discovery Validation ✅
- [ ] ~75% validation rate achieved
- [ ] 8-dimension scoring consistent
- [ ] Approval thresholds enforced (≥0.75 overall, ≥0.70 rigor)
- [ ] Actionable feedback provided

### Integration ✅
- [ ] 5-cycle workflow completes successfully
- [ ] All components integrate without errors
- [ ] Performance meets targets
- [ ] Final report generation functional

---

## 🐛 Troubleshooting

### Issue: "Skills directory not found"

```bash
# Check if subtree was added correctly
ls kosmos-claude-scientific-skills/scientific-skills/
# Should show: aeon, alphafold-database, anndata, etc.

# If missing, the subtree wasn't added properly
# Go back and run PROMPT_1_SETUP_REPOSITORIES.md
```

### Issue: "Import errors"

```bash
# Verify __init__.py files exist
ls kosmos/compression/__init__.py
ls kosmos/orchestration/__init__.py
ls kosmos/validation/__init__.py
ls kosmos/workflow/__init__.py
ls kosmos/reporting/__init__.py

# Create them if missing (see implementation guide)
```

### Issue: "Missing dependencies"

```bash
# Install required packages
pip install sentence-transformers pyyaml numpy anthropic

# Or add to pyproject.toml:
# sentence-transformers = ">=2.2.0"
# pyyaml = ">=6.0"
# numpy = ">=1.24.0"
# anthropic = ">=0.18.0"
```

### Issue: "Performance below targets"

**If compression ratio < 20:1:**
- Review hierarchical compression implementation
- Check if statistics extraction is working
- Verify LLM summarization is generating concise summaries

**If plan approval < 80%:**
- Review Plan Creator prompts in instructions.yaml
- Check if context from State Manager is sufficient
- Verify Plan Reviewer scoring is calibrated correctly

**If validation rate < 75%:**
- Review ScholarEval scoring dimensions
- Check if thresholds are too strict (can adjust)
- Verify findings have sufficient evidence

---

## 📊 Progress Tracking

As you implement, track your progress:

### Phase 1: Foundation
- [ ] Gap 0 complete + tested
- [ ] Gap 1 complete + tested
- [ ] Gap 5 complete + tested
- [ ] Gap 3 complete + tested
- [ ] All foundation components pass unit tests

### Phase 2: Orchestration
- [ ] Plan Creator complete + tested
- [ ] Plan Reviewer complete + tested
- [ ] Delegation Manager complete + tested
- [ ] Novelty Detector complete + tested
- [ ] instructions.yaml complete
- [ ] All orchestration components pass unit tests

### Phase 3: Integration
- [ ] Research Workflow complete + tested
- [ ] Report Synthesizer complete + tested (optional)
- [ ] 5-cycle integration test passes
- [ ] All success criteria met

---

## 🎓 Learning Checkpoints

At each phase, verify your understanding:

**After Phase 1:**
- Can you explain how context compression enables the system to function?
- Can you trace how a finding gets stored and retrieved from State Manager?
- Can you explain what each of the 8 ScholarEval dimensions measures?
- Can you show how skills are auto-loaded for a specific task type?

**After Phase 2:**
- Can you explain why exploration ratio decreases from 70% to 30%?
- Can you describe the Plan Creator → Reviewer → Revision flow?
- Can you explain how the novelty detector prevents redundant tasks?
- Can you trace how tasks are delegated to appropriate agents?

**After Phase 3:**
- Can you trace a complete cycle from context → plan → execution → validation → state update?
- Can you explain how all 6 gaps work together in the research loop?
- Can you identify the performance bottlenecks and optimization opportunities?
- Can you explain the design trade-offs that were made?

**If you can't answer these questions, you need to understand the architecture better before proceeding.**

---

## ✅ Final Checklist

Before considering implementation complete:

### Code Quality
- [ ] All 16 new files created
- [ ] All files have proper docstrings
- [ ] All imports work correctly
- [ ] No syntax errors
- [ ] Code follows project style

### Testing
- [ ] All unit tests pass
- [ ] 5-cycle integration test passes
- [ ] Performance metrics meet targets
- [ ] Error handling tested

### Documentation
- [ ] README updated with new architecture
- [ ] Gap solutions documented
- [ ] Usage examples provided
- [ ] Integration points documented

### Validation
- [ ] Compression: 20:1 ratio achieved
- [ ] Planning: 80%+ approval rate
- [ ] Validation: 75%+ pass rate
- [ ] Novelty: 75% similarity threshold working
- [ ] Integration: 5-cycle workflow completes

---

## 🚀 Next Steps After Implementation

Once implementation is complete and all tests pass:

1. **Run Extended Testing**
   ```bash
   # Run 20-cycle workflow
   python -m kosmos.workflow.research_loop --cycles 20
   ```

2. **Generate First Report**
   ```python
   from kosmos.reporting import ReportSynthesizer
   synthesizer = ReportSynthesizer()
   report = await synthesizer.generate_research_report(findings, objective)
   ```

3. **Performance Tuning**
   - Profile critical paths
   - Optimize slow components
   - Tune hyperparameters (thresholds, ratios, etc.)

4. **Documentation**
   - Document architecture decisions
   - Create usage guide
   - Add troubleshooting section

5. **Future Enhancements**
   - Implement Gap 4 fully (sandboxed execution)
   - Add knowledge graph layer (Gap 1 optional)
   - Add vector store (Gap 1 optional)
   - Optimize compression ratios
   - Add more scientific domains

---

## 📦 Summary

You are implementing solutions for 6 critical gaps in the Kosmos paper:

**Total Work**: 16 new files + 1 modified file (~3,487 lines)

**Implementation Time**: 8-10 days (with ULTRATHINKING and testing)

**Success Criteria**: All components working together in 20-cycle autonomous research loop

**Key Files to Reference**:
- `OPENQUESTIONS_SOLUTION.md` - WHY each solution works
- `KOSMOS_GAP_IMPLEMENTATION_PROMPT.md` - HOW to implement step-by-step

**Remember**: This is complex architectural work. Take your time, understand the WHY before implementing, test as you go, and verify performance at each checkpoint.

**Status**: 🟢 Ready to implement - all prerequisites met

Good luck! 🚀
