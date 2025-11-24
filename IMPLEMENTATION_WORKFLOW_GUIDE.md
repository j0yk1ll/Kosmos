# Kosmos Gap Implementation - Complete Workflow Guide

**Purpose**: Step-by-step guide for implementing the 6 Kosmos gaps
**Target**: Original https://github.com/jimmc414/kosmos repository
**Location**: `/mnt/c/python/kosmos` (NOT `/mnt/c/python/kosmos-research`)
**Total Time**: 8-10 days of focused work

---

## 📋 Complete Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│ START: Original kosmos repository                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Run PROMPT_1_SETUP_REPOSITORIES.md                 │
│ ✓ Pulls scientific-skills as git subtree (REQUIRED)        │
│ ✓ Clones 4 reference repos to kosmos-reference/            │
│ ✓ Verifies everything downloaded correctly                 │
│ ✓ NO code changes made yet                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Manual - Copy 3 MD files to repo root              │
│ • KOSMOS_GAP_IMPLEMENTATION_PROMPT.md                       │
│ • OPENQUESTIONS_SOLUTION.md                                 │
│ • OPEN_QUESTIONS.md (optional)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Run PROMPT_2_IMPLEMENT_GAPS.md                     │
│ ✓ Verifies prerequisites (repos + files)                   │
│ ✓ Reads OPENQUESTIONS_SOLUTION.md (WHY)                    │
│ ✓ Follows KOSMOS_GAP_IMPLEMENTATION_PROMPT.md (HOW)        │
│ ✓ Implements all 6 gaps in 3 phases                        │
│ ✓ Tests at each checkpoint                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ COMPLETE: Autonomous AI scientist system ready             │
│ • 16 new files + 1 modified (~3,487 lines)                 │
│ • 6 gaps fully implemented                                  │
│ • All tests passing                                         │
│ • Ready for 20-cycle research runs                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Files You Need

### Files to Provide for Implementation

Located in: `/mnt/c/python/kosmos-research/R&D/`

**Prompts (Give to AI):**

1. **PROMPT_1_SETUP_REPOSITORIES.md** (Step 1)
   - Run FIRST to pull down required repositories
   - Sets up git subtree and reference repos
   - No code changes

2. **PROMPT_2_IMPLEMENT_GAPS.md** (Step 3)
   - Run AFTER copying MD files
   - Implements all 6 gaps
   - Full implementation with testing

**Implementation Docs (Copy in Step 2):**

3. **KOSMOS_GAP_IMPLEMENTATION_PROMPT.md** (REQUIRED)
   - Detailed HOW-TO guide
   - Exact code templates
   - Step-by-step instructions

4. **OPENQUESTIONS_SOLUTION.md** (REQUIRED)
   - Deep WHY analysis
   - Evidence and metrics
   - Architectural context
   - **Self-contained** (includes problem statements)

5. **OPEN_QUESTIONS.md** (OPTIONAL - recommended for first-timers)
   - Original gap identification
   - Problem statements before solutions
   - Adds valuable context (15 min read)
   - **Can skip if experienced** (solution doc has problem context)

---

## 🚀 Step-by-Step Execution

### STEP 1: Setup Repositories

**When**: First step in the original kosmos repo
**What**: Pulls down required GitHub repositories
**Time**: 5-10 minutes

```bash
# Navigate to the ORIGINAL kosmos repository
cd /mnt/c/python/kosmos

# IMPORTANT: NOT /mnt/c/python/kosmos-research (that's just R&D)
# Verify you're in the right place:
pwd  # Should show: /mnt/c/python/kosmos
ls   # Should show: kosmos/ pyproject.toml README.md

# Give the AI assistant this prompt:
# "Run PROMPT_1_SETUP_REPOSITORIES.md"

# OR run the commands manually from that file
```

**What Gets Created**:
```
kosmos/
├── kosmos-claude-scientific-skills/  ← NEW (git subtree, 566 skills)
├── kosmos-reference/                  ← NEW (reference repos)
│   ├── kosmos-karpathy/
│   ├── kosmos-claude-skills-mcp/
│   ├── kosmos-claude-scientific-writer/
│   └── kosmos-agentic-data-scientist/
└── ...existing files...
```

**Success Criteria**:
- ✅ `kosmos-claude-scientific-skills/` exists with 566 skills
- ✅ `kosmos-reference/` contains 4 repos
- ✅ Verification script passes

**If Issues**: See troubleshooting in PROMPT_1_SETUP_REPOSITORIES.md

---

### STEP 2: Copy Implementation Files

**When**: After Step 1 completes successfully
**What**: Manually copy 2-3 MD files to kosmos repo root
**Time**: 1 minute

#### REQUIRED (2 files):
```bash
# From your source location (e.g., R&D directory)
cd /mnt/c/python/kosmos-research/R&D/

# Copy REQUIRED files to the ORIGINAL kosmos repo root
cp KOSMOS_GAP_IMPLEMENTATION_PROMPT.md /mnt/c/python/kosmos/
cp OPENQUESTIONS_SOLUTION.md /mnt/c/python/kosmos/
```

#### OPTIONAL (1 file - recommended for first-timers):
```bash
# Copy OPTIONAL file (adds context for first-time implementers)
cp OPEN_QUESTIONS.md /mnt/c/python/kosmos/
```

#### Verify Files Copied:
```bash
ls /mnt/c/python/kosmos/*.md
```

**Minimum Required**:
```
kosmos/
├── KOSMOS_GAP_IMPLEMENTATION_PROMPT.md  ← NEW (REQUIRED: step-by-step guide)
├── OPENQUESTIONS_SOLUTION.md             ← NEW (REQUIRED: deep analysis)
├── README.md                             ← Existing
└── ...other existing files...
```

**Recommended for First-Timers**:
```
kosmos/
├── KOSMOS_GAP_IMPLEMENTATION_PROMPT.md  ← NEW (REQUIRED: step-by-step guide)
├── OPENQUESTIONS_SOLUTION.md             ← NEW (REQUIRED: deep analysis)
├── OPEN_QUESTIONS.md                     ← NEW (OPTIONAL: problem context)
├── README.md                             ← Existing
└── ...other existing files...
```

**Why Manual**: Gives you control over when implementation starts

**Note**: OPENQUESTIONS_SOLUTION.md is self-contained (includes problem statements). OPEN_QUESTIONS.md adds valuable context for understanding the analytical process but isn't required.

---

### STEP 3: Implement Gaps

**When**: After Steps 1 and 2 complete
**What**: Full implementation of 6 gaps
**Time**: 8-10 days

```bash
# Navigate to the ORIGINAL kosmos repository
cd /mnt/c/python/kosmos

# Verify you're in the right place
pwd  # Should show: /mnt/c/python/kosmos

# Give the AI assistant this prompt:
# "Run PROMPT_2_IMPLEMENT_GAPS.md"
```

**What This Does**:

**Phase 1 (Days 1-3): Foundation**
- Implements Gap 0 (Context Compression) - 403 lines
- Implements Gap 1 (State Manager) - 410 lines
- Implements Gap 5 (Discovery Validation) - 407 lines
- Implements Gap 3 (Agent Integration) - 322+ lines
- **Checkpoint**: All foundation components tested

**Phase 2 (Days 4-6): Orchestration**
- Implements Gap 2 (Task Generation) - 1,945 lines
  - Plan Creator
  - Plan Reviewer
  - Delegation Manager
  - Novelty Detector
  - instructions.yaml
- **Checkpoint**: Orchestration components tested

**Phase 3 (Days 7-8): Integration**
- Implements Research Workflow - 687 lines
- Implements Report Synthesizer - 536 lines (optional)
- **Checkpoint**: 5-cycle end-to-end test passes

**Phase 4 (Days 9-10): Validation**
- Full integration testing
- Performance verification
- Documentation updates

**What Gets Created**:
```
kosmos/
├── kosmos/
│   ├── compression/           ← NEW (Gap 0)
│   │   ├── __init__.py
│   │   └── compressor.py
│   ├── orchestration/         ← NEW (Gap 2)
│   │   ├── __init__.py
│   │   ├── plan_creator.py
│   │   ├── plan_reviewer.py
│   │   ├── delegation.py
│   │   ├── novelty_detector.py
│   │   └── instructions.yaml
│   ├── validation/            ← NEW (Gap 5)
│   │   ├── __init__.py
│   │   └── scholar_eval.py
│   ├── workflow/              ← NEW (Integration)
│   │   ├── __init__.py
│   │   └── research_loop.py
│   ├── reporting/             ← NEW (Enhancement)
│   │   ├── __init__.py
│   │   └── report_synthesizer.py
│   ├── world_model/
│   │   └── artifacts.py       ← NEW (Gap 1)
│   └── agents/
│       ├── skill_loader.py    ← NEW (Gap 3)
│       └── data_analyst.py    ← MODIFIED (Gap 3)
```

**Success Criteria**:
- ✅ All 16 new files created + 1 modified
- ✅ All unit tests pass
- ✅ 5-cycle integration test passes
- ✅ Performance meets targets:
  - Compression: 20:1 ratio
  - Plan approval: 80%+
  - Validation: 75%+
  - Novelty: 75% threshold

---

## 🎯 Key Design Decisions

### Why Two Prompts?

**Separation of Concerns**:
- **Prompt 1**: Setup only (idempotent, safe to re-run)
- **Prompt 2**: Implementation (complex, requires understanding)

**User Control**:
- Can verify repos before implementing
- Can place MD files where needed
- Can pause between steps

**Error Recovery**:
- If Prompt 1 fails, fix and re-run (no code lost)
- If Prompt 2 fails mid-implementation, you know where you are

### Why Manual File Copy?

**Explicit Checkpoint**:
- Forces verification that Step 1 succeeded
- User consciously starts implementation phase
- Clear handoff between setup and implementation

**Flexibility**:
- User can modify MD files if needed
- User can add notes or customizations
- User can version control the implementation docs

### Why Check for Existing Repos?

**Idempotency**:
- Can re-run Prompt 1 without issues
- Won't re-download if already present
- Safe for iterative workflows

**Efficiency**:
- Doesn't waste time re-cloning
- Preserves any local modifications to reference repos
- Network-friendly

---

## 🔍 Verification at Each Step

### After Step 1 (Repositories):

```bash
cd /mnt/c/python/kosmos

# Should see:
ls kosmos-claude-scientific-skills/scientific-skills/ | wc -l
# Output: 566 (or close to it)

ls kosmos-reference/
# Output: kosmos-karpathy kosmos-claude-skills-mcp ...
```

### After Step 2 (Files Copied):

**Minimum Required (2 files)**:
```bash
ls *.md
# Should show at minimum:
# KOSMOS_GAP_IMPLEMENTATION_PROMPT.md  ← REQUIRED
# OPENQUESTIONS_SOLUTION.md            ← REQUIRED
# README.md                            ← Existing
```

**Recommended (3 files)**:
```bash
ls *.md
# Ideally shows:
# KOSMOS_GAP_IMPLEMENTATION_PROMPT.md  ← REQUIRED
# OPENQUESTIONS_SOLUTION.md            ← REQUIRED
# OPEN_QUESTIONS.md                    ← OPTIONAL (recommended)
# README.md                            ← Existing
```

### After Step 3 (Implementation):

```bash
# Check new directories
ls kosmos/compression kosmos/orchestration kosmos/validation kosmos/workflow kosmos/reporting

# Run tests
pytest tests/

# Run 5-cycle integration test
python -m kosmos.workflow.research_loop --cycles 5
```

---

## ⚠️ Common Issues and Solutions

### Issue: "kosmos-claude-scientific-skills not found" during implementation

**Cause**: Prompt 1 didn't complete successfully or subtree wasn't added
**Solution**:
```bash
cd /mnt/c/python/kosmos
git subtree add --prefix kosmos-claude-scientific-skills \
  https://github.com/jimmc414/kosmos-claude-scientific-skills.git \
  main --squash
```

### Issue: "Implementation files not found" when starting Prompt 2

**Cause**: Step 2 (manual copy) wasn't completed
**Solution**: Copy the 3 MD files to repo root (see Step 2)

### Issue: "Not in kosmos repository root"

**Cause**: Running from wrong directory
**Solution**:
```bash
# Navigate to the ORIGINAL kosmos repository
cd /mnt/c/python/kosmos

# IMPORTANT: NOT /mnt/c/python/kosmos-research (that's R&D)

# Verify
pwd  # Should show: /mnt/c/python/kosmos
ls kosmos/  # Should see Python package directory
```

### Issue: Import errors during implementation

**Cause**: Missing dependencies
**Solution**:
```bash
pip install sentence-transformers pyyaml numpy anthropic
```

### Issue: Performance below targets

**Cause**: Implementation details need tuning
**Solution**: See troubleshooting section in PROMPT_2_IMPLEMENT_GAPS.md

---

## 📊 Progress Tracking

Track your progress through the workflow:

- [ ] **Step 1 Complete**: Repos pulled, verification passed
- [ ] **Step 2 Complete**: 3 MD files copied to repo root
- [ ] **Phase 1 Complete**: Foundation gaps (0,1,3,5) implemented + tested
- [ ] **Phase 2 Complete**: Orchestration gap (2) implemented + tested
- [ ] **Phase 3 Complete**: Integration (workflow + reporting) implemented + tested
- [ ] **Phase 4 Complete**: All tests pass, performance verified
- [ ] **Final**: 20-cycle research run successful

---

## 📚 Documentation Reference

**During Setup (Step 1)**:
- Read: PROMPT_1_SETUP_REPOSITORIES.md

**Before Implementation (Step 3)**:
- Read: OPENQUESTIONS_SOLUTION.md (WHY - 30-45 min)
- Read: KOSMOS_GAP_IMPLEMENTATION_PROMPT.md sections (HOW - 15-20 min)

**During Implementation (Step 3)**:
- Follow: PROMPT_2_IMPLEMENT_GAPS.md (overall guidance)
- Reference: KOSMOS_GAP_IMPLEMENTATION_PROMPT.md (detailed instructions)
- Reference: OPENQUESTIONS_SOLUTION.md (architectural context)

**For Debugging**:
- Check: Troubleshooting sections in each prompt
- Review: Success criteria in KOSMOS_GAP_IMPLEMENTATION_PROMPT.md
- Study: Evidence and metrics in OPENQUESTIONS_SOLUTION.md

---

## 🎓 Learning Path

**Before Starting**:
- Understand: What are the 6 gaps?
- Understand: Why was each gap blocking reproduction?
- Understand: How do the solutions address each gap?

**During Implementation**:
- Phase 1: How do foundation components work independently?
- Phase 2: How does orchestration enable strategic planning?
- Phase 3: How do all components integrate?

**After Completion**:
- Can you explain the complete 20-cycle research loop?
- Can you trace information flow through all components?
- Can you justify the design decisions made?

---

## ✅ Final Checklist

Before considering the workflow complete:

### Setup
- [ ] Prompt 1 ran successfully
- [ ] All repositories present and verified
- [ ] 3 MD files copied to repo root

### Implementation
- [ ] All 16 new files created
- [ ] 1 file modified (data_analyst.py)
- [ ] All imports working
- [ ] No syntax errors

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] 5-cycle workflow completes
- [ ] Performance meets targets

### Validation
- [ ] Compression: 20:1 achieved
- [ ] Planning: 80%+ approval
- [ ] Validation: 75%+ pass rate
- [ ] All success criteria met

### Documentation
- [ ] README updated
- [ ] Usage examples provided
- [ ] Architecture documented

---

## 🚀 Success!

Once all checklists are complete:

**You now have**: A fully functional autonomous AI scientist system with:
- Context compression enabling 100K+ token workflows
- Hybrid state management for coherent multi-cycle research
- Strategic task generation with quality control
- Auto-loading domain expertise (120+ skills)
- Multi-dimension discovery validation
- Publication-quality report generation

**You can now**:
- Run 20-cycle autonomous research
- Generate validated scientific discoveries
- Produce publication-quality reports
- Extend with additional domains
- Scale to more complex research objectives

**Next steps**:
- Run your first 20-cycle research project
- Tune hyperparameters for your domain
- Add domain-specific skills
- Contribute improvements back to the project

Congratulations! 🎉
