# PROMPT 1: Repository Setup for Kosmos Gap Implementation

**⚠️ CRITICAL: Run this in `/mnt/c/python/kosmos` (the original kosmos repository)**
**NOT in `/mnt/c/python/kosmos-research` - that's just the R&D directory**
**Repository**: https://github.com/jimmc414/kosmos
**Purpose**: Pull down required GitHub repositories (SETUP ONLY - NO CODE CHANGES)
**Next Step**: After this completes, copy 3 MD files, then run PROMPT 2

---

## What This Prompt Does

This is **SETUP ONLY**. It will:
1. ✅ Check if repositories already exist (skip if present)
2. ✅ Pull `kosmos-claude-scientific-skills` as git subtree (REQUIRED)
3. ✅ Clone 4 reference repositories to `kosmos-reference/`
4. ✅ Verify all repositories downloaded successfully
5. ❌ Does NOT make any code changes
6. ❌ Does NOT implement any gaps yet

---

## Instructions for AI Assistant

You are setting up the repository structure needed for Kosmos gap implementation. This is a **prerequisite step** - you will NOT implement any code yet.

### Step 1: Verify Current Location

First, verify you're in the ORIGINAL kosmos repository root:

```bash
pwd
# MUST show: /mnt/c/python/kosmos
# NOT: /mnt/c/python/kosmos-research (that's the R&D directory)
# NOT: /mnt/c/python/kosmos-research/R&D/kosmos (that's a clone in R&D)

ls -la
# Should show: kosmos/ directory (the Python package)
# Should show: pyproject.toml, README.md, etc.
```

**⚠️ CRITICAL CHECK**: If pwd shows anything other than `/mnt/c/python/kosmos`, **STOP** and navigate to the correct directory:
```bash
cd /mnt/c/python/kosmos
```

---

### Step 2: Check and Pull Scientific Skills (REQUIRED)

The `kosmos-claude-scientific-skills` repository contains 566 skill markdown files that are **REQUIRED** for the SkillLoader component (Gap 3) to function. This must be integrated as a git subtree.

```bash
# Check if already present
if [ -d "kosmos-claude-scientific-skills" ]; then
    echo "✅ kosmos-claude-scientific-skills already exists - skipping"
    ls kosmos-claude-scientific-skills/scientific-skills/ | head -10
else
    echo "📥 Adding kosmos-claude-scientific-skills as git subtree..."

    # Add as subtree
    git subtree add --prefix kosmos-claude-scientific-skills \
      https://github.com/jimmc414/kosmos-claude-scientific-skills.git \
      main --squash

    # Verify it worked
    if [ -d "kosmos-claude-scientific-skills/scientific-skills" ]; then
        echo "✅ Successfully added kosmos-claude-scientific-skills"
        echo "📊 Skill count:"
        ls kosmos-claude-scientific-skills/scientific-skills/ | wc -l
        echo "   (Should be 560+)"

        echo "📝 Sample skills:"
        ls kosmos-claude-scientific-skills/scientific-skills/ | head -10
    else
        echo "❌ ERROR: Failed to add kosmos-claude-scientific-skills"
        echo "Please check your git configuration and try again"
        exit 1
    fi
fi
```

**Expected Output**:
```
✅ Successfully added kosmos-claude-scientific-skills
📊 Skill count:
   566
📝 Sample skills:
aeon
alphafold-database
anndata
arboreto
astropy
benchling-integration
biomni
biopython
biorxiv-database
bioservices
```

---

### Step 3: Check and Clone Reference Repositories

These 4 repositories provide **patterns to reference** during implementation. They are cloned to `kosmos-reference/` (inside the main repo) so you can study their code while implementing.

```bash
# Create reference directory if it doesn't exist
if [ ! -d "kosmos-reference" ]; then
    mkdir -p kosmos-reference
    echo "✅ Created kosmos-reference/"
fi

cd kosmos-reference

# Repository 1: kosmos-karpathy (orchestration patterns)
if [ -d "kosmos-karpathy" ]; then
    echo "✅ kosmos-karpathy already exists - skipping"
else
    echo "📥 Cloning kosmos-karpathy..."
    git clone https://github.com/jimmc414/kosmos-karpathy.git
    if [ -d "kosmos-karpathy" ]; then
        echo "✅ Successfully cloned kosmos-karpathy"
    else
        echo "❌ ERROR: Failed to clone kosmos-karpathy"
    fi
fi

# Repository 2: kosmos-claude-skills-mcp (context compression patterns)
if [ -d "kosmos-claude-skills-mcp" ]; then
    echo "✅ kosmos-claude-skills-mcp already exists - skipping"
else
    echo "📥 Cloning kosmos-claude-skills-mcp..."
    git clone https://github.com/jimmc414/kosmos-claude-skills-mcp.git
    if [ -d "kosmos-claude-skills-mcp" ]; then
        echo "✅ Successfully cloned kosmos-claude-skills-mcp"
    else
        echo "❌ ERROR: Failed to clone kosmos-claude-skills-mcp"
    fi
fi

# Repository 3: kosmos-claude-scientific-writer (ScholarEval patterns)
if [ -d "kosmos-claude-scientific-writer" ]; then
    echo "✅ kosmos-claude-scientific-writer already exists - skipping"
else
    echo "📥 Cloning kosmos-claude-scientific-writer..."
    git clone https://github.com/jimmc414/kosmos-claude-scientific-writer.git
    if [ -d "kosmos-claude-scientific-writer" ]; then
        echo "✅ Successfully cloned kosmos-claude-scientific-writer"
    else
        echo "❌ ERROR: Failed to clone kosmos-claude-scientific-writer"
    fi
fi

# Repository 4: kosmos-agentic-data-scientist (additional patterns - optional)
if [ -d "kosmos-agentic-data-scientist" ]; then
    echo "✅ kosmos-agentic-data-scientist already exists - skipping"
else
    echo "📥 Cloning kosmos-agentic-data-scientist..."
    git clone https://github.com/jimmc414/kosmos-agentic-data-scientist.git
    if [ -d "kosmos-agentic-data-scientist" ]; then
        echo "✅ Successfully cloned kosmos-agentic-data-scientist"
    else
        echo "⚠️  WARNING: Failed to clone kosmos-agentic-data-scientist (optional)"
    fi
fi

# Return to kosmos repository root
cd ..
```

---

### Step 4: Verify Repository Setup

Run this verification to ensure everything is in place:

```bash
echo "========================================="
echo "Repository Setup Verification"
echo "========================================="
echo ""

# Check main repo
echo "📁 Main repository:"
pwd

# Check scientific skills (REQUIRED)
echo ""
echo "📁 Scientific Skills (REQUIRED):"
if [ -d "kosmos-claude-scientific-skills/scientific-skills" ]; then
    skill_count=$(ls kosmos-claude-scientific-skills/scientific-skills/ | wc -l)
    echo "✅ kosmos-claude-scientific-skills: $skill_count skills"
    if [ $skill_count -lt 500 ]; then
        echo "⚠️  WARNING: Expected 560+ skills, found $skill_count"
    fi
else
    echo "❌ ERROR: kosmos-claude-scientific-skills NOT FOUND"
    echo "   This is REQUIRED for Gap 3 implementation"
fi

# Check reference repos
echo ""
echo "📁 Reference repositories:"
if [ -d "kosmos-reference" ]; then
    cd kosmos-reference 2>/dev/null

    if [ -d "kosmos-karpathy" ]; then
        echo "✅ kosmos-karpathy (orchestration patterns - Gap 2)"
    else
        echo "❌ kosmos-karpathy NOT FOUND"
    fi

    if [ -d "kosmos-claude-skills-mcp" ]; then
        echo "✅ kosmos-claude-skills-mcp (compression patterns - Gap 0)"
    else
        echo "❌ kosmos-claude-skills-mcp NOT FOUND"
    fi

    if [ -d "kosmos-claude-scientific-writer" ]; then
        echo "✅ kosmos-claude-scientific-writer (validation patterns - Gap 5)"
    else
        echo "❌ kosmos-claude-scientific-writer NOT FOUND"
    fi

    if [ -d "kosmos-agentic-data-scientist" ]; then
        echo "✅ kosmos-agentic-data-scientist (optional patterns)"
    else
        echo "⚠️  kosmos-agentic-data-scientist NOT FOUND (optional)"
    fi

    cd ..
else
    echo "❌ kosmos-reference/ directory NOT FOUND"
fi

echo ""
echo "========================================="
```

**Expected Output**:
```
=========================================
Repository Setup Verification
=========================================

📁 Main repository:
/mnt/c/python/kosmos

📁 Scientific Skills (REQUIRED):
✅ kosmos-claude-scientific-skills: 566 skills

📁 Reference repositories:
✅ kosmos-karpathy (orchestration patterns - Gap 2)
✅ kosmos-claude-skills-mcp (compression patterns - Gap 0)
✅ kosmos-claude-scientific-writer (validation patterns - Gap 5)
✅ kosmos-agentic-data-scientist (optional patterns)

=========================================
```

---

### Step 5: Repository Structure Summary

After successful setup, your directory structure should look like this:

```
kosmos/                                      ← Original kosmos repo (where you are)
├── kosmos-claude-scientific-skills/         ← ADDED: Git subtree (REQUIRED)
│   └── scientific-skills/                   ← 566 skill markdown files
│       ├── aeon/
│       ├── alphafold-database/
│       ├── anndata/
│       └── ... (563 more)
├── kosmos-reference/                        ← ADDED: Reference repos (inside main repo)
│   ├── kosmos-karpathy/                     ← For orchestration patterns (Gap 2)
│   ├── kosmos-claude-skills-mcp/            ← For compression patterns (Gap 0)
│   ├── kosmos-claude-scientific-writer/     ← For validation patterns (Gap 5)
│   └── kosmos-agentic-data-scientist/       ← Optional additional patterns
├── kosmos/                                  ← Existing Python package
│   ├── agents/
│   ├── analysis/
│   ├── core/
│   └── ...
├── tests/
├── pyproject.toml
└── README.md
```

---

## ✅ Setup Complete - Next Steps

If all verifications passed, repository setup is complete!

### What You Should See:
- ✅ `kosmos-claude-scientific-skills/` exists with 566 skills
- ✅ `kosmos-reference/` contains 4 cloned repositories
- ✅ No code changes have been made yet
- ✅ Ready for implementation files

### ⚠️ If You See Errors:

**Error: "kosmos-claude-scientific-skills NOT FOUND"**
- This is CRITICAL - implementation will fail without it
- Re-run the git subtree command manually
- Check your git configuration and network connection

**Error: "Reference repository NOT FOUND"**
- Less critical - you can reference patterns online if needed
- Try cloning manually from GitHub
- Check network connection and GitHub access

---

## 🎯 NEXT STEP: Copy Implementation Files

**You (the user) must now manually copy 2-3 files** to the kosmos repository:

### REQUIRED (2 files):
```bash
# 1. KOSMOS_GAP_IMPLEMENTATION_PROMPT.md  ← Step-by-step implementation guide
# 2. OPENQUESTIONS_SOLUTION.md            ← Deep analysis (WHY solutions work)

# From the R&D directory
cd /mnt/c/python/kosmos-research/R&D/

# Copy to the ORIGINAL kosmos repository
cp KOSMOS_GAP_IMPLEMENTATION_PROMPT.md /mnt/c/python/kosmos/
cp OPENQUESTIONS_SOLUTION.md /mnt/c/python/kosmos/
```

### OPTIONAL (1 file - recommended for first-timers):
```bash
# 3. OPEN_QUESTIONS.md                    ← Problem identification (adds context)

# Copy to the ORIGINAL kosmos repository
cp OPEN_QUESTIONS.md /mnt/c/python/kosmos/
```

**Note**: OPENQUESTIONS_SOLUTION.md is self-contained and includes all problem statements from OPEN_QUESTIONS.md. The optional file adds valuable context for first-time implementers but isn't strictly required.

**After copying the files**, run **PROMPT 2: IMPLEMENT_GAPS.md** to start the actual implementation.

---

## 📊 Summary

**What Was Done:**
- ✅ Pulled kosmos-claude-scientific-skills as git subtree (REQUIRED for code)
- ✅ Cloned 4 reference repositories to kosmos-reference/
- ✅ Verified all repositories are present
- ✅ No code changes made yet

**What Comes Next:**
1. User copies 3 MD files to kosmos repo
2. User runs PROMPT 2 to start implementation
3. Implementation follows KOSMOS_GAP_IMPLEMENTATION_PROMPT.md step-by-step

**Status**: 🟢 Repository setup complete - ready for implementation files
