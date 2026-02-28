# ICML 2026 Submission Checklist

## Pre-Submission Anonymization

### Step 1: Create Anonymous Repository

- [ ] Run anonymization script:
  ```bash
  # Windows PowerShell
  ./create_anonymous_submission.ps1
  
  # Linux/Mac
  bash create_anonymous_submission.sh
  ```

- [ ] Verify anonymous repository created at: `../ACE-anonymous-submission/`
- [ ] Verify submission archive created: `ACE-submission-YYYYMMDD.zip`

### Step 2: Verify Anonymization

#### Check Git Configuration
```bash
cd ../ACE-anonymous-submission
git config user.name    # Should show: Anonymous Researcher
git config user.email   # Should show: anonymous@institution.edu
```

- [ ] Git config shows anonymous information

#### Check Git History
```bash
git log --pretty=format:"%an <%ae>" | sort -u
```

- [ ] All commits show: `Anonymous Researcher <anonymous@institution.edu>`
- [ ] No personal names or emails in commit history

#### Check Paper
```bash
grep -i "anonymous" paper/paper.tex
```

- [ ] Paper uses "Anonymous Author(s)" (line 84)
- [ ] Affiliation is "Anonymous Institution" (line 87)
- [ ] Email is "anonymous@institution.edu" (line 89)

#### Check Code Files
```bash
# Search for any personal information
grep -r "Patrick\|Cooper\|patrick.allen.cooper" . --exclude-dir=.git
```

- [ ] No personal names in code files
- [ ] No personal emails in code files
- [ ] No personal paths (e.g., `/Users/patri/`)

### Step 3: Prepare Submission Materials

#### Required Files

- [ ] Paper PDF (compiled from `paper/paper.tex`)
- [ ] Supplementary material (if any)
- [ ] Code archive: `ACE-submission-YYYYMMDD.zip`
- [ ] README for code (already in repo)

#### Code Archive Contents

The archive should contain:
- [ ] Source code (`.py` files)
- [ ] Configuration files (`environment.yml`, `requirements.txt`)
- [ ] Job scripts (`jobs/*.sh`)
- [ ] Test suite (`tests/`)
- [ ] Documentation (`README.md`, `guidance_documents/`)
- [ ] Paper source (`paper/`)
- [ ] Example results (small subset in `results/`)

The archive should NOT contain:
- [ ] Large checkpoint files (`.pt`, `.pth`)
- [ ] Full experimental results (keep only examples)
- [ ] Build artifacts (`__pycache__`, `.pyc`)
- [ ] Git history (`.git/`)
- [ ] Personal identifying information
- [ ] This checklist file

### Step 4: Final Verification

- [ ] Compile paper from anonymous repository
- [ ] Test code execution from anonymous archive
- [ ] Run basic tests: `pytest tests/ -m "not slow"`
- [ ] Verify all paths are relative (no absolute paths)
- [ ] Check that experiments can be reproduced

### Step 5: ICML Submission

- [ ] Upload paper PDF to ICML submission system
- [ ] Upload code archive (or provide anonymous GitHub link)
- [ ] Fill out paper metadata (title, abstract, keywords)
- [ ] Confirm double-blind submission requirements met
- [ ] Submit before deadline

## Post-Submission

- [ ] Keep original repository (with full history) safe
- [ ] Do not update anonymous repository during review
- [ ] Prepare response to reviewers (keep anonymous)

## After Acceptance (De-anonymization)

When paper is accepted:

- [ ] Update `paper/paper.tex` with real author names
- [ ] Update affiliations and contact information
- [ ] Push original repository to public GitHub
- [ ] Link accepted paper to code repository
- [ ] Add acknowledgments if needed
- [ ] Update README with paper link

## Emergency Contact

If you need to communicate with reviewers during review:

- Use anonymous email: anonymous@institution.edu (or conference-provided channel)
- Do not reveal identity
- Refer to "the authors" in third person
