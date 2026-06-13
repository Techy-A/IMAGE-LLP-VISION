# IMAGE-LLP-VISION

Pairwise image evaluation system using five vision-language models.
Images are ranked via a prompt-group round-robin tournament - images are scored only against other images within the same prompt group (e.g. prompt sheet), Elo and Bradley-Terry MLE ratings are updated per group after each match, and final results are exported to combined and per-group CSVs, JSON, and charts.

**Scorers:** PickScore · HPSv2 · ImageReward · VQAScore · VLM-as-a-Judge  
**Stack:** Python 3.10+, PyTorch 2.2+, HuggingFace Transformers 4.x, InstructBLIP  
**Platforms:** macOS (Apple Silicon / Intel MPS/CPU) · Linux + CUDA (tested on 2× RTX A6000)

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Local Setup - Mac](#2-local-setup--mac)
3. [Cloud Setup - Linux + CUDA (SSH)](#3-cloud-setup--linux--cuda-ssh)
4. [Uploading the Project to a Cloud Server](#4-uploading-the-project-to-a-cloud-server)
5. [Running the Tournament](#5-running-the-tournament)
6. [All CLI Commands](#6-all-cli-commands)
7. [Running in the Background (nohup)](#7-running-in-the-background-nohup)
8. [Output Files Reference](#8-output-files-reference)
9. [Resuming an Interrupted Run](#9-resuming-an-interrupted-run)
10. [Enabling the 5th Metric (ImageReward)](#10-enabling-the-5th-metric-imagereward)
11. [Architecture & Error Handling](#11-architecture--error-handling)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Structure

```
InstructBLIP/
├── main.py                    # Entry point
├── config.yaml                # All configuration — never hardcode values
├── requirements.txt           # Mac / CPU dependencies
├── requirements-cuda.txt      # Linux / CUDA dependencies
├── setup_cuda.sh              # One-shot CUDA environment setup script
├── fix_imagereward.sh         # Install ImageReward (5th metric)
├── verify_imagereward.py      # Verify ImageReward loads correctly
│
├── data/
│   ├── Metric Evaluation Text to Image.xlsx   # Main evaluation file
│   └── example_metadata.csv                   # Example CSV input
│
├── src/
│   ├── models/                # Vision model adapters
│   │   ├── base_vision_model.py
│   │   ├── instructblip_adapter.py   # Shared by VQAScore + VLMJudge
│   │   ├── clip_adapter.py
│   │   ├── blip_adapter.py
│   │   └── blip2_adapter.py
│   ├── scoring/               # Scorer implementations
│   │   ├── base_scorer.py
│   │   ├── pickscore.py
│   │   ├── hpsv2.py
│   │   ├── image_reward.py
│   │   ├── vqascore.py
│   │   └── vlm_judge.py
│   ├── visualization/
│   │   ├── heatmap_generator.py
│   │   └── chart_generator.py
│   ├── arena.py               # Tournament orchestration
│   ├── ensemble.py            # Weighted score aggregation
│   ├── elo_system.py
│   ├── bradley_terry_mle.py
│   ├── cli.py                 # Click CLI
│   ├── csv_loader.py          # CSV + multi-sheet XLSX loading
│   ├── image_loader.py
│   ├── results_exporter.py
│   ├── checkpoint_manager.py
│   ├── config_manager.py
│   └── data_models.py
│
├── tests/                     # Mirrors src/ structure
├── results/                   # Created at runtime
├── checkpoints/               # Auto-saved during tournament
└── extracted_images/          # Embedded images extracted from XLSX
```

---

## 2. Local Setup - Mac

Tested on macOS (Apple Silicon and Intel). Uses the `.venv312` environment.

```bash
# 1. Go to the project folder
cd ~/Downloads/IMAGE-LLP-VISION

# 2. Create a Python 3.12 virtual environment (once)
python3.12 -m venv .venv312

# 3. Activate it
source .venv312/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Install ImageReward (optional 5th metric, ~2 GB download)
./fix_imagereward.sh

# 6. Verify everything works
python -m pytest tests/ -q        # expect: 265 passed
python verify_imagereward.py      # expect: PASS
```

To activate the environment in future sessions:
```bash
source .venv312/bin/activate
```

---

## 3. Cloud Setup - Linux + CUDA (SSH)

Tested on 2× NVIDIA RTX A6000 (49 GB VRAM each), CUDA 12.8, Python 3.12.

After uploading the project (see Section 4), run the one-shot setup script:

```bash
cd ~/InstructBLIP
bash setup_cuda.sh
```

This script:
1. Creates `.venv_cuda/` with a fresh virtual environment
2. Installs PyTorch from the CUDA 12.8 wheel index (plain `pip install torch` gives CPU-only)
3. Installs all other dependencies including ImageReward
4. Runs `265 passed` unit tests to verify the install
5. Runs `verify_imagereward.py` to confirm the 5th metric loads

After setup, activate the environment:
```bash
source .venv_cuda/bin/activate
```

**Check CUDA is visible:**
```bash
python -c "import torch; print(torch.cuda.device_count(), 'GPU(s)'); print(torch.cuda.get_device_name(0))"
```

---

## 4. Uploading the Project to a Cloud Server

Choose one of these methods depending on your situation.

### Option A — rsync (recommended, fastest)

```bash
# From your Mac, in the terminal:
rsync -avz --exclude '.venv*' --exclude '__pycache__' --exclude '*.pyc' \
  ~/Downloads/IMAGE-LLP-VISION/ \
  your_user@your.server.ip:~/InstructBLIP/
```

Re-run the same command at any time to sync only changed files.

### Option B - scp (simple, no exclusions)

```bash
scp -r ~/Downloads/IMAGE-LLP-VISION your_user@your.server.ip:~/
```

### Option C - Git (best for ongoing development)

```bash
# On your Mac — push to GitHub/GitLab
cd ~/Downloads/IMAGE-LLP-VISION
git init
git add .
git commit -m "initial"
git remote add origin https://github.com/YOUR_USERNAME/InstructBLIP.git
git push -u origin main

# On the server — clone it
ssh your_user@your.server.ip
git clone https://github.com/YOUR_USERNAME/InstructBLIP.git
cd InstructBLIP
```

> **Do not upload** `.venv*/`, `checkpoints/`, `results/`, or `extracted_images/` —
> these are large and/or machine-specific. The `.gitignore` excludes them automatically.
> Recreate the venv on the server with `bash setup_cuda.sh`.

### Connecting via SSH

```bash
# Basic connection
ssh your_user@your.server.ip

# With a key file
ssh -i ~/.ssh/your_key.pem your_user@your.server.ip

# Keep the connection alive (prevents timeout during long runs)
ssh -o ServerAliveInterval=60 your_user@your.server.ip
```

---

## 5. Running the Tournament

All commands must be run from the project root with the venv active.

### Quickstart - the real XLSX file

```bash
mkdir -p results

python main.py tournament \
  --data        "data/Metric Evaluation Text to Image.xlsx" \
  --config      config.yaml \
  --sheet-range "Mermaid Metric:Underwatercity Metric" \
  --cell-range  "E2:E11" \
  --export-csv  results/evaluation_scores.csv \
  --output      results/
```

This runs 50 images divided into 5 prompt groups (10 images each). It generates comparisons only within the same group, resulting in 45 matches per group (225 total matches) instead of 1225. It writes:
- `results/evaluation_scores.csv` — combined ranked metrics table with a `group` column
- `results/evaluation_scores_<group_name>.csv` — individual ranked metrics table per group
- `results/tournament_results.json` — full match data for visualization

### Standard CSV input

```bash
python main.py tournament \
  --data       data/example_metadata.csv \
  --config     config.yaml \
  --export-csv results/evaluation_scores.csv \
  --output     results/
```

### Multi-sheet XLSX - how it works

When images are embedded in spreadsheet cells (not stored as file paths),
use `--sheet-range` and `--cell-range` together:

```bash
python main.py tournament \
  --data        "data/Metric Evaluation Text to Image.xlsx" \
  --config      config.yaml \
  --sheet-range "Mermaid Metric:Underwatercity Metric"  \
  --cell-range  "E2:E11" \
  --export-csv  results/evaluation_scores.csv
```

- `--sheet-range` specifies the first and last sheet name (inclusive), separated by `:`
- `--cell-range` specifies which cells to extract per sheet (single column only, e.g. `E2:E11`)
- Both flags must always be used together
- Embedded images are extracted to `extracted_images/` automatically

---

## 6. All CLI Commands

### Top-level help

```bash
python main.py --help
python main.py --version
```

### Tournament

```bash
python main.py tournament --help

# All flags:
python main.py tournament \
  --data        <path>        # CSV or XLSX file (required)
  --config      config.yaml   # Config file (required)
  --output      results/      # Save tournament_results.json here
  --export-csv  results/scores.csv  # Save per-image metrics CSV
  --checkpoint  checkpoints/checkpoint_100.json  # Resume from here
  --sheet-range "Sheet1:Sheet5"   # XLSX multi-sheet mode
  --cell-range  "E2:E11"          # XLSX cell range (with --sheet-range)
```

### Visualize

```bash
python main.py visualize --help

# Generate all 4 charts
python main.py visualize \
  --results results/tournament_results.json \
  --output  results/viz/

# Generate specific charts
python main.py visualize \
  --results results/tournament_results.json \
  --output  results/viz/ \
  --types   rankings \
  --types   heatmap

# Available --types values:
#   heatmap      --> pairwise win/loss matrix
#   rankings     --> bar chart of final Elo / BT-MLE rankings
#   progression  --> Elo ratings changing match-by-match
#   distribution --> histogram of ensemble score distribution
#   all          --> all four (default)
```

### Benchmark

```bash
python main.py benchmark --help

# Single model
python main.py benchmark \
  --models clip \
  --images data/example_metadata.csv \
  --config config.yaml

# Multiple models, 20 iterations
python main.py benchmark \
  --models clip --models hpsv2 --models pickscore \
  --images data/example_metadata.csv \
  --config config.yaml \
  --iterations 20

# Benchmarkable model names: clip, blip, blip2, instructblip, pickscore, hpsv2, image_reward
```

### Unit tests

```bash
# Run all 265 tests
python -m pytest tests/ -q

# Run a specific test file
python -m pytest tests/test_arena.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## 7. Running in the Background (nohup)

Use this on both Mac and Linux so the run continues after the terminal closes
or the display sleeps.

```bash
mkdir -p results

nohup python main.py tournament \
  --data        "data/Metric Evaluation Text to Image.xlsx" \
  --config      config.yaml \
  --sheet-range "Mermaid Metric:Underwatercity Metric" \
  --cell-range  "E2:E11" \
  --export-csv  results/evaluation_scores.csv \
  --output      results/ \
  > results/tournament.log 2>&1 &

echo "Started. PID: $!"
```

Monitor progress:
```bash
tail -f results/tournament.log
```

Check if it is still running:
```bash
ps aux | grep main.py
```

Stop it:
```bash
kill <PID>
```

After it finishes, generate charts:
```bash
python main.py visualize \
  --results results/tournament_results.json \
  --output  results/viz/
```

---

## 8. Output Files Reference

### `results/evaluation_scores.csv`

One row per image, sorted by rank (rank 1 = best).

| Column | Description |
|--------|-------------|
| `rank` | Final position within the group. 1 = best. |
| `image_id` | Image ID from the input CSV/XLSX |
| `image_path` | Path to the image file |
| `group` | The prompt group (e.g. XLSX sheet name or prompt text) |
| `prompt` | Text prompt used to generate the image |
| `model` | Which AI model generated the image |
| `elo_rating` | Elo score after all matches within the group (higher = better) |
| `bt_rating` | Bradley-Terry MLE score within the group (used for final rank) |
| `wins` | Pairwise comparisons won |
| `losses` | Pairwise comparisons lost |
| `matches_played` | Total comparisons (wins + losses) |
| `win_rate` | wins ÷ matches_played (1.0 = won all) |
| `avg_ensemble_score` | Weighted average across all scorers |
| `avg_pickscore` | PickScore aesthetic preference average |
| `avg_hpsv2` | HPSv2 Human Preference Score average |
| `avg_image_reward` | ImageReward text-image alignment average |
| `avg_vqascore` | VQAScore visual question-answering average |
| `avg_vlm_judge` | VLM-as-a-judge verdict average |

### `results/evaluation_scores_<group_name>.csv`

Individual CSV files containing ranked metrics tables specifically for each prompt group (e.g. `evaluation_scores_Mermaid_Metric.csv`), containing 10 rows each.

### `results/tournament_results.json`

Full machine-readable tournament data: every match, all Elo/BT ratings,
final rankings, image metadata, run timestamps. Input for `visualize` mode.

### `results/viz/`

PNG charts generated by `python main.py visualize` (both combined and individual per-group charts, totaling 24 files):
- `heatmap.png` / `heatmap_<group>.png` — pairwise win/loss matrices
- `rankings.png` / `rankings_<group>.png` — standings bar charts
- `progression.png` / `progression_<group>.png` — Elo rating history per image
- `distribution.png` / `distribution_<group>.png` — score distribution histograms

### `checkpoints/`

Auto-saved every 100 matches (configurable in `config.yaml`).
Used to resume interrupted runs (see Section 9).

---

## 9. Resuming an Interrupted Run

The tournament saves a checkpoint every 100 matches.
If interrupted, resume from the latest checkpoint:

```bash
# List available checkpoints
ls -lh checkpoints/

# Resume (replace checkpoint_100.json with the latest file)
python main.py tournament \
  --data        "data/Metric Evaluation Text to Image.xlsx" \
  --config      config.yaml \
  --sheet-range "Mermaid Metric:Underwatercity Metric" \
  --cell-range  "E2:E11" \
  --export-csv  results/evaluation_scores.csv \
  --output      results/ \
  --checkpoint  checkpoints/checkpoint_100.json
```

The run will skip all already-completed matches and pick up where it left off.

---

## 10. Enabling the 5th Metric (ImageReward)

ImageReward requires three extra pip packages (~2 GB checkpoint download on first use).
The tournament works at 4/5 metrics without it — this is optional.

```bash
source .venv312/bin/activate   # or .venv_cuda on Linux

./fix_imagereward.sh           # installs + verifies
```

What the script installs and why:

| Package | Without it you get |
|---|---|
| `image-reward` | `No module named 'ImageReward'` |
| `clip-anytorch` | `No module named 'clip'` |
| `setuptools<81` | `No module named 'pkg_resources'` |
| `diffusers>=0.29,<0.30` | `module 'torch' has no attribute 'xpu'` |

Manual install equivalent:
```bash
pip install "setuptools<81" "diffusers>=0.29,<0.30" image-reward clip-anytorch
python verify_imagereward.py   # expect: PASS
```

After install you should see `✅ Active scorers (5/5)` when the tournament starts.

---

## 11. Architecture & Error Handling

### Scorer failure handling

If a scorer's `compare()` raises during a match, the failure is counted in
`TournamentMetadata.failed_scorers` and that scorer is excluded from the match.
`Ensemble.aggregate_scores` renormalises the remaining scorers' weights so the
match still produces a valid result. Only if every scorer fails does the ensemble
return a neutral 0.5. The tournament never crashes due to a single scorer failing.

Console output when a scorer is skipped at load time:
```
⚠️  ImageReward failed to load: ...
    → Tip: `pip install image-reward` to enable this scorer.
✅ Active scorers (4/5): hpsv2, pickscore, vlm_judge, vqa
⚠️  Degraded run — ensemble will renormalise without: image_reward
```

### InstructBLIP sharing

VQAScore and VLMJudge both use InstructBLIP internally. The adapter is loaded
once and injected into both scorers — there is no duplicate model load.

### VQAScore Improvements

Four major improvements have been implemented for the `VQAScore` scorer:
1. **Prompt-Aware Fidelity**: Appends prompt-specific validation questions ("Does this image accurately depict '<prompt>'?") rather than only asking generic quality questions.
2. **Robust Response Parser**: Uses regex word boundaries (`\byes\b`/`\bno\b`) and a first-signal-wins rule to avoid false matches (e.g. `"notable"` triggering a negative match because of `"no"`).
3. **Full Score Range**: Outputs span the full `[0.0, 1.0]` range instead of being artificially capped at `[0.2, 0.8]`, resulting in much better ranking discrimination.
4. **Batch Caching**: Implements local caching for batch comparisons to avoid re-running InstructBLIP on the same image-prompt combination multiple times in the tournament.

### Quantization

| Setting | VRAM | Use on |
|---|---|---|
| `4bit` | ~2 GB | CUDA (recommended for InstructBLIP on GPU) |
| `8bit` | ~4 GB | CUDA |
| `fp16` | ~10 GB | Mac MPS / CPU (default for Mac) |
| `fp32` | ~20 GB | Not recommended |

The `config.yaml` sets `4bit` for the CUDA server and `fp16` is the fallback
when CUDA is not available (bitsandbytes requires CUDA for quantization).

### Elo correctness

Elo updates express the score from the **winner's perspective** (not image_a's).
When image_b wins (ensemble_score < 0.5), `winner_score = 1.0 - ensemble_score`
is passed to `EloSystem.update_ratings`. This was the root cause of an earlier
bug where image_b winning caused it to lose Elo points — now fixed.

---

## 12. Troubleshooting

### HPSv2 fails with "You have to specify input_ids"

HPSv2 uses a CLIP-based model. The full forward pass requires both image and
text tokens. The scorer now calls `model.get_image_features()` (image-only)
instead of `model(**inputs)`. If you see this error, verify you have the
latest `src/scoring/hpsv2.py`.

### ImageReward fails with "module 'torch' has no attribute 'xpu'"

The latest `diffusers` package requires torch ≥ 2.4 (for `torch.xpu`). Pin it:
```bash
pip install "diffusers>=0.29,<0.30"
```

### ImageReward fails with "No module named 'pkg_resources'"

Python 3.12 venvs omit `setuptools`. Install it pinned below the version that
removed `pkg_resources`:
```bash
pip install "setuptools<81"
```

### InstructBLIP slow on Mac

The model is ~10 GB on MPS (fp16). Expect ~60–90 seconds per match on an M-series
Mac. On a CUDA GPU with 4-bit quantization it loads in ~2 GB and runs much faster.

### CUDA not found after install on Linux

`pip install torch` installs the CPU build by default. Use the CUDA wheel:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
Or just run `bash setup_cuda.sh` which handles this automatically.

### Out of memory on CUDA

The two A6000s have 49 GB VRAM each. InstructBLIP at 4-bit uses ~2 GB.
If you still run out, switch to `quantization: "4bit"` in `config.yaml`
(it already is by default for CUDA). Check VRAM usage with:
```bash
watch -n 1 nvidia-smi
```

### Tournament is slow — check progress

Each match takes ~60 s on Mac MPS, ~5–10 s on GPU. For 50 images grouped into 5 sets of 10 (225 matches total):
- Mac MPS: ~3.7 hours (previously ~20 hours for 1225 matches)
- GPU (A6000): ~20–35 minutes (previously ~2–3 hours for 1225 matches)

Monitor live:
```bash
tail -f results/tournament.log | grep "progress\|match\|WARNING\|ERROR"
```

### Tests fail after a code change

```bash
python -m pytest tests/ -q --tb=short
```

The suite must always return **265 passed** before running a tournament.
