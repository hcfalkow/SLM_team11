# Sustainability Assessment of a Small Language Model (SLM)

This repository contains the codebase for the case study **“Assessing the Sustainability of Language Models”**, developed for the course **1210X Quantitative Methods to Assess Sustainability**.

In this project, you will train a **Small Language Model (SLM)** based on a GPT-style Transformer architecture and assess the sustainability impacts of both the **training phase** and the **prompting phase** across environmental, economic, and social dimensions.

The goal of the case study is **not to optimize model performance**, but to understand how computational design and usage choices translate into sustainability impacts.

---

## Repository structure

```text
.
├── data/
│   └── prepare.py        # Dataset preparation (Tiny Shakespeare, character-level)
│
├── src/
│   ├── model.py          # Model architecture
│   ├── train.py          # Training script
│   └── prompt.py         # Inference/ prompting script
│
└── README.md
```

---

## Sync the environment for this project
To create a new projet environment, in the project directory run:
```bash
uv sync --locked --dev
```
For running python scripts, you should use:
```bash
uv run script.py
```

## General workflow

1. Prepare the dataset  
2. Review predefined scenarios and invariants
3. Train the language model  
4. Run inference (prompting)  
5. Quantify sustainability impacts using structured outputs  

Each step is described below.

---

## 1. Dataset preparation (`data/prepare.py`)

This script prepares the **Tiny Shakespeare** dataset for character-level language modeling.

It performs the following steps:

- Downloads the dataset (if not already present)
- Builds a character-level vocabulary
- Splits the data into training and validation sets
- Generates:
  - `train.bin`
  - `val.bin`
  - `meta.pkl` (vocabulary and encoding metadata)

Run this script **once** before training:

```bash
uv run data/prepare.py
```

You are **not required to modify** this file.

---

## 2. Model architecture (`src/model.py`)

This file contains the **complete definition of the language model**.

- Architecture: **Transformer, decoder-only**
- Conceptually similar to GPT-family models (e.g. GPT‑2 / ChatGPT), but much smaller
- Includes:
  - Token and positional embeddings
  - Masked multi-head self-attention
  - Feed-forward (MLP) layers
  - Residual connections and Layer Normalization

You are **not required** to modify this file. However, students interested in model design are encouraged to experiment with more complex variants.

In all cases, you should:

- Inspect it to understand the model structure  
- Report the architecture and number of parameters in your sustainability assessment  

---

## 3. Training (`src/train.py`)

This script trains the Small Language Model from scratch.

Run this script using:
```bash
uv run src/train.py
```

The script supports scenario/CLI configuration and writes each execution to:

```text
out/runs/train/<scenario_id>/<run_id>/
```

### Tunable parameters

At the top of `train.py`, you will find a configuration section where you can adjust parameters such as:

- **Model size**
  - Number of layers
  - Embedding dimension
  - Number of attention heads
- **Training workload**
  - Batch size
  - Number of training iterations
- **Hardware**
  - CPU or GPU

These parameters are the **main levers** you should use for:

- Sensitivity analysis  
- Scenario comparison  
- Sustainability trade-off evaluation  

### What you should NOT change

- The training loop logic  
- The loss function  
- The data loading logic  

### Where to implement CodeCarbon (training)

CodeCarbon is integrated with stage-split tracking for the training lifecycle:

- TR1: setup and initialization
- TR2: core training compute
- TR3: periodic evaluation and control
- TR4: finalization and artifact write

Per-stage energy and emissions are logged separately and aggregated in run metadata.

- Energy consumption  
- CO₂-equivalent emissions during training  

### Validation threshold stopping (training)

A validation-loss threshold stop option is included and configurable:

- `--val-threshold-stopping` (true/false)
- `--val-loss-threshold`
- `--val-threshold-min-evals`

Training stops when validation loss is below the configured threshold after the minimum number of evaluations.
The stop reason and stopping iteration are logged in run metadata and summary tables.

---

## 4. Inference / Prompting (`src/prompt.py`)

Run this script using:
```bash
uv run src/prompt.py
```

This script performs **inference** using a trained model checkpoint.

It:

- Loads the trained model  
- Accepts either a single prompt or a fixed prompt workload file  
- Generates new tokens autoregressively  

### Tunable parameters

You may adjust:

- Prompt text  
- Number of generated tokens  
- Sampling parameters (e.g. temperature, top-k)  

These parameters control the **inference workload**, which is essential for:

- Comparing training vs usage impacts  
- Scaling impacts to realistic deployment scenarios  

### Where to implement CodeCarbon (inference)

CodeCarbon is integrated with stage-split tracking for the inference lifecycle:

- IN1: load and prepare
- IN2: input processing
- IN3: generation compute
- IN4: postprocess and persist

Per-stage energy and emissions are logged separately and aggregated in run metadata.

---

## Scenario-driven setup (small and predefined)

This repository now includes a lightweight scenario layer intended for narrow, fair comparisons.

- Training scenarios: `scenarios/training_scenarios.json`
- Inference scenarios: `scenarios/inference_scenarios.json`
- Fixed reusable inference workload: `scenarios/prompts.txt`
- Runner for one scenario or full sweep: `main.py`

### Comparison invariants

Unless explicitly varied by a scenario, these stay fixed and are logged:

- dataset and split
- evaluation procedure (`eval_interval`, `eval_iters`)
- prompt workload file for inference
- inference output length (`max_new_tokens`) for comparable inference scenarios
- software/hardware context metadata

### Run commands

List scenarios:

```bash
uv run main.py list --phase all
```

Run one training scenario:

```bash
uv run main.py run --phase train --scenario-id train_baseline
```

Run one inference scenario:

```bash
uv run main.py run --phase inference --scenario-id infer_baseline
```

Run full sweeps:

```bash
uv run main.py sweep --phase train --skip-if-complete
uv run main.py sweep --phase inference --skip-if-complete
```

Inference checkpoint resolution order:

- Use `--checkpoint-path` if provided and the file exists.
- Otherwise use `model_training_scenario_id` from `scenarios/inference_scenarios.json` and load
  checkpoint path from `out/runs/train/<model_training_scenario_id>/latest_run.json`.
- If the checkpoint cannot be resolved, the inference scenario is skipped with a warning.

Optional per-run overrides can still be passed while staying scenario-first:

```bash
uv run main.py run --phase train --scenario-id train_baseline --extra --max-iters 1200 --learning-rate 2e-4
```

### Outputs for downstream analysis

Each scenario run produces machine-readable artifacts:

- `effective_config.json`
- `run_metadata.json`
- `emissions_<stage_id>.csv` (one per lifecycle stage)
- `train_metrics.csv` (training only)
- `generated_outputs.jsonl` (inference only)
- `ckpt.pt` (training only)

Aggregate CSV tables are appended to:

- `out/scenario_summaries/train_summary.csv`
- `out/scenario_summaries/inference_summary.csv`
- `out/scenario_summaries/train_stage_summary.csv`
- `out/scenario_summaries/inference_stage_summary.csv`

These summaries combine configuration, runtime, workload size, and energy/emissions values.

---


## Learning objectives

After completing this case study, you should be able to:

- Apply life-cycle thinking to digital and computational systems  
- Understand the structure of modern language models  
- Quantify environmental impacts of training and inference  
- Perform sensitivity analysis on computational parameters  
- Reflect on trade-offs across environmental, economic, and social dimensions  
- Connect small-scale experiments to large-scale AI deployment  

---

## Important notes

- Model performance (text quality) is **not graded**  
- Transparency, assumptions, and reproducibility are essential  
- Clearly document all parameter choices and scenarios in your report  
- Focus on **sustainability insights**, not deep learning optimization  

If you have questions about the code structure or the scope of allowed modifications, refer to the assignment description or contact the course TAs.
