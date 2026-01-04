# Conventional Edge-Cloud Deployment

This folder contains code for the conventional edge-cloud deployment scenario.

## Files

- **hugging_face_model.py**: Unified model interface
  - Supports both Qwen2-1.5B-Instruct (edge) and Qwen2-7B-Instruct (cloud)
  - Two confidence modes: logit-based and self-evaluation

- **evaluation_tool.py**: Evaluation functions
  - Accuracy computation using Rouge-L
  - Confidence extraction

- **run_experiment.py**: Main experiment script
  - Option permutation experiments
  - Configurable confidence mode

## Usage

### Basic Run

```bash
python run_experiment.py
```

This will:
1. Load sampled TeleQnA dataset
2. Run 5 permutation experiments
3. Save results to `./results_logit/` (or `./results_self/`)

### Configuration

Edit the configuration section at the top of `run_experiment.py`:

```python
# Confidence mode
USE_SELF_CONFIDENCE = False  # True for self-evaluation, False for logit-based

# Save directory (automatically set based on confidence mode)
SAVE_DIR = './results_self/' if USE_SELF_CONFIDENCE else './results_logit/'

# Model selection
model_name = "Qwen/Qwen2-1.5B-Instruct"  # Change to "Qwen/Qwen2-7B-Instruct" for cloud

# Number of permutations
K = 5
```

### Switching Models

**For Edge Model (Qwen2-1.5B):**
```python
model_name = "Qwen/Qwen2-1.5B-Instruct"
```

**For Cloud Model (Qwen2-7B):**
```python
model_name = "Qwen/Qwen2-7B-Instruct"
```

No other code changes needed!

### Switching Confidence Methods

**Logit-based Confidence (Default):**
- Set `USE_SELF_CONFIDENCE = False`
- Uses token-level logits to compute confidence
- Faster, no extra model calls

**Self-Evaluation Confidence:**
- Set `USE_SELF_CONFIDENCE = True`
- Asks model to evaluate its own answers
- More accurate but slower (2x inference time)


## Notes

- Ensure you have GPU with at least 16GB VRAM for Qwen2-1.5B
- For Qwen2-7B, recommend 40GB+ VRAM
- Adjust `max_memory` in `hugging_face_model.py` based on your GPU
