# Reasoning-Enhanced Cloud Deployment

This folder contains code for the reasoning-enhanced cloud deployment scenario using Qwen3-4B with thinking budget.

## Files

- **hugging_face_model.py**: Model interface with thinking budget allocation
  - Supports Qwen3-4B reasoning model
  - Budget allocation between answer and confidence evaluation

- **evaluation_tool.py**: Evaluation with budget scaling
  - Adapted for reasoning models

- **run_experiment.py**: Experiment script with thinking budget
  - Configurable budget allocation

## Usage

### Basic Run

```bash
python run_experiment.py
```

### Configuration

Edit the configuration section at the top of `run_experiment.py`:

```python
# Total thinking token budget
TOTAL_BUDGET = 200  # Adjust as needed (100-400)

# Ratio of budget for answer thinking
# 1.0 = all budget for answer, confidence without thinking
# 0.5 = half for answer, half for confidence
# 0.0 = all budget for confidence, answer without thinking
ANSWER_THINKING_RATIO = 0.5

# Save directory
SAVE_DIR = './results_reasoning/'

# Model
model_name = "Qwen/Qwen3-4B"
```

## Budget Allocation

The thinking budget is split between two stages:

### Stage 1: Answer Generation
- Budget: `TOTAL_BUDGET * ANSWER_THINKING_RATIO`
- Model thinks step-by-step before answering
- More budget = better reasoning

### Stage 2: Confidence Evaluation
- Budget: `TOTAL_BUDGET * (1 - ANSWER_THINKING_RATIO)`
- Model thinks step-by-step to assess answer correctness
- More budget = better confidence estimation

## Example Configurations

### All Budget for Answer
```python
TOTAL_BUDGET = 200
ANSWER_THINKING_RATIO = 1.0
# Result: 200 tokens for answer thinking, direct confidence evaluation
```

### Balanced
```python
TOTAL_BUDGET = 200
ANSWER_THINKING_RATIO = 0.5
# Result: 100 tokens for answer, 100 tokens for confidence
```

### All Budget for Confidence
```python
TOTAL_BUDGET = 200
ANSWER_THINKING_RATIO = 0.0
# Result: Direct answer, 200 tokens for confidence thinking
```

## Performance Notes

- Qwen3-4B requires ~20GB VRAM
- Thinking budget increases inference time proportionally
- Higher budgets generally improve accuracy but increase cost
- Recommended range: 100-400 tokens total budget

## Comparison with Conventional

The reasoning model provides:
- Better performance on complex questions
- Explainable reasoning process
- Adjustable compute budget
- Trade-off between cost and accuracy
