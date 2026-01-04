# Dataset

This folder contains the TeleQnA dataset for experiments.

## Files

- **sampled_teleQnA_1k.txt**: Sample subset (1000 questions)
  - Used for quick testing and development
  - JSON format with questions and answers

- **TeleQnA.txt**: Full dataset
  - Complete telecommunications Q&A dataset
  - Same format as sample


## Usage

### Loading Data

```python
import json

# Load sample data
with open("data/sampled_teleQnA_1k.txt", encoding="utf-8") as f:
    questions = json.loads(f.read())

print(f"Loaded {len(questions)} questions")
```

### Accessing Questions

```python
for q_id, q_data in questions.items():
    question_text = q_data["question"]
    correct_answer = q_data["answer"]
    category = q_data["category"]
    
    # Access options
    options = [
        q_data["option 1"],
        q_data["option 2"],
        q_data["option 3"],
        q_data["option 4"]
    ]
```