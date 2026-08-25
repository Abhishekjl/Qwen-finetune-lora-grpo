# Fine-Tuning Qwen3-4B with Unsloth & QLoRA

A complete pipeline for fine-tuning **Qwen3-4B** using **QLoRA** and **Unsloth** to create an expert AI engineering tutor (`ProTutor`). This project demonstrates efficient fine-tuning of smaller language models with synthetic data generation via Groq's API.

## 🎯 Project Overview

This notebook implements an end-to-end workflow for:

1. **Dataset Generation**: Uses Groq's fast LLM to generate 300+ high-quality Q&A pairs across 16 AI/ML topics
2. **Efficient Fine-Tuning**: Leverages Unsloth and QLoRA for memory-efficient model adaptation
3. **Inference & Evaluation**: Tests the fine-tuned model on diverse prompts with clear output examples

### Key Features

- ✅ **Memory Efficient**: QLoRA + Unsloth reduces VRAM requirements by ~60%
- ✅ **Fast Data Generation**: Async Groq API calls with rate-limit backoff
- ✅ **Production-Ready**: Clear loss curves and evaluation metrics
- ✅ **Reusable Pipeline**: Modular functions for dataset generation and training

## 📋 Prerequisites

### Hardware
- GPU with 8GB+ VRAM (tested on NVIDIA GPUs; T4/A100 recommended)
- Python 3.10+

### API Keys
- **Groq API Key**: For dataset generation ([get here](https://console.groq.com/))

### Environment
This notebook runs on **Kaggle** or local Jupyter environments with GPU support.

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
pip install unsloth
pip install --no-deps --upgrade "torchao>=0.16.0"
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
pip install groq
```

### 2. Set Up Groq API Key

**For Kaggle:**
1. Get your Groq API key from [console.groq.com](https://console.groq.com/)
2. Add it to Kaggle secrets as `GROQ_API_KEY`

**For Local Development:**
```bash
export GROQ_API_KEY="your-api-key-here"
```

## 📊 Workflow

### Step 1: Generate Dataset with Groq

The notebook generates Q&A pairs using `openai/gpt-oss-120b` (via Groq) to create diverse training examples:

```python
TOPICS = [
    "Python basics and data structures",
    "machine learning fundamentals",
    "transformers and attention",
    "fine-tuning and LoRA",
    # ... 16 total topics
]
```

**Output**: ~300 question-answer pairs split into train/held-out sets

**Example generated answer:**
```
Q: Explain what a Python decorator is.
A: A decorator is like a frame around a picture: it wraps the original 
picture (function) and adds extra features without changing the picture 
itself. In Python, a decorator takes another function as an argument, 
modifies it, and returns a new function. Key Takeaway: A decorator wraps 
a function to add behavior without altering its core logic.
```

### Step 2: Load Model & Configure QLoRA

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B",
    max_seq_length=2048,
    load_in_4bit=True,  # 4-bit quantization
    dtype=torch.float16,
)
```

### Step 3: Add LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
```

### Step 4: Train with TRL

```python
trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(
        num_train_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
    ),
    train_dataset=dataset["train"],
)
trainer.train()
```

### Step 5: Evaluate on Test Prompts

The notebook runs inference on diverse test queries:

```
✅ Math problems (arithmetic, logic puzzles)
✅ AI/ML concepts (decorators, complexity, LoRA)
✅ Trivia (geography, general knowledge)
✅ System design (scalable RAG architecture)
✅ Debugging tasks
```

## 📈 Results

### Training Metrics
- **Final Training Loss**: ~0.94 (converged after 60 steps)
- **Train/Eval Split**: 288/20 examples
- **Training Time**: ~1.5 hours on Kaggle GPU

### Model Outputs

| Prompt | Quality |
|--------|---------|
| "What is a Python decorator?" | ✅ Clear, analogy-based explanation with key takeaway |
| "Explain O(n²) complexity" | ✅ Librarian analogy, practical example |
| "Design scalable RAG for 10K users" | ✅ Detailed architecture with caching/sharding |

All responses follow the training format: **clear explanation → real-world analogy → key takeaway**.

## 🏗️ Project Structure

```
.
├── finetuning-qwen.ipynb          # Main notebook (all-in-one workflow)
├── desitutor_data.json            # Generated dataset (train/heldout)
├── README.md                       # This file
└── outputs/                        # Fine-tuned model checkpoint (GGUF format)
```

## 🔧 Configuration

### Hyperparameters (Tunable)

```python
# Dataset generation
TOPICS = [...]  # List of training topics
n_examples_per_topic = 20

# LoRA config
r = 16           # Rank (lower = faster, less capable)
lora_alpha = 16  # Scaling factor
lora_dropout = 0.05

# Training
num_train_epochs = 3
learning_rate = 2e-4
per_device_train_batch_size = 4  # Adjust based on VRAM
```

## 📝 API Rate Limits

The Groq free tier allows **8,000 tokens/minute**:
- ~1 API call per 45 seconds for ~20 Q&A pairs per topic
- The notebook batches 2 topics at a time with exponential backoff on rate limits

Upgrade to Groq Dev Tier for faster dataset generation.

## 🎓 How This Works

### Why Unsloth?
- **3x faster** training than standard transformers
- **60% less VRAM** via Unsloth's optimized kernels
- Drop-in replacement for Hugging Face models

### Why QLoRA?
- Fine-tune large models on consumer GPUs (8GB+)
- Adds ~4% additional parameters (16-rank adapters)
- Production models use merged weights (~4B parameters)

### Why Synthetic Data?
- Generates domain-specific training data without manual labeling
- Using a strong teacher model (Groq's 120B) ensures data quality
- Scales to hundreds of examples in minutes

## 🚦 Next Steps

1. **Push to Hub**: Upload fine-tuned model to Hugging Face Hub
   ```bash
   model.push_to_hub("your-username/ProTutor-Qwen3-4B")
   ```

2. **Deploy**: Export to GGUF/ONNX for edge inference
   ```python
   model.save_pretrained("./protutor-gguf")
   ```

3. **Evaluate**: Run BLEU/ROUGE scores against held-out set

4. **Extend**: Add more topics or fine-tune on custom data

## 📚 References

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-4B)
- [Groq API Docs](https://console.groq.com/docs)

## 📄 License

This project is open source and available under the **MIT License**.

## 🤝 Contributing

Contributions welcome! Feel free to:
- Improve dataset generation logic
- Add new training topics
- Optimize hyperparameters
- Extend to other base models (Llama, Mistral, etc.)

## 📧 Contact

For questions or issues, open a GitHub issue or reach out.

---

**Note**: This project was built on Kaggle. For local execution, ensure you have CUDA 12.1+ and sufficient GPU memory.
