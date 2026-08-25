---
jupyter:
  kaggle:
    accelerator: none
    dockerImageVersionId: 28755
    isGpuEnabled: false
    isInternetEnabled: false
    language: python
    sourceType: notebook
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.13
  nbformat: 4
  nbformat_minor: 4
---

::: {.cell .code execution_count="1" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" execution="{\"iopub.execute_input\":\"2026-08-24T21:05:19.190715Z\",\"iopub.status.busy\":\"2026-08-24T21:05:19.190454Z\",\"iopub.status.idle\":\"2026-08-24T21:05:20.221879Z\",\"shell.execute_reply\":\"2026-08-24T21:05:20.221238Z\",\"shell.execute_reply.started\":\"2026-08-24T21:05:19.190693Z\"}" trusted="true"}
``` python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')
```
:::

::: {.cell .markdown}
Fine-Tuning Qwen3-4B with Unsloth & QLoRA
:::

::: {.cell .code execution_count="7" execution="{\"iopub.execute_input\":\"2026-08-24T22:00:51.067445Z\",\"iopub.status.busy\":\"2026-08-24T22:00:51.067065Z\",\"iopub.status.idle\":\"2026-08-24T22:01:25.314805Z\",\"shell.execute_reply\":\"2026-08-24T22:01:25.314022Z\",\"shell.execute_reply.started\":\"2026-08-24T22:00:51.067418Z\"}" trusted="true"}
``` python
%%capture
import os

!pip install pip3-autoremove
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
!pip install unsloth
!pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
!pip install groq
```
:::

::: {.cell .markdown execution="{\"iopub.execute_input\":\"2026-08-24T18:25:40.512804Z\",\"iopub.status.busy\":\"2026-08-24T18:25:40.512174Z\",\"iopub.status.idle\":\"2026-08-24T18:25:40.517412Z\",\"shell.execute_reply\":\"2026-08-24T18:25:40.516397Z\",\"shell.execute_reply.started\":\"2026-08-24T18:25:40.512755Z\"}"}
#### Generate the Dataset with Groq¶
:::

::: {.cell .code execution_count="3" execution="{\"iopub.execute_input\":\"2026-08-24T22:00:40.780343Z\",\"iopub.status.busy\":\"2026-08-24T22:00:40.779931Z\",\"iopub.status.idle\":\"2026-08-24T22:00:41.526142Z\",\"shell.execute_reply\":\"2026-08-24T22:00:41.525517Z\",\"shell.execute_reply.started\":\"2026-08-24T22:00:40.780313Z\"}" trusted="true"}
``` python
import asyncio, json, random
from groq import AsyncGroq
from kaggle_secrets import UserSecretsClient

client = AsyncGroq(api_key=UserSecretsClient().get_secret("GROQ_API_KEY"))

TOPICS = [
    "Python basics and data structures", "pandas and data cleaning",
    "machine learning fundamentals", "overfitting and regularization",
    "neural networks and backpropagation", "transformers and attention",
    "LLMs and how they are trained", "prompt engineering",
    "RAG and vector databases", "fine-tuning and LoRA",
    "AI agents and tool calling", "model evaluation and metrics",
    "APIs and deployment basics", "SQL and databases",
    "statistics for data science", "career advice for AI engineers",
]

SYSTEM_PROMPT = """You generate training data for 'ProTutor', an expert AI engineering tutor.
Each example is a question a beginner software developer would ask, plus ProTutor's answer.
Every answer MUST follow these 3 rules:
1. Written in clear, professional, and easy-to-understand standard English.
2. Explains the concept using a simple, universal real-world analogy (e.g., a library, a kitchen, a post office).
3. Ends with a clear one-line summary starting with 'Key Takeaway:'.
Keep answers 60-120 words. Vary question phrasing and difficulty.
Return JSON: {"examples": [{"user": "...", "assistant": "..."}, ...]}"""

async def generate_batch(topic: str, n: int = 20, retries: int = 3) -> list[dict]:
    """Ask the teacher model for n Q&A pairs about one topic. Backs off on rate limits. this will take topics one by one"""
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": f"Generate {n} question-answer pairs about: {topic}"}],
                response_format={"type": "json_object"},
                temperature=0.9,
            )
            return json.loads(resp.choices[0].message.content)["examples"]
        except Exception as e:
            print(f" retry {attempt + 1} for '{topic}': {e}")
            await asyncio.sleep(20 * (attempt + 1))
    return []




```
:::

::: {.cell .code trusted="true"}
``` python
```
:::

::: {.cell .code execution_count="5" execution="{\"iopub.execute_input\":\"2026-08-24T20:54:15.747784Z\",\"iopub.status.busy\":\"2026-08-24T20:54:15.747493Z\",\"iopub.status.idle\":\"2026-08-24T21:00:09.577747Z\",\"shell.execute_reply\":\"2026-08-24T21:00:09.576882Z\",\"shell.execute_reply.started\":\"2026-08-24T20:54:15.747758Z\"}" trusted="true"}
``` python

async def build_dataset() -> list[dict]:
    """Generate all topics, 2 at a time - free tier allows 8,000 tokens/min, one call uses ~3,300."""
    pairs = []
    for i in range(0, len(TOPICS), 2):
        results = await asyncio.gather(*[generate_batch(t) for t in TOPICS[i:i + 2]])
        for batch in results:
            pairs.extend(batch)
        print(f"{len(pairs)} examples generated")
    return pairs

pairs = await build_dataset()
random.shuffle(pairs)

train_pairs, heldout_pairs = pairs[:-20], pairs[-20:]

with open("/kaggle/working/desitutor_data.json", "w") as f:
    json.dump({"train": train_pairs, "heldout": heldout_pairs}, f, ensure_ascii=False, indent=2)
```

::: {.output .stream .stdout}
    43 examples generated
    86 examples generated
     retry 1 for 'neural networks and backpropagation': Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m0tgvedmenk8pvkgtpwpck30` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 7967, Requested 3266. Please try again in 24.2475s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
    131 examples generated
    176 examples generated
    217 examples generated
     retry 1 for 'model evaluation and metrics': Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m0tgvedmenk8pvkgtpwpck30` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 4867, Requested 3264. Please try again in 982.5ms. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
    259 examples generated
     retry 1 for 'APIs and deployment basics': Error code: 400 - {'error': {'message': "Failed to validate JSON. Please adjust your prompt. See 'failed_generation' for more details.", 'type': 'invalid_request_error', 'code': 'json_validate_failed', 'failed_generation': ''}}
    301 examples generated
     retry 1 for 'statistics for data science': Error code: 400 - {'error': {'message': "Failed to validate JSON. Please adjust your prompt. See 'failed_generation' for more details.", 'type': 'invalid_request_error', 'code': 'json_validate_failed', 'failed_generation': ''}}
    342 examples generated
:::
:::

::: {.cell .code execution_count="4" execution="{\"iopub.execute_input\":\"2026-08-24T22:00:41.527234Z\",\"iopub.status.busy\":\"2026-08-24T22:00:41.526984Z\",\"iopub.status.idle\":\"2026-08-24T22:00:41.541981Z\",\"shell.execute_reply\":\"2026-08-24T22:00:41.541196Z\",\"shell.execute_reply.started\":\"2026-08-24T22:00:41.527212Z\"}" trusted="true"}
``` python

file_path = "/kaggle/input/datasets/preatcher/desitutor-data/desitutor_data.json"

# 2. Open and load the JSON file
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. Extract the lists from the dictionary
train_pairs = data["train"]
heldout_pairs = data["heldout"]
```
:::

::: {.cell .code execution_count="5" execution="{\"iopub.execute_input\":\"2026-08-24T22:00:41.543745Z\",\"iopub.status.busy\":\"2026-08-24T22:00:41.543420Z\",\"iopub.status.idle\":\"2026-08-24T22:00:41.548659Z\",\"shell.execute_reply\":\"2026-08-24T22:00:41.547744Z\",\"shell.execute_reply.started\":\"2026-08-24T22:00:41.543723Z\"}" trusted="true"}
``` python
print(f"Train: {len(train_pairs)}, Held-out: {len(heldout_pairs)}")
print(json.dumps(train_pairs[0], ensure_ascii=False, indent=2))
```

::: {.output .stream .stdout}
    Train: 322, Held-out: 20
    {
      "user": "What is a stored procedure and when should I use it?",
      "assistant": "A stored procedure is like a pre‑written recipe stored in the kitchen pantry: you can call it whenever you need the same set of steps, without rewriting them each time. It’s a batch of SQL statements saved on the server, which can accept parameters, contain logic, and return results. Use it for repetitive tasks, complex business logic, or to centralize security. Key Takeaway: A stored procedure is a reusable, server‑side SQL script that encapsulates logic for consistent, efficient execution."
    }
:::
:::

::: {.cell .markdown}
### Load Qwen3-4B in 4-bit and Attach LoRA¶
:::

::: {.cell .code execution_count="8" execution="{\"iopub.execute_input\":\"2026-08-24T22:01:33.646144Z\",\"iopub.status.busy\":\"2026-08-24T22:01:33.645661Z\",\"iopub.status.idle\":\"2026-08-24T22:02:37.605718Z\",\"shell.execute_reply\":\"2026-08-24T22:02:37.604743Z\",\"shell.execute_reply.started\":\"2026-08-24T22:01:33.646109Z\"}" trusted="true"}
``` python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)
```

::: {.output .stream .stdout}
    🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
    🦥 Unsloth Zoo will now patch everything to make training faster!
    ==((====))==  Unsloth 2026.8.19: Fast Qwen3 patching. Transformers: 4.56.2.
       \\   /|    Tesla T4. Num GPUs = 2. Max memory: 14.562 GB. Platform: Linux.
    O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 7.5. CUDA Toolkit: 12.8. Triton: 3.6.0
    \        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
     "-____-"     Free license: http://github.com/unslothai/unsloth
    Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
:::
:::

::: {.cell .code execution_count="9" execution="{\"iopub.execute_input\":\"2026-08-24T22:02:37.607917Z\",\"iopub.status.busy\":\"2026-08-24T22:02:37.607429Z\",\"iopub.status.idle\":\"2026-08-24T22:02:37.620549Z\",\"shell.execute_reply\":\"2026-08-24T22:02:37.619805Z\",\"shell.execute_reply.started\":\"2026-08-24T22:02:37.607866Z\"}" trusted="true"}
``` python
from unsloth import FastLanguageModel

FastLanguageModel.for_inference(model)
```

::: {.output .execute_result execution_count="9"}
    Qwen3ForCausalLM(
      (model): Qwen3Model(
        (embed_tokens): Embedding(151936, 2560, padding_idx=151654)
        (layers): ModuleList(
          (0): Qwen3DecoderLayer(
            (self_attn): Qwen3Attention(
              (q_proj): Linear(in_features=2560, out_features=4096, bias=False)
              (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
              (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
              (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
              (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): Qwen3MLP(
              (gate_proj): Linear4bit(in_features=2560, out_features=9728, bias=False)
              (up_proj): Linear4bit(in_features=2560, out_features=9728, bias=False)
              (down_proj): Linear4bit(in_features=9728, out_features=2560, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
            (post_attention_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
          )
          (1-2): 2 x Qwen3DecoderLayer(
            (self_attn): Qwen3Attention(
              (q_proj): Linear4bit(in_features=2560, out_features=4096, bias=False)
              (k_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
              (v_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
              (o_proj): Linear4bit(in_features=4096, out_features=2560, bias=False)
              (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): Qwen3MLP(
              (gate_proj): Linear(in_features=2560, out_features=9728, bias=False)
              (up_proj): Linear(in_features=2560, out_features=9728, bias=False)
              (down_proj): Linear(in_features=9728, out_features=2560, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
            (post_attention_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
          )
          (3): Qwen3DecoderLayer(
            (self_attn): Qwen3Attention(
              (q_proj): Linear(in_features=2560, out_features=4096, bias=False)
              (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
              (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
              (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
              (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): Qwen3MLP(
              (gate_proj): Linear(in_features=2560, out_features=9728, bias=False)
              (up_proj): Linear(in_features=2560, out_features=9728, bias=False)
              (down_proj): Linear(in_features=9728, out_features=2560, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
            (post_attention_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
          )
          (4): Qwen3DecoderLayer(
            (self_attn): Qwen3Attention(
              (q_proj): Linear4bit(in_features=2560, out_features=4096, bias=False)
              (k_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
              (v_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
              (o_proj): Linear4bit(in_features=4096, out_features=2560, bias=False)
              (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): Qwen3MLP(
              (gate_proj): Linear4bit(in_features=2560, out_features=9728, bias=False)
              (up_proj): Linear4bit(in_features=2560, out_features=9728, bias=False)
              (down_proj): Linear4bit(in_features=9728, out_features=2560, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
            (post_attention_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
          )
          (5): Qwen3DecoderLayer(
            (self_attn): Qwen3Attention(
              (q_proj): Linear4bit(in_features=2560, out_features=4096, bias=False)
              (k_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
              (v_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
              (o_proj): Linear4bit(in_features=4096, out_features=2560, bias=False)
              (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): Qwen3MLP(
              (gate_proj): Linear(in_features=2560, out_features=9728, bias=False)
              (up_proj): Linear(in_features=2560, out_features=9728, bias=False)
              (down_proj): Linear(in_features=9728, out_features=2560, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
            (post_attention_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
          )
          (6): Qwen3DecoderLayer(
            (self_attn): Qwen3Attention(
              (q_proj): Linear(in_features=2560, out_features=4096, bias=False)
              (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
              (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
              (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
              (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): Qwen3MLP(
              (gate_proj): Linear(in_features=2560, out_features=9728, bias=False)
              (up_proj): Linear(in_features=2560, out_features=9728, bias=False)
              (down_proj): Linear(in_features=9728, out_features=2560, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
            (post_attention_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
          )
          (7-33): 27 x Qwen3DecoderLayer(
            (self_attn): Qwen3Attention(
              (q_proj): Linear4bit(in_features=2560, out_features=4096, bias=False)
              (k_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
              (v_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
              (o_proj): Linear4bit(in_features=4096, out_features=2560, bias=False)
              (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): Qwen3MLP(
              (gate_proj): Linear4bit(in_features=2560, out_features=9728, bias=False)
              (up_proj): Linear4bit(in_features=2560, out_features=9728, bias=False)
              (down_proj): Linear4bit(in_features=9728, out_features=2560, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
            (post_attention_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
          )
          (34-35): 2 x Qwen3DecoderLayer(
            (self_attn): Qwen3Attention(
              (q_proj): Linear4bit(in_features=2560, out_features=4096, bias=False)
              (k_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
              (v_proj): Linear4bit(in_features=2560, out_features=1024, bias=False)
              (o_proj): Linear4bit(in_features=4096, out_features=2560, bias=False)
              (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): Qwen3MLP(
              (gate_proj): Linear(in_features=2560, out_features=9728, bias=False)
              (up_proj): Linear(in_features=2560, out_features=9728, bias=False)
              (down_proj): Linear(in_features=9728, out_features=2560, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
            (post_attention_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
          )
        )
        (norm): Qwen3RMSNorm((2560,), eps=1e-06)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (lm_head): Linear(in_features=2560, out_features=151936, bias=False)
    )
:::
:::

::: {.cell .code execution_count="11" execution="{\"iopub.execute_input\":\"2026-08-24T22:07:48.475540Z\",\"iopub.status.busy\":\"2026-08-24T22:07:48.475007Z\",\"iopub.status.idle\":\"2026-08-24T22:10:57.776967Z\",\"shell.execute_reply\":\"2026-08-24T22:10:57.776182Z\",\"shell.execute_reply.started\":\"2026-08-24T22:07:48.475449Z\"}" trusted="true"}
``` python
test_prompts = [
    "What is the capital of France?",

    "What is 27 * 43?",

    "Explain what a Python decorator is.",

    "Why does this Python code have O(n²) complexity?",

    "Find the bug in this code and explain how to fix it.",

    "A train travels 60 km in 45 minutes. What is its average speed?",

    "You have 12 balls and one is heavier. Find the heavier ball using a balance scale in 3 weighings.",

    "Design a scalable RAG architecture capable of handling 10,000 concurrent users.",

    "Why might a distributed system return stale data intermittently?",

    "Explain the difference between LoRA and full fine-tuning."
]
baseline_responses = []

for prompt in test_prompts:

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )

    baseline_responses.append({
        "prompt": prompt,
        "response": response,
    })

    print("=" * 80)
    print("PROMPT:")
    print(prompt)
    print("\nRESPONSE:")
    print(response)
```

::: {.output .stream .stdout}
    ================================================================================
    PROMPT:
    What is the capital of France?

    RESPONSE:
    The capital of France is Paris.
    ================================================================================
    PROMPT:
    What is 27 * 43?

    RESPONSE:
    Let's calculate $ 27 \times 43 $ step by step.

    We can use the distributive property:

    $$
    27 \times 43 = 27 \times (40 + 3) = (27 \times 40) + (27 \times 3)
    $$

    Now compute each part:

    - $ 27 \times 40 = 27 \times 4 \times 10 = 108 \times 10 = 1080 $
    - $ 27 \times 3 = 81 $

    Now add them:

    $$
    1080 + 81 = 1161
    $$

    ✅ So, $ 27 \times 43 = \boxed{1161} $
    ================================================================================
    PROMPT:
    Explain what a Python decorator is.

    RESPONSE:
    A **Python decorator** is a special function that allows you to modify or enhance the behavior of another function without directly changing its code. It works by wrapping a function and adding extra functionality—like logging, timing, authentication, or caching—before or after the original function executes.

    ### How Decorators Work

    Decorators are implemented using **higher-order functions**—functions that take other functions as arguments and can return functions.

    Here’s a simple example:

    ```python
    def my_decorator(func):
        def wrapper():
            print("Something before the function executes")
            func()
            print("Something after the function executes")
        return wrapper

    @my_decorator
    def say_hello():
        print("Hello!")

    say_hello()
    ```

    **Output:**
    ```
    Something before the function executes
    Hello!
    Something after the function executes
    ```

    ### Key Points:

    - The `@my_decorator` syntax is a **syntactic sugar** for `say_hello = my_decorator(say_hello)`.
    - The `my_decorator` function takes `say_hello` as an argument and returns a new function (`wrapper`) that wraps the original function.
    - The original function (`say_hello`) is not modified directly—it's wrapped and extended.

    ### Common Use Cases

    1. **Logging** – Log when a function is called.
    2. **Timing** – Measure how long a function takes to execute.
    3. **Caching** – Store results of expensive function calls.
    4. **Authentication** – Check if a user is logged in before executing a function.
    5. **Error Handling** – Wrap functions to catch and handle exceptions gracefully.

    ### Example: Timing Decorator

    ```python
    import time

    def timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} took {end - start:.4f} seconds")
            return result
        return wrapper

    @timer
    def slow_function():
        time.sleep(1)
        return "Done!"

    slow_function()
    ```

    **Output:**
    ```
    slow_function took 1.0001 seconds
    Done!
    ```

    ### Summary

    A Python decorator is a powerful and elegant way to **add cross-cutting concerns** (like logging, timing, or security) to functions in a clean, reusable, and readable way. It promotes **separation of concerns** and makes code more modular and maintainable.

    💡 Think of it as
    ================================================================================
    PROMPT:
    Why does this Python code have O(n²) complexity?

    RESPONSE:
    I'd be happy to help you understand why a Python code snippet might have O(n²) complexity — but I don't see the actual code in your message!

    To give you a precise explanation, I'll need to see the specific Python code you're referring to. Without it, I can only provide general examples of common O(n²) patterns.

    **Common causes of O(n²) complexity in Python include:**

    1. **Nested loops** - Two nested `for` loops where each iteration depends on the outer loop
       ```python
       for i in range(n):
           for j in range(i, n):
               # O(n²) due to nested iteration
       ```

    2. **Nested list comprehensions or dictionary lookups**
       ```python
       for item in list1:
           for item2 in list2:
               # O(n²) if both lists have n elements
       ```

    3. **Brute-force search algorithms** like finding pairs or checking all combinations
       ```python
       for i in range(n):
           for j in range(i+1, n):
               # O(n²) for checking all pairs
       ```

    4. **Unoptimized sorting or searching** with nested operations

    **Please share your specific code**, and I'll:
    - Identify exactly where the O(n²) complexity arises
    - Explain the algorithmic reasoning
    - Suggest optimizations if possible

    For example, if you show something like:
    ```python
    def has_duplicate(nums):
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] == nums[j]:
                    return True
        return False
    ```
    I can explain why this is O(n²) and how to fix it with a hash set (O(n)).

    Just paste your code, and I'll give you a detailed breakdown! 🚀
    ================================================================================
    PROMPT:
    Find the bug in this code and explain how to fix it.

    RESPONSE:
    I'd be happy to help you find a bug in your code and explain how to fix it! However, I don't see any code posted in your message.

    Could you please share the code you'd like me to review? Once you provide it, I'll:

    1. Analyze the code for potential bugs (logical errors, syntax issues, runtime errors, performance problems, etc.)
    2. Explain exactly what the bug is and why it occurs
    3. Provide a clear, step-by-step explanation of how to fix it
    4. Suggest any improvements or best practices if applicable

    Just paste the code snippet, and I'll be happy to help you debug and improve it! 🐞➡️✅
    ================================================================================
    PROMPT:
    A train travels 60 km in 45 minutes. What is its average speed?

    RESPONSE:
    To find the **average speed** of the train, use the formula:

    $$
    \text{Average Speed} = \frac{\text{Total Distance}}{\text{Total Time}}
    $$

    ### Given:
    - Distance = 60 km  
    - Time = 45 minutes

    But speed is usually expressed in **km per hour (km/h)**, so we need to convert 45 minutes into hours.

    $$
    45 \text{ minutes} = \frac{45}{60} \text{ hours} = 0.75 \text{ hours}
    $$

    Now calculate the average speed:

    $$
    \text{Average Speed} = \frac{60 \text{ km}}{0.75 \text{ hours}} = 80 \text{ km/h}
    $$

    ### ✅ Answer:
    $$
    \boxed{80} \text{ km/h}
    $$
    ================================================================================
    PROMPT:
    You have 12 balls and one is heavier. Find the heavier ball using a balance scale in 3 weighings.

    RESPONSE:
    Yes! This is a classic logic puzzle. You have **12 balls**, one of which is **heavier** than the others, and you need to **identify the heavier ball using a balance scale in exactly 3 weighings**.

    Let’s go step-by-step to solve it.

    ---

    ### Goal:
    Find the one heavier ball among 12 balls using a balance scale in **3 weighings**.

    ---

    ### Strategy:
    We use a **divide-and-conquer** strategy with information theory.

    Each weighing on a balance scale has **3 possible outcomes**:
    - Left side heavier
    - Right side heavier
    - Balanced

    So with 3 weighings, we can distinguish among up to $3^3 = 27$ different outcomes.

    Since there are 12 balls and only one is heavier, there are 12 possible "heavier" cases — and 12 < 27 — so it's **theoretically possible**.

    We need to design a strategy that identifies the heavier ball.

    ---

    ## Step-by-Step Solution:

    We label the balls from 1 to 12.

    We will perform 3 weighings, each time comparing groups of balls.

    We need to assign each ball to a specific role in the weighings so that based on the results (left heavy, right heavy, balanced), we can uniquely identify the heavier ball.

    ---

    ### Weighing 1: Compare (1, 2, 3, 4) vs (5, 6, 7, 8)

    **Case A**: Left side heavier → one of 1, 2, 3, 4 is heavier  
    **Case B**: Right side heavier → one of 5, 6, 7, 8 is heavier  
    **Case C**: Balanced → the heavier ball is in {9, 10, 11, 12}

    ---

    ### Case A: Left side heavier (1–4 heavier)

    Now we know the heavier ball is among 1, 2, 3, 4.

    We now do **Weighing 2**: Compare (1, 2, 5) vs (3, 6, 9)

    Why this? We want to isolate the heavier one.

    Let’s analyze:

    - 5, 6, 9 are from the other group or known normal balls.
    - 9 is from the group that was balanced — so it's normal.

    So if the scale tips:
    - If left
    ================================================================================
    PROMPT:
    Design a scalable RAG architecture capable of handling 10,000 concurrent users.

    RESPONSE:
    Designing a **scalable Retrieval-Augmented Generation (RAG) architecture** capable of handling **10,000 concurrent users** requires a layered, distributed, and optimized system that balances performance, latency, reliability, and cost. Below is a comprehensive, production-grade RAG architecture designed specifically for high concurrency and scalability.

    ---

    ## 🚀 Scalable RAG Architecture for 10,000 Concurrent Users

    ---

    ### 🔍 Key Objectives
    - Support **10,000 concurrent users** with **sub-500ms latency** (target: <300ms for 95% of requests).
    - Ensure **high availability (99.99%)** and **fault tolerance**.
    - Scale horizontally across regions and data centers.
    - Maintain **low cost** and **efficient resource utilization**.
    - Support **real-time updates**, **search relevance**, and **secure access control**.

    ---

    ## 🏗️ System Architecture Overview

    ```
    +-------------------------+
    |     User App (Frontend) |
    +----------+--------------+
               |
               v
    +-------------------------+
    |   API Gateway (Load Balancer) |
    +----------+--------------+
               |
               v
    +-------------------------+
    |   Request Routing & Auth |
    |   (JWT, OAuth2, RBAC)    |
    +----------+--------------+
               |
               v
    +-------------------------+
    |   Request Dispatcher     |
    |   (Routing to Services)  |
    +----------+--------------+
               |
               v
    +-------------------------+       +-------------------------+
    |   Query Routing Layer    |<----->|   Vector Indexing Layer |
    |   (Sharding, Caching)    |       |   (FAISS, Pinecone, Weaviate) |
    +----------+--------------+       +-------------------------+
               |                           |
               v                           v
    +-------------------------+   +-------------------------+
    |   Embedding Generation  |   |   Retrieval Service (Search) |
    |   (LLM + Embedding Model)|   |   (Multi-tenant, distributed) |
    +----------+--------------+   +-------------------------+
               |                           |
               v                           v
    +-------------------------+   +-------------------------+
    |   RAG Generation Engine |   |   Response Aggregation & |
    |   (LLM + Prompting)     |   |   Caching (Redis, Memcached) |
    +----------+--------------+   +-------------------------+
               |
    ================================================================================
    PROMPT:
    Why might a distributed system return stale data intermittently?

    RESPONSE:
    A distributed system might return stale data intermittently due to several interrelated factors related to how data is synchronized, accessed, and cached across nodes. Here are the most common and important reasons:

    ### 1. **Network Latency and Partitioning**
    - In a distributed system, nodes may communicate over a network. If there's high latency or network partitioning, a client may receive a response from a node that hasn't yet received the latest update.
    - During a network delay or failure, a node may serve data from a previous version of the state, leading to stale responses.

    ### 2. **Cache Invalidation and Expiration**
    - Many systems use caching to improve performance. If cache entries are not invalidated or expired properly when data changes, clients may continue to receive old cached values.
    - Cache expiration policies (e.g., TTL – Time To Live) may be too long or not dynamically adjusted, causing stale data to persist even after updates.

    ### 3. **Eventual Consistency Models**
    - Distributed systems often use eventual consistency to maintain availability and partition tolerance (as per CAP theorem).
    - In such models, updates are propagated asynchronously across nodes. Clients may read data from a node that hasn’t yet received the update, resulting in stale data.
    - The system may not guarantee immediate consistency, so clients may see outdated values until the update propagates.

    ### 4. **Replication Lag**
    - Data is often replicated across multiple nodes for availability and fault tolerance.
    - Replication may not be instantaneous; there can be a delay between when a write occurs and when it’s reflected on all replicas.
    - A client reading from a replica may get data from a node that is still in the process of syncing.

    ### 5. **Read-Only Replicas and Load Balancing**
    - Read replicas are often used to offload read traffic. If the load balancer routes a request to a replica that hasn’t been updated, stale data is returned.
    - If replicas are not synchronized in real time, or if the system doesn’t use a mechanism like vector clocks or causal consistency, stale reads can occur.

    ### 6. **Lack of Strong Consistency Guarantees**
    - If the system doesn’t use strong consistency (e.g., via two-phase commit or consensus protocols like Paxos or Raft), clients may see outdated data.
    - Without proper coordination, updates may not be globally visible immediately.

    ### 7. **Client-Side Caching**
    - Clients may cache responses locally. If the client doesn’t refresh
    ================================================================================
    PROMPT:
    Explain the difference between LoRA and full fine-tuning.

    RESPONSE:
    Certainly! **LoRA (Low-Rank Adaptation)** and **full fine-tuning** are two different approaches used to adapt large language models (LLMs) to specific tasks or domains. While both aim to improve model performance without requiring massive computational resources, they differ significantly in their methodology, efficiency, and resource requirements.

    ---

    ### 🔍 Full Fine-Tuning

    **What it is:**
    Full fine-tuning involves updating *all* the model parameters (weights) in the pre-trained language model during training.

    **How it works:**
    - The model is trained on a dataset relevant to the target task (e.g., classification, summarization, dialogue).
    - All layers of the model (from embeddings to output layers) have their weights updated.
    - The optimizer adjusts every parameter in the model to minimize the loss function.

    **Pros:**
    - Can achieve high performance, especially on complex or nuanced tasks.
    - Often leads to better generalization when the task is well-matched to the model's capabilities.

    **Cons:**
    - **Computationally expensive**: Requires significant GPU/TPU memory and time.
    - **High cost**: Needs large amounts of data and powerful hardware.
    - **Risk of overfitting** or damaging the original model's general knowledge due to extensive parameter updates.
    - **Slow to train** and not scalable for small datasets.

    > ✅ Best for: High-performance tasks with large datasets and access to expensive hardware.

    ---

    ### 🚀 LoRA (Low-Rank Adaptation)

    **What it is:**
    LoRA is a parameter-efficient fine-tuning (PEFT) technique that only updates a small portion of the model’s parameters — specifically, the weights in the model’s layers — by introducing low-rank matrices.

    **How it works:**
    - Instead of updating all weights, LoRA modifies only a small number of parameters by adding a low-rank matrix to the original weight matrix.
    - For example, if a layer has a weight matrix of size \( W \), LoRA approximates the update as:
      \[
      W_{\text{new}} = W + \Delta W
      \]
      where \( \Delta W \) is a low-rank matrix (e.g., \( \Delta W = A \cdot B \), with \( A \) and \( B \) being small matrices).
    - Only the matrices \( A \) and \( B \) are trained (and updated), while the original weights remain largely unchanged.

    **Pros:**
    - **
:::
:::

::: {.cell .code execution_count="12" execution="{\"iopub.execute_input\":\"2026-08-24T22:10:57.778521Z\",\"iopub.status.busy\":\"2026-08-24T22:10:57.778205Z\",\"iopub.status.idle\":\"2026-08-24T22:11:03.000224Z\",\"shell.execute_reply\":\"2026-08-24T22:11:02.999558Z\",\"shell.execute_reply.started\":\"2026-08-24T22:10:57.778437Z\"}" trusted="true"}
``` python

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```

::: {.output .stream .stderr}
    Unsloth 2026.8.19 patched 36 layers with 36 QKV layers, 36 O layers and 36 MLP layers.
:::
:::

::: {.cell .markdown}
### Format the Data with the Chat Template¶
:::

::: {.cell .code execution_count="36" execution="{\"iopub.execute_input\":\"2026-08-24T22:17:47.603990Z\",\"iopub.status.busy\":\"2026-08-24T22:17:47.603181Z\",\"iopub.status.idle\":\"2026-08-24T22:17:47.610045Z\",\"shell.execute_reply\":\"2026-08-24T22:17:47.609377Z\",\"shell.execute_reply.started\":\"2026-08-24T22:17:47.603944Z\"}" trusted="true"}
``` python
from unsloth.chat_templates import get_chat_template
from datasets import Dataset

tokenizer = get_chat_template(tokenizer, chat_template = "qwen3-instruct")

def to_conversation(pair: dict) -> dict:
    """Wrap one Q&A pair in the role/content structure chat templates expect."""
    return {"conversations": [
        {"role": "user", "content": pair["user"]},
        {"role": "assistant", "content": pair["assistant"]},
    ]}

def formatting_prompts_func(examples: dict) -> dict:
    """Serialize each conversation into one ChatML training string."""
    texts = [tokenizer.apply_chat_template(c, tokenize = False, add_generation_prompt = False) 
             for c in examples["conversations"]]
    return {"text": texts}

```
:::

::: {.cell .code execution_count="37" execution="{\"iopub.execute_input\":\"2026-08-24T22:17:48.359684Z\",\"iopub.status.busy\":\"2026-08-24T22:17:48.359229Z\",\"iopub.status.idle\":\"2026-08-24T22:17:48.372813Z\",\"shell.execute_reply\":\"2026-08-24T22:17:48.372003Z\",\"shell.execute_reply.started\":\"2026-08-24T22:17:48.359640Z\"}" trusted="true"}
``` python
dataset = Dataset.from_list([to_conversation(p) for p in train_pairs])
```
:::

::: {.cell .code execution_count="38" execution="{\"iopub.execute_input\":\"2026-08-24T22:17:50.502264Z\",\"iopub.status.busy\":\"2026-08-24T22:17:50.501819Z\",\"iopub.status.idle\":\"2026-08-24T22:17:50.507793Z\",\"shell.execute_reply\":\"2026-08-24T22:17:50.506972Z\",\"shell.execute_reply.started\":\"2026-08-24T22:17:50.502220Z\"}" trusted="true"}
``` python
print(dataset[0])
```

::: {.output .stream .stdout}
    {'conversations': [{'role': 'user', 'content': 'What is a stored procedure and when should I use it?'}, {'role': 'assistant', 'content': 'A stored procedure is like a pre‑written recipe stored in the kitchen pantry: you can call it whenever you need the same set of steps, without rewriting them each time. It’s a batch of SQL statements saved on the server, which can accept parameters, contain logic, and return results. Use it for repetitive tasks, complex business logic, or to centralize security. Key Takeaway: A stored procedure is a reusable, server‑side SQL script that encapsulates logic for consistent, efficient execution.'}]}
:::
:::

::: {.cell .code execution_count="39" execution="{\"iopub.execute_input\":\"2026-08-24T22:17:51.012475Z\",\"iopub.status.busy\":\"2026-08-24T22:17:51.011927Z\",\"iopub.status.idle\":\"2026-08-24T22:17:51.182894Z\",\"shell.execute_reply\":\"2026-08-24T22:17:51.182187Z\",\"shell.execute_reply.started\":\"2026-08-24T22:17:51.012417Z\"}" trusted="true"}
``` python
dataset = dataset.map(formatting_prompts_func, batched = True)
print(dataset[0]["text"])
```

::: {.output .display_data}
``` json
{"model_id":"1f5d97b63ae84d44979cae007b832399","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    <|im_start|>user
    What is a stored procedure and when should I use it?<|im_end|>
    <|im_start|>assistant
    A stored procedure is like a pre‑written recipe stored in the kitchen pantry: you can call it whenever you need the same set of steps, without rewriting them each time. It’s a batch of SQL statements saved on the server, which can accept parameters, contain logic, and return results. Use it for repetitive tasks, complex business logic, or to centralize security. Key Takeaway: A stored procedure is a reusable, server‑side SQL script that encapsulates logic for consistent, efficient execution.<|im_end|>
:::
:::

::: {.cell .code execution_count="40" execution="{\"iopub.execute_input\":\"2026-08-24T22:18:25.124244Z\",\"iopub.status.busy\":\"2026-08-24T22:18:25.123649Z\",\"iopub.status.idle\":\"2026-08-24T22:23:29.782308Z\",\"shell.execute_reply\":\"2026-08-24T22:23:29.781287Z\",\"shell.execute_reply.started\":\"2026-08-24T22:18:25.124202Z\"}" trusted="true"}
``` python
from transformers import TextStreamer

def ask(question: str, max_new_tokens: int = 200) -> str:
    """Generate one answer from the model in its current state."""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    inputs = tokenizer(text, return_tensors = "pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens = max_new_tokens, 
                         temperature = 0.7, top_p = 0.8, top_k = 20)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True)

heldout_questions = [p["user"] for p in heldout_pairs]
base_answers = [ask(q) for q in heldout_questions]

print("Q:", heldout_questions[0])
print("BASE MODEL:", base_answers[0][:400])
```

::: {.output .stream .stdout}
    Q: What does the SELECT statement do in SQL?
    BASE MODEL: The **SELECT statement** in SQL is used to **retrieve data** from one or more tables in a database.

    ### Key Functions of the SELECT Statement:
    - **Queries data**: It allows you to specify which columns (or expressions) you want to retrieve.
    - **Filters data**: Using clauses like `WHERE`, you can filter rows based on certain conditions.
    - **Sorts data**: Using `ORDER BY`, you can sort the results 
:::
:::

::: {.cell .code execution_count="41" execution="{\"iopub.execute_input\":\"2026-08-24T22:24:05.788167Z\",\"iopub.status.busy\":\"2026-08-24T22:24:05.787525Z\",\"iopub.status.idle\":\"2026-08-24T22:27:15.556907Z\",\"shell.execute_reply\":\"2026-08-24T22:27:15.556165Z\",\"shell.execute_reply.started\":\"2026-08-24T22:24:05.788121Z\"}" trusted="true"}
``` python
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

stats = trainer.train()
```

::: {.output .display_data}
``` json
{"model_id":"fe1b54abdf2847c78a3e62dbe48206f2","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stdout}
    🦥 Unsloth: Padding-free auto-enabled, enabling faster training.
:::

::: {.output .display_data}
``` json
{"model_id":"309a538a88fb4b95a0291216b3764487","version_major":2,"version_minor":0}
```
:::

::: {.output .stream .stderr}
    ==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
       \\   /|    Num examples = 322 | Num Epochs = 2 | Total steps = 60
    O^O/ \_/ \    Batch size per device = 2 | Gradient accumulation steps = 4
    \        /    Data Parallel GPUs = 1 | Total batch size (2 x 4 x 1) = 8
     "-____-"     Trainable parameters = 66,060,288 of 4,088,528,384 (1.62% trained)
:::

::: {.output .display_data}
```{=html}

    <div>
      
      <progress value='60' max='60' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [60/60 02:56, Epoch 1/2]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>2.880400</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.853600</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2.904600</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.479800</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2.187100</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2.140800</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2.019400</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.878000</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.750400</td>
    </tr>
    <tr>
      <td>10</td>
      <td>1.641100</td>
    </tr>
    <tr>
      <td>11</td>
      <td>1.765700</td>
    </tr>
    <tr>
      <td>12</td>
      <td>1.474200</td>
    </tr>
    <tr>
      <td>13</td>
      <td>1.684600</td>
    </tr>
    <tr>
      <td>14</td>
      <td>1.316400</td>
    </tr>
    <tr>
      <td>15</td>
      <td>1.591500</td>
    </tr>
    <tr>
      <td>16</td>
      <td>1.441300</td>
    </tr>
    <tr>
      <td>17</td>
      <td>1.424200</td>
    </tr>
    <tr>
      <td>18</td>
      <td>1.356100</td>
    </tr>
    <tr>
      <td>19</td>
      <td>1.339600</td>
    </tr>
    <tr>
      <td>20</td>
      <td>1.345800</td>
    </tr>
    <tr>
      <td>21</td>
      <td>1.356200</td>
    </tr>
    <tr>
      <td>22</td>
      <td>1.277400</td>
    </tr>
    <tr>
      <td>23</td>
      <td>1.343100</td>
    </tr>
    <tr>
      <td>24</td>
      <td>1.202400</td>
    </tr>
    <tr>
      <td>25</td>
      <td>1.350300</td>
    </tr>
    <tr>
      <td>26</td>
      <td>1.331400</td>
    </tr>
    <tr>
      <td>27</td>
      <td>1.246700</td>
    </tr>
    <tr>
      <td>28</td>
      <td>1.101600</td>
    </tr>
    <tr>
      <td>29</td>
      <td>1.418100</td>
    </tr>
    <tr>
      <td>30</td>
      <td>1.221800</td>
    </tr>
    <tr>
      <td>31</td>
      <td>1.245100</td>
    </tr>
    <tr>
      <td>32</td>
      <td>1.238600</td>
    </tr>
    <tr>
      <td>33</td>
      <td>1.244300</td>
    </tr>
    <tr>
      <td>34</td>
      <td>1.149700</td>
    </tr>
    <tr>
      <td>35</td>
      <td>1.204600</td>
    </tr>
    <tr>
      <td>36</td>
      <td>1.286900</td>
    </tr>
    <tr>
      <td>37</td>
      <td>1.245900</td>
    </tr>
    <tr>
      <td>38</td>
      <td>1.221800</td>
    </tr>
    <tr>
      <td>39</td>
      <td>1.209600</td>
    </tr>
    <tr>
      <td>40</td>
      <td>1.189900</td>
    </tr>
    <tr>
      <td>41</td>
      <td>1.245900</td>
    </tr>
    <tr>
      <td>42</td>
      <td>1.017900</td>
    </tr>
    <tr>
      <td>43</td>
      <td>1.036200</td>
    </tr>
    <tr>
      <td>44</td>
      <td>1.059800</td>
    </tr>
    <tr>
      <td>45</td>
      <td>1.021700</td>
    </tr>
    <tr>
      <td>46</td>
      <td>1.061000</td>
    </tr>
    <tr>
      <td>47</td>
      <td>1.078500</td>
    </tr>
    <tr>
      <td>48</td>
      <td>0.977600</td>
    </tr>
    <tr>
      <td>49</td>
      <td>1.003300</td>
    </tr>
    <tr>
      <td>50</td>
      <td>1.057800</td>
    </tr>
    <tr>
      <td>51</td>
      <td>1.020600</td>
    </tr>
    <tr>
      <td>52</td>
      <td>1.070000</td>
    </tr>
    <tr>
      <td>53</td>
      <td>1.036000</td>
    </tr>
    <tr>
      <td>54</td>
      <td>0.931500</td>
    </tr>
    <tr>
      <td>55</td>
      <td>1.087600</td>
    </tr>
    <tr>
      <td>56</td>
      <td>0.873000</td>
    </tr>
    <tr>
      <td>57</td>
      <td>0.986200</td>
    </tr>
    <tr>
      <td>58</td>
      <td>0.976500</td>
    </tr>
    <tr>
      <td>59</td>
      <td>0.943800</td>
    </tr>
    <tr>
      <td>60</td>
      <td>1.092000</td>
    </tr>
  </tbody>
</table><p>
```
:::
:::

::: {.cell .code execution_count="43" execution="{\"iopub.execute_input\":\"2026-08-24T22:31:41.614062Z\",\"iopub.status.busy\":\"2026-08-24T22:31:41.613275Z\",\"iopub.status.idle\":\"2026-08-24T22:33:04.004004Z\",\"shell.execute_reply\":\"2026-08-24T22:33:04.003095Z\",\"shell.execute_reply.started\":\"2026-08-24T22:31:41.614013Z\"}" trusted="true"}
``` python
finetuned_responses = []

for prompt in test_prompts:

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        do_sample=False,
    )

    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )

    finetuned_responses.append({
        "prompt": prompt,
        "response": response,
    })

    print("=" * 80)
    print("PROMPT:")
    print(prompt)

    print("\nFINE-TUNED RESPONSE:")
    print(response)
```

::: {.output .stream .stdout}
    ================================================================================
    PROMPT:
    What is the capital of France?

    FINE-TUNED RESPONSE:
    Paris is the capital of France, much like the city of New York is the capital of the United States. It is the country’s political, cultural, and economic hub, home to the French government, the Louvre Museum, and many international organizations. Key Takeaway: Paris is the capital of France, serving as its main center for government, culture, and commerce.
    ================================================================================
    PROMPT:
    What is 27 * 43?

    FINE-TUNED RESPONSE:
    Think of 27 as 20 + 7 and 43 as 40 + 3. Multiply each pair: (20×40)=800, (20×3)=60, (7×40)=280, (7×3)=21. Add them: 800 + 60 + 280 + 21 = 1161. So, 27 × 43 = 1161. Key Takeaway: Break numbers into parts, multiply each pair, then sum the results.
    ================================================================================
    PROMPT:
    Explain what a Python decorator is.

    FINE-TUNED RESPONSE:
    A decorator is like a frame around a picture: it wraps the original picture (function) and adds extra features (side effects) without changing the picture itself. In Python, a decorator is a function that takes another function as an argument, modifies it, and returns a new function. You apply it using @decorator_name above the target function. Decorators can add logging, timing, authentication, or caching, making code cleaner and more reusable. Key Takeaway: A decorator wraps a function to add behavior without altering its core logic.
    ================================================================================
    PROMPT:
    Why does this Python code have O(n²) complexity?

    FINE-TUNED RESPONSE:
    Think of a librarian who must check every book on each shelf to find a specific title. If there are n shelves and each shelf has n books, the librarian checks n×n cells, resulting in O(n²) time. Similarly, nested loops where the inner loop runs n times for each of the outer loop’s n iterations produce O(n²) complexity. Key Takeaway: Nested loops with both indices going up to n yield O(n²) time complexity.
    ================================================================================
    PROMPT:
    Find the bug in this code and explain how to fix it.

    FINE-TUNED RESPONSE:
    The code has a logical error: it checks if the user is logged in before attempting to access the user object, but the login check is performed after the user object is already accessed. This can cause a KeyError if the user is not logged in. To fix it, move the login check to the beginning of the function and ensure the user object is only accessed if the user is authenticated. Key Takeaway: Always validate user state before accessing user data to prevent KeyError.
    ================================================================================
    PROMPT:
    A train travels 60 km in 45 minutes. What is its average speed?

    FINE-TUNED RESPONSE:
    Think of a car that drives 60 miles in 1 hour: its speed is 60 mph. Similarly, the train covers 60 km in 45 minutes, which is 0.75 hours. Divide distance by time: 60 km ÷ 0.75 h = 80 km/h. Answer: 80 km/h. Key Takeaway: Average speed is total distance divided by total time.
    ================================================================================
    PROMPT:
    You have 12 balls and one is heavier. Find the heavier ball using a balance scale in 3 weighings.

    FINE-TUNED RESPONSE:
    Think of the balls as 12 rooms, and each weighing as a door that splits the rooms into two groups. On each weighing, place some balls on each side of the scale. If the scale tips, the heavier group contains the odd ball; if balanced, the odd ball is among the unweighed ones. Repeat this process, narrowing the suspect group until only one remains. With three weighings, you can isolate the heavier ball among 12. Key Takeaway: Divide the balls into thirds, compare groups, and iteratively narrow the suspect set to find the heavier ball in three weighings.
    ================================================================================
    PROMPT:
    Design a scalable RAG architecture capable of handling 10,000 concurrent users.

    FINE-TUNED RESPONSE:
    Imagine a city’s traffic system: the RAG is the road network, the index is the traffic lights, and the query router is the traffic controller. To handle 10,000 users, deploy multiple lightweight routers (edge nodes) that offload queries to the nearest index shard. Use a distributed cache to store frequently accessed chunks, reducing redundant retrievals. Add rate limiting and circuit breakers to prevent overload. Monitor latency and throughput, scaling the index shards horizontally as demand grows. This modular design ensures high availability and performance under load. Key Takeaway: Scale RAG by distributing routers, caching, and sharding to manage high user concurrency efficiently.
    ================================================================================
    PROMPT:
    Why might a distributed system return stale data intermittently?

    FINE-TUNED RESPONSE:
    Imagine a library where books are updated on a central server, but patrons can only check the shelf labels, which are updated slowly. If a patron checks the shelf while the server is still syncing, they see the old label. Similarly, in a distributed system, a client may read a cached copy that hasn’t been refreshed since the last update, leading to stale data. This is especially common with eventual consistency models where updates propagate asynchronously. Key Takeaway: Stale data occurs when clients read cached or outdated versions before the system updates them.
    ================================================================================
    PROMPT:
    Explain the difference between LoRA and full fine-tuning.

    FINE-TUNED RESPONSE:
    Imagine a kitchen where you can either replace every utensil (full fine‑tuning) or add a few new tools (LoRA) to a set of existing ones. Full fine‑tuning updates every weight, which is powerful but expensive. LoRA adds a small, trainable layer (the adapter) that modifies the original weights without changing them, saving compute and memory. LoRA is like adding a new recipe without buying a new stove. Key Takeaway: LoRA fine‑tunes a model by adding a lightweight adapter, while full fine‑tuning updates all weights.
:::
:::
