# Weekly AI Insight — 2026-10-30

## Title
**Extending FunctionGemma for Practical On-Device Mobile Function Calling**

## Source
- **arXiv:** https://arxiv.org/abs/2609.25373
- **Authors:** Ali Rezagholizadeh et al.
- **Submitted:** September 21, 2026
- **Venue:** EMNLP 2026
- **GitHub / HuggingFace:** No standalone repo; official guide and base model at
  - https://ai.google.dev/gemma/docs/mobile-actions
  - https://huggingface.co/google/functiongemma-270m-it
  - Community model: https://huggingface.co/Yagna1/functiongemma-270m-mobile-actions

---

## Why It Matters

Most function-calling work targets cloud-hosted, multi-billion-parameter models.
This paper asks a different question: *can a 270-million-parameter model run entirely
on a phone and still handle the full breadth of real-world Android actions?*

The answer is yes — with the right data. The authors build **MOBILEACTIONSEXTENDED**,
a synthetic, schema-validated dataset of ~9,500 conversations spanning **15 device-control
categories** (messaging, alarms, calls, app launches, web search, calendar, settings, …).
They fine-tune Google's FunctionGemma 270M on this data and raise end-to-end accuracy from
**29.3% → 76.5%** on their benchmark, reaching **82.3%** on Google's Mobile Actions benchmark
— only 8 points behind Google's specialist model, at a fraction of the compute.

### Why a student or developer should care
- **No GPU required for inference:** FunctionGemma 270M runs on a mid-range phone CPU via LiteRT-LM,
  and on a laptop in under a second per call.
- **No API fees:** All inference stays local; no cloud token costs.
- **Data-centric insight:** The largest accuracy gain comes from *expanding the category coverage*
  of training data, not from scaling the model. A student can replicate this on any domain
  by building a similarly structured synthetic dataset (a few thousand examples) with an LLM.
- **Re-usable fine-tuning recipe:** The paper uses standard Hugging Face TRL; any laptop with 8 GB
  RAM can fine-tune a 270M model in a few hours.

---

## Implementation Details

### Existing resources
| Resource | Link |
|---|---|
| Base model (HF) | `google/functiongemma-270m-it` |
| Google fine-tuning guide | https://ai.google.dev/gemma/docs/mobile-actions |
| Community mobile-actions model | `Yagna1/functiongemma-270m-mobile-actions` |
| Mobile Actions dataset (HF) | `google/mobile-actions` |

### How to try it in 5 minutes (no fine-tuning required)
```bash
pip install transformers accelerate
```
Then use the proof-of-concept script in this folder (`functiongemma_poc.py`).
Set `DEMO_MODE = False` to load the actual 270M model from Hugging Face and run live inference.

### Fine-tune your own domain variant
```python
from trl import SFTTrainer, SFTConfig
# Build a dataset of {system_prompt, tools_json, user_request, expected_call} rows
# Fine-tune with SFTTrainer on ~5,000–10,000 examples
# Total GPU memory needed: ~4 GB (or CPU with more time)
```
The paper shows that 9,500 domain-specific examples are enough to triple accuracy for a 270M model.

---

## Generated Proof-of-Concept Script

See [`functiongemma_poc.py`](./functiongemma_poc.py) in this folder.

The script:
1. Defines four mobile-action tools (send_message, set_alarm, open_app, web_search) in JSON schema format.
2. Builds a FunctionGemma-style prompt with the tools serialized in the system turn.
3. Parses the model output into a validated function call dict.
4. Runs in **demo mode** (no download) by default; set `DEMO_MODE = False` to use the real model.

---

## Key Takeaway
The paper's insight — that tiny models become capable function-callers when given
*comprehensive, schema-validated training data* — generalises far beyond mobile actions.
Any developer can apply the same recipe to their own tool-use domain
(home automation, API testing, local scripts) using open weights and a modest dataset.
