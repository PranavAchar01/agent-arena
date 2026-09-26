"""Measure, don't guess: how long does one LoRA step on a small VLM take on this Mac's CPU?

Model: HuggingFaceTB/SmolVLM2-500M-Video-Instruct (already in the HF cache). LoRA r=8 on the language model's
q/k/v/o projections. Each example = one verifier contact sheet + the rubric question -> the verdict text.
Reports load time, seconds per step, trainable params and peak RSS. It does NOT claim a useful tuned model:
a handful of steps on a handful of examples only measures cost.

usage: lora_cpu_probe.py SHEET_JPG [SHEET_JPG ...]
"""

import resource
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
torch.set_num_threads(4)
t0 = time.time()
proc = AutoProcessor.from_pretrained(MID, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    MID, local_files_only=True, torch_dtype=torch.bfloat16
)
load_s = time.time() - t0
model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

sheets = sys.argv[1:]
labels = ["reject: no person or hand manipulating a small object"] * len(sheets)
steps, times = 6, []
model.train()
for s in range(steps):
    img = Image.open(sheets[s % len(sheets)]).convert("RGB")
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Does this show one person's hand doing a tabletop pick-and-place? Answer accept or reject with a reason.",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": labels[s % len(labels)]}],
        },
    ]
    text = proc.apply_chat_template(msgs, add_generation_prompt=False)
    batch = proc(text=text, images=[img], return_tensors="pt")
    batch = {
        k: (v.to(torch.bfloat16) if v.dtype.is_floating_point else v)
        for k, v in batch.items()
    }
    t = time.time()
    out = model(**batch, labels=batch["input_ids"])
    out.loss.backward()
    opt.step()
    opt.zero_grad()
    times.append(time.time() - t)
    print(
        f"step {s} loss {out.loss.item():.3f} {times[-1]:.1f}s tokens {batch['input_ids'].shape[1]}",
        flush=True,
    )
rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
steady = sorted(times[1:])[len(times[1:]) // 2]
print(
    f"RESULT load_s={load_s:.1f} median_step_s={steady:.1f} first_step_s={times[0]:.1f} trainable={trainable} "
    f"total={total} peak_rss_gb={rss_gb:.2f}"
)
