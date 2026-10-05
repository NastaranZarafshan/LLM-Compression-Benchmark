from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

from .batching import collate_fixed_blocks


@dataclass
class DistillationResult:
    student: torch.nn.Module
    final_loss: float
    steps: int


def _autocast_context(device: str, enabled: bool):
    if enabled and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def distill_student(
    teacher_id: str,
    student_id: str,
    train_dataset: Any,
    *,
    device: str,
    trust_remote_code: bool,
    temperature: float,
    alpha_kl: float,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    warmup_steps: int,
    weight_decay: float,
    max_grad_norm: float,
    fp16: bool,
) -> DistillationResult:
    teacher_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, trust_remote_code=trust_remote_code, torch_dtype=teacher_dtype
    ).to(device)
    # Keep trainable weights in FP32; autocast + GradScaler handles mixed-precision compute.
    student = AutoModelForCausalLM.from_pretrained(
        student_id, trust_remote_code=trust_remote_code, torch_dtype=torch.float32
    ).to(device)
    teacher.eval()
    student.train()

    if teacher.config.vocab_size != student.config.vocab_size:
        raise ValueError(
            "Teacher and student vocabulary sizes differ. Classical logit KD requires aligned vocabularies. "
            f"teacher={teacher.config.vocab_size}, student={student.config.vocab_size}"
        )

    loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collate_fixed_blocks,
        drop_last=False,
    )
    if len(loader) == 0:
        raise ValueError("Training DataLoader is empty")

    optimizer = AdamW(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_steps),
        num_training_steps=int(max_steps),
    )

    use_scaler = bool(fp16 and device.startswith("cuda"))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    iterator = iter(loader)
    last_loss = float("nan")
    opt_step = 0
    micro_step = 0
    progress = tqdm(total=max_steps, desc="Distillation", unit="step")

    while opt_step < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            with _autocast_context(device, use_scaler):
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

        with _autocast_context(device, use_scaler):
            s_out = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            student_logits = s_out.logits
            t = float(temperature)
            token_kl = F.kl_div(
                F.log_softmax(student_logits / t, dim=-1),
                F.softmax(teacher_logits / t, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            valid = attention_mask.bool()
            kl = token_kl.masked_select(valid).mean() * (t * t)
            ce = s_out.loss
            loss = alpha_kl * kl + (1.0 - alpha_kl) * ce
            scaled_loss = loss / int(gradient_accumulation_steps)

        scaler.scale(scaled_loss).backward()
        micro_step += 1
        last_loss = float(loss.detach().item())

        if micro_step % int(gradient_accumulation_steps) == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            opt_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{last_loss:.3f}")

    progress.close()
    del teacher
    if device.startswith("cuda"):
        student = student.to(dtype=torch.float16)
    student.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return DistillationResult(student=student, final_loss=last_loss, steps=opt_step)
