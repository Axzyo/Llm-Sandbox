"""LoRA SFT on the reward-filtered goal-set dataset (expert iteration's
imitation step).

Input rows are {system, user, assistant} (train/extract_dataset.py). Each is
rendered through the base model's chat template; loss is masked to the
assistant completion only, so the student learns to EMIT goal-sets, not to
parrot prompts. Base defaults to Qwen2.5-3B-Instruct, loaded 4-bit when
bitsandbytes is available (RTX 3080 Ti / 12 GB), fp16 otherwise.

Requires the GPU stack (train/requirements-train.txt).

Usage:
    python train/train_lora.py --data train/data/sft.jsonl --out train/rounds/round_1/adapter
"""
import argparse
import json
import os


def load_rows(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/data/sft.jsonl")
    ap.add_argument("--out", required=True, help="adapter output dir")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = load_rows(args.data)
    if not rows:
        raise SystemExit(f"no training rows in {args.data}")
    print(f"{len(rows)} examples from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    # prompt/completion columns: TRL masks loss to the completion (the goal-set)
    def to_row(row):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": row["system"]},
             {"role": "user", "content": row["user"]}],
            tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "completion": row["assistant"] + tokenizer.eos_token}

    dataset = Dataset.from_list(rows).map(to_row, remove_columns=["system", "user", "assistant"])

    try:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
        )
        print("base loaded 4-bit (bitsandbytes)")
    except Exception as exc:
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                                     device_map="auto")
        print(f"4-bit unavailable ({exc.__class__.__name__}), base loaded fp16")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            max_length=args.max_seq,
            completion_only_loss=True,        # loss on the goal-set, not the prompt
            logging_steps=10,
            save_strategy="no",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            report_to=[],
        ),
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved -> {args.out}")


if __name__ == "__main__":
    main()
