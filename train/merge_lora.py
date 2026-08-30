"""Merge a trained LoRA adapter into its base model and write a standalone
fp16 checkpoint, so the student can be served fast via Ollama (llama.cpp)
instead of slow bitsandbytes 4-bit inference.

Merges on CPU to keep VRAM free for a concurrent rollout. The output dir is a
plain HF model folder; `ollama create ... -f Modelfile` (FROM <this dir>)
converts it to GGUF internally.

Usage:
    python train/merge_lora.py --adapter train/rounds/round_1/adapter --out train/rounds/round_1/merged
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir from train_lora.py")
    ap.add_argument("--out", required=True, help="output dir for the merged fp16 model")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading base {args.base} (fp16, cpu) ...")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16, device_map="cpu")
    print(f"applying adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(base, args.adapter)
    print("merging ...")
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"merged model -> {args.out}")


if __name__ == "__main__":
    main()
