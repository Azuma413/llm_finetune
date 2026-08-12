"""Qwen/Qwen3.5-9B-Base を Discord ログで LoRA fine-tuning する (素の next-token prediction).

対話データではないので chat template は使わず、ドキュメントを EOS 区切りで連結して
max_seq_length のブロックに packing し、全トークンを予測対象とする。
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # .env の WANDB_API_KEY を読み込む

import torch
from unsloth import FastModel  # noqa: E402,F401  (transformers より先に import する必要がある)
from transformers import Trainer, TrainingArguments  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--data", default="data/processed/discord.jsonl")
    ap.add_argument("--output-dir", default="outputs/lora-qwen3.5-9b-discord")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--load-in-4bit", action="store_true", help="QLoRA (VRAM が足りない場合)")
    # LoRA
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    # optim
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="スモークテスト用")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    # wandb
    ap.add_argument("--wandb-project", default="llm-finetune-discord")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false", default=True)
    return ap.parse_args()


class PackedBlocks(torch.utils.data.Dataset):
    """固定長ブロック列。全ブロックが同じ長さなので padding も collator も不要。"""

    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        ids = self.blocks[i]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)}


def collate(features):
    return {
        k: torch.tensor([f[k] for f in features], dtype=torch.long)
        for k in ("input_ids", "attention_mask", "labels")
    }


def build_blocks(texts, tok, block_size, eos_id):
    """ドキュメントを EOS 区切りで連結し、block_size ごとに切り出す。"""
    stream = []
    for t in texts:
        stream.extend(tok(t, add_special_tokens=False)["input_ids"])
        stream.append(eos_id)
    n = (len(stream) // block_size) * block_size
    return [stream[i : i + block_size] for i in range(0, n, block_size)], len(stream)


def setup_wandb(args):
    """.env の WANDB_API_KEY があれば wandb を有効にする。無ければ黙って無効化。"""
    if not args.wandb:
        return False
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY が無いので wandb へのログ送信は無効にします。")
        return False
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name
    return True


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_wandb = setup_wandb(args)

    model, processor = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,  # Qwen3.5 の gated-deltanet は fp16 だと NaN になる
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )
    tok = getattr(processor, "tokenizer", processor)
    eos_id = tok.eos_token_id
    print(f"eos_token = {tok.eos_token!r} (id={eos_id})")

    # ---- data ----------------------------------------------------------------
    records = [json.loads(l) for l in Path(args.data).open(encoding="utf-8")]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_ratio))
    val_recs, train_recs = records[:n_val], records[n_val:]

    train_blocks, n_train_tok = build_blocks(
        [r["text"] for r in train_recs], tok, args.max_seq_length, eos_id
    )
    val_blocks, n_val_tok = build_blocks(
        [r["text"] for r in val_recs], tok, args.max_seq_length, eos_id
    )
    print(
        f"docs train/val = {len(train_recs)}/{len(val_recs)}  "
        f"tokens = {n_train_tok}/{n_val_tok}  "
        f"blocks({args.max_seq_length}) = {len(train_blocks)}/{len(val_blocks)}"
    )
    if not val_blocks:  # holdout が 1 ブロックに満たない場合は学習側に戻す
        val_blocks = train_blocks[:1]

    # 検証用ドキュメントは step3 の perplexity 評価でも使うので保存しておく
    with (out_dir / "val_docs.jsonl").open("w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- LoRA ----------------------------------------------------------------
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        finetune_vision_layers=False,  # テキストのみ学習
        finetune_language_layers=True,
        finetune_attention_modules=True,  # full-attention と linear-attention (GDN) の両方
        finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
    )
    model.print_trainable_parameters()
    targeted = sorted({n.rsplit(".lora_A", 1)[0].rsplit(".", 1)[-1]
                       for n, _ in model.named_parameters() if ".lora_A" in n})
    print("LoRA target modules:", targeted)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if use_wandb:
        # Trainer より先に run を作っておくと、以下の config も一緒に残せる
        # (HF の WandbCallback は既存 run があればそれを再利用する)
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **vars(args),
                "train_docs": len(train_recs),
                "val_docs": len(val_recs),
                "train_tokens": n_train_tok,
                "val_tokens": n_val_tok,
                "train_blocks": len(train_blocks),
                "val_blocks": len(val_blocks),
                "lora_target_modules": targeted,
                "trainable_params": n_trainable,
                "gpu": torch.cuda.get_device_name(0),
            },
        )

    # ---- train ---------------------------------------------------------------
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        report_to="wandb" if use_wandb else "none",
        run_name=args.wandb_run_name,
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=PackedBlocks(train_blocks),
        eval_dataset=PackedBlocks(val_blocks),
        data_collator=collate,
    )

    before = trainer.evaluate()
    print(f"[before] eval_loss={before['eval_loss']:.4f} ppl={math.exp(before['eval_loss']):.2f}")

    result = trainer.train()

    after = trainer.evaluate()
    print(f"[after ] eval_loss={after['eval_loss']:.4f} ppl={math.exp(after['eval_loss']):.2f}")

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "train_blocks": len(train_blocks),
                "val_blocks": len(val_blocks),
                "train_tokens": n_train_tok,
                "eval_loss_before": before["eval_loss"],
                "eval_loss_after": after["eval_loss"],
                "ppl_before": math.exp(before["eval_loss"]),
                "ppl_after": math.exp(after["eval_loss"]),
                "train_runtime_sec": result.metrics.get("train_runtime"),
                "log_history": trainer.state.log_history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    peak_vram_gb = torch.cuda.max_memory_reserved() / 1e9
    print(f"saved adapter -> {adapter_dir}")
    print(f"peak VRAM = {peak_vram_gb:.1f} GB")

    if use_wandb:
        import wandb

        wandb.summary.update(
            {
                "eval_loss_before": before["eval_loss"],
                "eval_loss_after": after["eval_loss"],
                "ppl_before": math.exp(before["eval_loss"]),
                "ppl_after": math.exp(after["eval_loss"]),
                "train_runtime_sec": result.metrics.get("train_runtime"),
                "peak_vram_gb": peak_vram_gb,
            }
        )
        wandb.save(str(out_dir / "train_summary.json"), base_path=str(out_dir))
        wandb.finish()


if __name__ == "__main__":
    main()
