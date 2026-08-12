"""学習した LoRA を Qwen/Qwen3.5-9B (instruct) に適用し、会話口調が転移したかを確認する.

- 同一プロセスに読み込んだ 1 つのモデルで adapter ON/OFF を切り替えて比較する
  (peft の disable_adapter() を使うので、重み以外の条件は完全に同一)。
- 定性: chat template ありの応答 / 素の continuation を並べて出力
- 定量: 学習に使っていない Discord ドキュメントの perplexity と、文体マーカーの出現率
"""

import argparse
import json
import math
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, set_seed

CHAT_PROMPTS = [
    "最近アニメ何か見た？おすすめある？",
    "円安ってこのまま進むと思う？",
    "週末どっか旅行行きたいんだけど、どこがいい？",
    "新しいGPU買おうか迷ってるんだけど、どう思う？",
    "上司に理不尽に怒られた。慰めて。",
    "Pythonの型ヒントって書く意味ある？",
]

RAW_PROMPTS = [
    "今日は",
    "最近読んだ本の感想。",
    "正直に言うと、",
    "この前の飲み会、",
]

# 文体マーカー: 学習コーパスで特徴的な表記・語尾
MARKERS = {
    "全角？": "？",
    "全角！": "！",
    "まぁ": "まぁ",
    "〜だよね": "だよね",
    "〜かな": "かな",
    "〜そう": "そう",
    "草": "草",
    "……/......": r"(?:……|\.\.\.\.\.\.)",
    "〜けれど": "けれど",
    "俺": "俺",
}


def load_model(model_name: str, adapter: str, dtype=torch.bfloat16):
    config = AutoConfig.from_pretrained(model_name)
    arch = config.architectures[0]
    import transformers

    model_cls = getattr(transformers, arch)
    model = model_cls.from_pretrained(model_name, dtype=dtype, device_map="cuda:0")
    model.eval()
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        tok = getattr(processor, "tokenizer", processor)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name)
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt_ids, max_new_tokens, temperature, top_p, seed):
    set_seed(seed)
    # instruct の <|im_end|> と、LoRA が Base 側で学習したドキュメント区切り
    # <|endoftext|> の両方で停止させる
    eos_ids = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"),
                           tok.convert_tokens_to_ids("<|endoftext|>"),
                           tok.eos_token_id) if i is not None]
    out = model.generate(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=sorted(set(eos_ids)),
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(out[0][prompt_ids.shape[1] :], skip_special_tokens=True).strip()


@torch.no_grad()
def perplexity(model, tok, texts, device, max_len=1024):
    """ドキュメント単位の token 平均 NLL から perplexity を出す。"""
    total_nll, total_tok = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids[:, :max_len].to(device)
        if ids.shape[1] < 2:
            continue
        loss = model(input_ids=ids, labels=ids).loss.float().item()
        n = ids.shape[1] - 1
        total_nll += loss * n
        total_tok += n
    return math.exp(total_nll / total_tok), total_tok


def style_stats(texts):
    joined = "\n".join(texts)
    chars = max(len(joined), 1)
    stats = {}
    for name, pat in MARKERS.items():
        stats[name] = 1000 * len(re.findall(pat, joined)) / chars  # 1000 文字あたり
    stats["平均文長(文字)"] = sum(len(t) for t in texts) / max(len(texts), 1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="LoRA の適用先 (instruct)")
    ap.add_argument("--adapter", default="outputs/lora-qwen3.5-9b-discord/adapter")
    ap.add_argument("--val-docs", default="outputs/lora-qwen3.5-9b-discord/val_docs.jsonl")
    ap.add_argument("--out", default="outputs/eval_style.md")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ppl-docs", type=int, default=150)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false", default=False)
    ap.add_argument("--thinking", dest="thinking", action="store_true")
    args = ap.parse_args()

    model, tok = load_model(args.model, args.adapter)
    device = model.device

    rows = []  # (kind, prompt, base_output, lora_output)

    def run_pair(prompt_ids):
        with model.disable_adapter():
            base = generate(model, tok, prompt_ids, args.max_new_tokens,
                            args.temperature, args.top_p, args.seed)
        lora = generate(model, tok, prompt_ids, args.max_new_tokens,
                        args.temperature, args.top_p, args.seed)
        return base, lora

    print("=== chat prompts ===", flush=True)
    for p in CHAT_PROMPTS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        )
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        base, lora = run_pair(ids)
        rows.append(("chat", p, base, lora))
        print(f"\n--- {p}\n[base] {base}\n[lora] {lora}", flush=True)

    print("\n=== raw continuation ===", flush=True)
    for p in RAW_PROMPTS:
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        base, lora = run_pair(ids)
        rows.append(("raw", p, base, lora))
        print(f"\n--- {p}\n[base] {base}\n[lora] {lora}", flush=True)

    # ---- perplexity on held-out discord text ---------------------------------
    val = [json.loads(l)["text"] for l in Path(args.val_docs).open(encoding="utf-8")]
    val = val[: args.ppl_docs]
    with model.disable_adapter():
        ppl_base, n_tok = perplexity(model, tok, val, device)
    ppl_lora, _ = perplexity(model, tok, val, device)
    print(f"\nheld-out Discord ppl: base={ppl_base:.2f} lora={ppl_lora:.2f} ({n_tok} tokens)")

    # ---- style markers -------------------------------------------------------
    base_gen = [r[2] for r in rows]
    lora_gen = [r[3] for r in rows]
    ref = val
    s_base, s_lora, s_ref = style_stats(base_gen), style_stats(lora_gen), style_stats(ref)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# 口調転移の確認: `{args.model}` + LoRA(`{args.adapter}`)\n\n")
        f.write(f"- sampling: temperature={args.temperature}, top_p={args.top_p}, "
                f"seed={args.seed}, thinking={args.thinking}\n")
        f.write(f"- held-out Discord perplexity: **base {ppl_base:.2f} → LoRA {ppl_lora:.2f}** "
                f"({len(val)} docs / {n_tok} tokens)\n\n")
        f.write("## 文体マーカー出現率 (1000文字あたり)\n\n")
        f.write("| marker | base 出力 | LoRA 出力 | 学習コーパス |\n|---|---:|---:|---:|\n")
        for k in s_base:
            f.write(f"| {k} | {s_base[k]:.2f} | {s_lora[k]:.2f} | {s_ref[k]:.2f} |\n")
        f.write("\n## 生成比較\n")
        for kind, p, b, l in rows:
            f.write(f"\n### [{kind}] {p}\n\n**base (LoRA なし)**\n\n```\n{b}\n```\n\n"
                    f"**LoRA 適用**\n\n```\n{l}\n```\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
