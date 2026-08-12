# llm finetune

自分の Discord 発言ログ (DiscordChatExporter の JSON) で `Qwen/Qwen3.5-9B-Base` を
LoRA fine-tuning し、その LoRA を instruct モデル `Qwen/Qwen3.5-9B` に載せて
会話口調が転移するかを見る実験。

## ルール
- uvを使う

## セットアップ

```bash
uv venv --python 3.12
UV_TORCH_BACKEND=auto uv pip install torch torchvision
uv pip install unsloth unsloth_zoo trl peft datasets bitsandbytes accelerate
```

以降のコマンドは `uv run python ...` で実行する。

## 1. データ整形

```bash
uv run python scripts/prepare_data.py
```

`data/*.json` (20 チャンネル / 41,538 発言) から

- URL (`http(s)://`, `www.`, `<url>`, Markdown リンク) を除去
- URL や添付のみで空になった発言を削除
- 同一チャンネル内で 30 分以内に連続した発言を 1 ドキュメントにまとめる
  (最大 16 発言 / 1500 文字)
- 完全重複ドキュメントを除去

して `data/processed/discord.jsonl` (1 行 = `{"text", "channel", ...}`) を出力する。
全角/半角の使い分けは文体そのものなので NFKC 正規化はかけていない。

## 2. LoRA 学習

```bash
uv run python scripts/train_lora.py
```

- Unsloth + peft、16bit LoRA (`--load-in-4bit` で QLoRA にも切替可)
- 対話データではないのでチャットテンプレートは使わず、ドキュメントを
  `<|endoftext|>` 区切りで連結して 1024 トークンのブロックに packing し、
  全トークンを対象にした素の next-token prediction で学習する
- LoRA は language model 側のみ (vision tower は凍結)。full-attention 層の
  `q/k/v/o_proj` と gated-deltanet 層の `in_proj_qkv/z/a/b`・`out_proj`、
  および MLP を対象にする
- Qwen3.5 の gated-deltanet は fp16 で NaN になるため bf16 固定

出力: `outputs/lora-qwen3.5-9b-discord/adapter/`

## 3. 口調転移の確認

```bash
uv run python scripts/eval_style.py
```

`Qwen/Qwen3.5-9B` (instruct) を 1 つだけ読み込み、peft の `disable_adapter()` で
LoRA の ON/OFF だけを切り替えて比較する。

- 定性: chat template 経由の応答と素の continuation を並べて出力
- 定量: 学習に使っていない Discord ドキュメントの perplexity、文体マーカー出現率

結果は `outputs/eval_style.md` に書き出される。
