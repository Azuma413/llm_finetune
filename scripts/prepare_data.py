"""data/ 以下の DiscordChatExporter JSON から URL を除去して 1 つの jsonl に統合する.

出力: 1 行 1 ドキュメント の jsonl。
同一チャンネル内で時間的に近い連続発言をまとめて 1 ドキュメント (= 1 学習サンプル) とする。
"""

import argparse
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

# http(s):// 形式・www. 始まり・スキーム無しの明らかなドメイン付きパス
URL_RE = re.compile(
    r"""(?ix)
    \b(?:
        (?:https?|ftp)://[^\s<>"'）」』】]+
      | www\.[^\s<>"'）」』】]+
    )
    """
)
# Markdown リンク [表示テキスト](url) は表示テキストのみ残す
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:https?|ftp)://[^)\s]*\)")
# Discord の <url> 埋め込み抑制記法
ANGLE_URL_RE = re.compile(r"<(?:https?|ftp)://[^>\s]*>")

ZERO_WIDTH_RE = re.compile(r"[​-\u200F  ﻿]")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
TRAILING_WS_RE = re.compile(r"[ \t　]+$", re.MULTILINE)

SKIP_TYPES = {"ChannelPinnedMessage", "ThreadCreated", "GuildMemberJoin", "Call"}


def strip_urls(text: str) -> str:
    text = MD_LINK_RE.sub(r"\1", text)
    text = ANGLE_URL_RE.sub("", text)
    text = URL_RE.sub("", text)
    return text


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = strip_urls(text)
    text = TRAILING_WS_RE.sub("", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def load_messages(path: Path):
    with path.open(encoding="utf-8") as f:
        export = json.load(f)
    channel = export.get("channel", {}).get("name", path.stem)
    out = []
    for m in export.get("messages", []):
        if m.get("type") in SKIP_TYPES:
            continue
        content = clean(m.get("content", ""))
        if not content:  # URL のみ / 添付のみの発言は除去
            continue
        out.append(
            {
                "ts": parse_ts(m["timestamp"]),
                "author": m["author"].get("nickname") or m["author"]["name"],
                "text": content,
            }
        )
    out.sort(key=lambda x: x["ts"])
    return channel, out


def group_sessions(messages, gap_minutes: int, max_msgs: int, max_chars: int):
    """時間的に近い連続発言をまとめる。"""
    sessions, cur = [], []
    for m in messages:
        if cur:
            gap = (m["ts"] - cur[-1]["ts"]).total_seconds() / 60.0
            too_long = len(cur) >= max_msgs or sum(len(x["text"]) for x in cur) >= max_chars
            if gap > gap_minutes or too_long:
                sessions.append(cur)
                cur = []
        cur.append(m)
    if cur:
        sessions.append(cur)
    return sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/processed/discord.jsonl")
    ap.add_argument("--gap-minutes", type=int, default=30)
    ap.add_argument("--max-msgs", type=int, default=16)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--min-chars", type=int, default=8, help="この文字数未満のドキュメントは捨てる")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_raw = n_kept = n_docs = 0
    per_channel = []
    seen = set()

    with out_path.open("w", encoding="utf-8") as fo:
        for path in sorted(data_dir.glob("*.json")):
            with path.open(encoding="utf-8") as f:
                n_raw += len(json.load(f).get("messages", []))
            channel, msgs = load_messages(path)
            n_kept += len(msgs)
            docs_here = 0
            for sess in group_sessions(msgs, args.gap_minutes, args.max_msgs, args.max_chars):
                text = "\n".join(m["text"] for m in sess).strip()
                if len(text) < args.min_chars:
                    continue
                if text in seen:  # 完全重複の除去
                    continue
                seen.add(text)
                rec = {
                    "text": text,
                    "channel": channel,
                    "source": path.name,
                    "n_messages": len(sess),
                    "start": sess[0]["ts"].isoformat(),
                    "end": sess[-1]["ts"].isoformat(),
                }
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                docs_here += 1
            n_docs += docs_here
            per_channel.append((path.name, channel, len(msgs), docs_here))

    print(f"{'file':<12}{'channel':<24}{'msgs':>8}{'docs':>8}")
    for fn, ch, nm, nd in per_channel:
        print(f"{fn:<12}{ch:<24}{nm:>8}{nd:>8}")
    print("-" * 52)
    print(f"raw messages     : {n_raw}")
    print(f"kept messages    : {n_kept} (URL 除去後に空になったものを削除)")
    print(f"documents        : {n_docs}")
    print(f"wrote            : {out_path}")


if __name__ == "__main__":
    main()
