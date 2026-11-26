#!/usr/bin/env python3
"""Encode title/caption/tag text into separate BGE and CLIP blocks."""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from brand_encoder.common import (
    load_yaml, ensure_dir, select_device, normalize_missing_text,
)
from brand_encoder.encoders.text_encoder import TextEncoder
from brand_encoder.encoders.image_encoder import ClipEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out = Path(cfg["project"]["output_root"])
    posts = pd.read_parquet(out / "canonical/posts.parquet")
    device = select_device(cfg["models"]["device"])
    offline = cfg["models"].get("offline", False)

    bge = TextEncoder(
        cfg["models"]["text_encoder"], device,
        cfg["text"]["max_chars_per_chunk"],
        cfg["text"]["normalize_embeddings"], offline,
    )
    clip = ClipEncoder(cfg["models"]["image_encoder"], device, offline)

    titles = [normalize_missing_text(v) for v in posts.get("title", pd.Series("", index=posts.index))]
    captions = [normalize_missing_text(v) for v in posts.get("caption", pd.Series("", index=posts.index))]
    tags = [
        normalize_missing_text(v).replace("|", " ")
        for v in posts.get("tags", pd.Series("", index=posts.index))
    ]
    batch_size = int(cfg["models"]["text_batch_size"])

    title_bge = bge.encode_many(titles, batch_size)
    caption_bge = bge.encode_many(captions, batch_size)
    tags_bge = bge.encode_many(tags, batch_size)
    title_clip = clip.text_many(titles, batch_size)
    caption_clip = clip.text_many(captions, batch_size)

    rows = []
    for i, (_, post) in enumerate(tqdm(posts.iterrows(), total=len(posts))):
        title, caption, tag = titles[i], captions[i], tags[i]
        rows.append({
            "post_id": post.post_id,
            "title_available": int(bool(title)),
            "caption_available": int(bool(caption)),
            "tags_available": int(bool(tag)),
            "title_bge": title_bge[i].tolist() if title else None,
            "caption_bge": caption_bge[i].tolist() if caption else None,
            "tags_bge": tags_bge[i].tolist() if tag else None,
            "title_clip": title_clip[i].tolist() if title else None,
            "caption_clip": caption_clip[i].tolist() if caption else None,
            "bge_model": bge.model_name,
            "clip_model": clip.model_name,
        })
    dest = ensure_dir(out / "embeddings") / "post_text.parquet"
    pd.DataFrame(rows).to_parquet(dest, index=False)
    print(dest, "posts", len(rows))


if __name__ == "__main__":
    main()
