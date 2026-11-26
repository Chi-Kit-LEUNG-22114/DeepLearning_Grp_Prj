from __future__ import annotations
import glob
import json
from pathlib import Path
import pandas as pd
from .common import read_any, stable_id, normalize_missing_text

CANON={
"posts":["post_id","brand_id","platform","created_at","title","caption","tags","image_path","likes","comments_count","shares","views","impressions","followers","is_brand_owned","is_paid","measurement_at","source_name","source_file"],
"comments":["comment_id","post_id","created_at","text","likes","parent_comment_id","source_name","source_file"],
"images":["image_id","post_id","image_path","image_order","source_name","source_file"],
"trends":["brand_id","date","google_trends","geo","query","source_name","source_file"],
"brands":["brand_id","brand_name","industry","market","primary_language","source_name","source_file"]}

def expand_patterns(patterns:list[str], base:Path)->list[Path]:
    out=[]
    for pat in patterns:
        full=str((base/pat) if not Path(pat).is_absolute() else Path(pat))
        out.extend(Path(p) for p in glob.glob(full, recursive=True))
    return sorted(set(p for p in out if p.is_file()))

def canonicalize(source:dict, base:Path)->pd.DataFrame:
    kind=source["kind"]; frames=[]
    for path in expand_patterns(source.get("paths",[]), base):
        df=read_any(path).rename(columns=source.get("mapping",{}))
        for k,v in source.get("constants",{}).items(): df[k]=v
        df["source_name"]=source["name"]; df["source_file"]=str(path)
        if kind=="posts" and "post_id" not in df:
            df["post_id"]=[stable_id(source["name"], path, i, r.get("created_at"), r.get("caption")) for i,r in df.iterrows()]
        if kind=="comments" and "comment_id" not in df:
            df["comment_id"]=[stable_id(source["name"], path, i, r.get("post_id"), r.get("text")) for i,r in df.iterrows()]
        if kind=="images" and "image_id" not in df:
            df["image_id"]=[stable_id(source["name"], path, i, r.get("post_id"), r.get("image_path")) for i,r in df.iterrows()]
        frames.append(df)
    if not frames: return pd.DataFrame(columns=CANON[kind])
    merged=pd.concat(frames, ignore_index=True, sort=False)
    for c in CANON[kind]:
        if c not in merged: merged[c]=pd.NA
    return merged[CANON[kind]+[c for c in merged.columns if c not in CANON[kind]]]

def ingest_all(manifest:dict, base:Path)->dict[str,pd.DataFrame]:
    grouped={k:[] for k in CANON}
    for src in manifest.get("sources",[]): grouped[src["kind"]].append(canonicalize(src,base))
    out={}
    for kind,frames in grouped.items():
        out[kind]=pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame(columns=CANON[kind])
    # Derive image table from post image_path when no explicit images table exists.
    if out["images"].empty and not out["posts"].empty:
        rows=[]
        for _,r in out["posts"].iterrows():
            val=normalize_missing_text(r.get("image_path"))
            for idx,p in enumerate(val.split("|") if val else [],1):
                rows.append({"image_id":stable_id(r["post_id"],p),"post_id":r["post_id"],"image_path":p,"image_order":idx,"source_name":r.get("source_name"),"source_file":r.get("source_file")})
        out["images"]=pd.DataFrame(rows,columns=CANON["images"])
    return out

def ingest_embedding_json(path: str | Path, image_root: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the aggregated YouTube JSON into the pipeline's canonical tables."""
    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload.get("brands"), list):
        raise ValueError("Embedding JSON must contain a 'brands' list")

    rows: dict[str, list[dict]] = {kind: [] for kind in CANON}
    image_index: list[Path] = []
    if image_root is not None and Path(image_root).exists():
        image_index = [
            candidate for candidate in Path(image_root).rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    for brand in payload["brands"]:
        brand_id = normalize_missing_text(brand.get("brand_id"))
        if not brand_id:
            continue
        channel = brand.get("youtube_channel") or {}
        rows["brands"].append({
            "brand_id": brand_id,
            "brand_name": brand.get("brand_name"),
            "source_name": "embedding_json",
            "source_file": str(path),
        })
        for period in brand.get("periods") or []:
            for video in period.get("videos") or []:
                post_id = normalize_missing_text(video.get("video_id")) or stable_id(
                    brand_id, video.get("published_at_utc"), video.get("title")
                )
                metrics = video.get("metrics") or {}
                thumbnail = video.get("thumbnail") or {}
                image_path = normalize_missing_text(thumbnail.get("local_path"))
                if not image_path:
                    matches = [candidate for candidate in image_index if post_id in candidate.stem]
                    if matches:
                        matches.sort(key=lambda candidate: (
                            "thumbnails_b/video" not in candidate.as_posix(),
                            "thumbnails_a/video" not in candidate.as_posix(),
                            candidate.as_posix(),
                        ))
                        image_path = str(matches[0])
                rows["posts"].append({
                    "post_id": post_id,
                    "brand_id": brand_id,
                    "platform": "youtube",
                    "created_at": video.get("published_at_utc"),
                    "title": video.get("title"),
                    "caption": video.get("description"),
                    "tags": video.get("tags"),
                    "image_path": image_path,
                    "likes": metrics.get("like_count"),
                    "comments_count": metrics.get("comment_count_reported"),
                    "shares": None,
                    "views": metrics.get("view_count"),
                    "impressions": None,
                    "followers": channel.get("subscriber_count"),
                    "measurement_at": str(video.get("retrieved_at_hkt") or "").replace(" HKT", "+08:00") or None,
                    "source_name": "embedding_json",
                    "source_file": str(path),
                })
                if image_path:
                    rows["images"].append({
                        "image_id": stable_id(post_id, image_path),
                        "post_id": post_id,
                        "image_path": image_path,
                        "image_order": 1,
                        "source_name": "embedding_json",
                        "source_file": str(path),
                    })
                for comment in video.get("comments") or []:
                    rows["comments"].append({
                        "comment_id": comment.get("comment_id") or stable_id(post_id, comment.get("text")),
                        "post_id": post_id,
                        "created_at": comment.get("published_at_utc"),
                        "text": comment.get("text"),
                        # Current comment likes are not historical snapshots.
                        "likes": None,
                        "parent_comment_id": comment.get("parent_comment_id"),
                        "source_name": "embedding_json",
                        "source_file": str(path),
                    })
        # Trends metadata in this JSON describes the query, not observed weekly values.
        # Do not fabricate numerical trend observations.

    return {
        kind: pd.DataFrame(data, columns=CANON[kind])
        for kind, data in rows.items()
    }
