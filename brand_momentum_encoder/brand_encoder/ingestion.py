from __future__ import annotations
import glob
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
