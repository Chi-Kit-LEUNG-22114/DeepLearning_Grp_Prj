#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from brand_encoder.common import load_yaml,ensure_dir,select_device,normalize_missing_text
from brand_encoder.encoders.text_encoder import TextEncoder
from brand_encoder.encoders.image_encoder import ClipEncoder

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config/pipeline.yaml");a=p.parse_args();cfg=load_yaml(a.config);out=Path(cfg["project"]["output_root"])
 posts=pd.read_parquet(out/"canonical/posts.parquet");dev=select_device(cfg["models"]["device"])
 bge=TextEncoder(cfg["models"]["text_encoder"],dev,cfg["text"]["max_chars_per_chunk"],cfg["text"]["normalize_embeddings"])
 clip=ClipEncoder(cfg["models"]["image_encoder"],dev);rows=[]
 for _,r in tqdm(posts.iterrows(),total=len(posts)):
  title=normalize_missing_text(r.get("title"));caption=normalize_missing_text(r.get("caption"));tags=normalize_missing_text(r.get("tags"))
  rows.append({"post_id":r.post_id,"title_available":int(bool(title)),"caption_available":int(bool(caption)),"tags_available":int(bool(tags)),"title_bge":bge.encode_one(title).tolist(),"caption_bge":bge.encode_one(caption).tolist(),"tags_bge":bge.encode_one(tags.replace("|"," ")).tolist(),"caption_clip":clip.text(caption).tolist(),"bge_model":bge.model_name,"clip_model":clip.model_name})
 dest=ensure_dir(out/"embeddings")/"post_text.parquet";pd.DataFrame(rows).to_parquet(dest,index=False);print(dest)
if __name__=="__main__":main()
