#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from tqdm import tqdm
from brand_encoder.common import load_yaml,ensure_dir,select_device,numeric
from brand_encoder.encoders.text_encoder import TextEncoder
from brand_encoder.aggregation.pooling import pool_matrix

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config/pipeline.yaml");a=p.parse_args();cfg=load_yaml(a.config);out=Path(cfg["project"]["output_root"])
 c=pd.read_parquet(out/"canonical/comments.parquet");enc=TextEncoder(cfg["models"]["text_encoder"],select_device(cfg["models"]["device"]),cfg["text"]["max_chars_per_chunk"],True);rows=[]
 if c.empty:
  pd.DataFrame(rows).to_parquet(ensure_dir(out/"embeddings")/"comments_by_post.parquet",index=False);return
 c=numeric(c,["likes"]); maxn=cfg["comments"]["max_comments_per_post"]
 for pid,g in tqdm(c.groupby("post_id"),total=c.post_id.nunique()):
  if len(g)>maxn:
   top=g.nlargest(min(maxn//2,len(g)),"likes");rest=g.drop(top.index).sample(n=min(maxn-len(top),len(g)-len(top)),random_state=cfg["comments"]["sampling_seed"]);g=pd.concat([top,rest])
  mat=np.vstack([enc.encode_one(x) for x in g.text.fillna("")]);w=1+np.log1p(g.likes.fillna(0).clip(lower=0).to_numpy());pooled=pool_matrix(mat,w)
  rows.append({"post_id":pid,"comment_count_encoded":len(g),"comment_mean":pooled["mean"].tolist(),"comment_weighted_mean":pooled["weighted_mean"].tolist(),"comment_std":pooled["std"].tolist(),"comment_likes_total":float(g.likes.fillna(0).sum()),"encoder":enc.model_name})
 dest=ensure_dir(out/"embeddings")/"comments_by_post.parquet";pd.DataFrame(rows).to_parquet(dest,index=False);print(dest)
if __name__=="__main__":main()
