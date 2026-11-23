#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from brand_encoder.common import load_yaml,ensure_dir,week_start
from brand_encoder.aggregation.pooling import pool_matrix

EMBED=["image_embedding","title_bge","caption_bge","tags_bge","comment_mean","comment_weighted_mean"]
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config/pipeline.yaml");a=p.parse_args();cfg=load_yaml(a.config);out=Path(cfg["project"]["output_root"]);df=pd.read_parquet(out/"features/post_vectors.parquet")
 df["week_start"]=week_start(df.created_at,cfg["project"]["timezone"]);rows=[]
 for (brand,week),g in df.dropna(subset=["brand_id","week_start"]).groupby(["brand_id","week_start"]):
  r={"brand_id":brand,"week_start":week,"post_count":len(g),"likes_total":g.likes.fillna(0).sum(),"comments_total":g.comments_count.fillna(0).sum(),"shares_total":g.shares.fillna(0).sum(),"engagement_weight_total":g.engagement_weight.sum()}
  for col in EMBED:
   vals=[];weights=[]
   if col not in g:continue
   for _,x in g.iterrows():
    if isinstance(x[col],(list,np.ndarray)):vals.append(np.asarray(x[col],np.float32));weights.append(x.engagement_weight)
   if vals:
    q=pool_matrix(np.vstack(vals),weights,cfg["pooling"]["max"],cfg["pooling"]["min"])
    for stat,v in q.items():r[f"{col}__{stat}"]=v.tolist()
  rows.append(r)
 dest=ensure_dir(out/"weekly")/"brand_week_unreduced.parquet";pd.DataFrame(rows).to_parquet(dest,index=False);print(dest,len(rows))
if __name__=="__main__":main()
