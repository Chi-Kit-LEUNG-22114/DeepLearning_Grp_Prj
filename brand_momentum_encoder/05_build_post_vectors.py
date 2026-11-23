#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from brand_encoder.common import load_yaml,ensure_dir,numeric

def mean_lists(series):
 vals=[np.asarray(v,dtype=np.float32) for v in series if isinstance(v,(list,np.ndarray))]
 return np.mean(vals,axis=0).tolist() if vals else None

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config/pipeline.yaml");a=p.parse_args();cfg=load_yaml(a.config);out=Path(cfg["project"]["output_root"])
 posts=pd.read_parquet(out/"canonical/posts.parquet");txt=pd.read_parquet(out/"embeddings/post_text.parquet");img=pd.read_parquet(out/"embeddings/images.parquet");com=pd.read_parquet(out/"embeddings/comments_by_post.parquet")
 if not img.empty: img=img.groupby("post_id",as_index=False).agg(image_embedding=("embedding",mean_lists),image_count=("image_id","count"),image_available=("available","max"))
 df=posts.merge(txt,on="post_id",how="left").merge(img,on="post_id",how="left").merge(com,on="post_id",how="left")
 df=numeric(df,["likes","comments_count","shares","views","impressions","followers"]);df["engagement_weight"]=1+np.log1p(df.likes.fillna(0)+2*df.comments_count.fillna(0)+3*df.shares.fillna(0))
 dest=ensure_dir(out/"features")/"post_vectors.parquet";df.to_parquet(dest,index=False);print(dest,df.shape)
if __name__=="__main__":main()
