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
 if not img.empty:
  img=img.groupby("post_id",as_index=False).agg(
   image_embedding=("embedding",lambda s:mean_lists(s[pd.to_numeric(img.loc[s.index,"available"],errors="coerce").fillna(0)>0])),
   image_count=("image_id","count"),image_available=("available","max"))
 else: img=pd.DataFrame(columns=["post_id","image_embedding","image_count","image_available"])
 df=posts.merge(txt,on="post_id",how="left").merge(img,on="post_id",how="left").merge(com,on="post_id",how="left")
 metrics=["likes","comments_count","shares","views","impressions"]
 if "measurement_at" in df:
  observed=pd.to_datetime(df.measurement_at,utc=True,errors="coerce")
  created=pd.to_datetime(df.created_at,utc=True,errors="coerce").dt.tz_convert(cfg["project"]["timezone"])
  week_end=(created-pd.to_timedelta(created.dt.weekday,unit="D")).dt.normalize()+pd.Timedelta(days=7)-pd.Timedelta(nanoseconds=1)
  cutoff=week_end.dt.tz_convert("UTC")
  df["metrics_snapshot_late"]=(observed>cutoff).fillna(False).astype("int8")
 else:
  df["metrics_snapshot_late"]=0
 df=numeric(df,["likes","comments_count","shares","views","impressions","followers"]);df["engagement_weight"]=1+np.log1p(df.likes.fillna(0)+2*df.comments_count.fillna(0)+3*df.shares.fillna(0))
 def alignment(row):
  image=row.get("image_embedding"); caption=row.get("caption_clip")
  if not isinstance(caption,(list,np.ndarray)):caption=row.get("title_clip")
  if not isinstance(image,(list,np.ndarray)) or not isinstance(caption,(list,np.ndarray)):return np.nan
  image=np.asarray(image,dtype=np.float32);caption=np.asarray(caption,dtype=np.float32)
  if image.shape!=caption.shape or not np.linalg.norm(image) or not np.linalg.norm(caption):return np.nan
  return float(np.dot(image/np.linalg.norm(image),caption/np.linalg.norm(caption)))
 df["image_text_alignment"]=df.apply(alignment,axis=1)
 dest=ensure_dir(out/"features")/"post_vectors.parquet";df.to_parquet(dest,index=False);print(dest,df.shape)
if __name__=="__main__":main()
