#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from brand_encoder.common import load_yaml,ensure_dir,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config/pipeline.yaml");a=p.parse_args();cfg=load_yaml(a.config);out=Path(cfg["project"]["output_root"]);df=pd.read_parquet(out/"weekly/brand_week_unreduced.parquet");df["week_start"]=pd.to_datetime(df.week_start);train=df[df.week_start<=pd.Timestamp(cfg["splits"]["train_end"])]
 art=ensure_dir(out/"artifacts");manifest={}
 mapping={"image_embedding":cfg["reduction"]["image_components"],"title_bge":cfg["reduction"]["title_components"],"caption_bge":cfg["reduction"]["caption_components"],"comment_mean":cfg["reduction"]["comment_components"],"comment_weighted_mean":cfg["reduction"]["comment_components"],"tags_bge":cfg["reduction"]["tag_components"]}
 for prefix,n in mapping.items():
  for col in [c for c in train if c.startswith(prefix+"__")]:
   vals=[np.asarray(v,np.float32) for v in train[col] if isinstance(v,(list,np.ndarray))]
   if len(vals)<2:continue
   x=np.vstack(vals);k=max(1,min(int(n),x.shape[1],x.shape[0]-1));model=PCA(k,random_state=cfg["project"]["random_seed"]).fit(x);joblib.dump(model,art/f"{col}.joblib");manifest[col]={"components":k,"explained_variance":float(model.explained_variance_ratio_.sum())}
 num=[c for c in ["post_count","likes_total","comments_total","shares_total","engagement_weight_total"] if c in train];sc=StandardScaler().fit(np.log1p(train[num].fillna(0)));joblib.dump((sc,num),art/"numeric_scaler.joblib");write_json(art/"reducer_manifest.json",manifest);print(art)
if __name__=="__main__":main()
