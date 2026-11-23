#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from brand_encoder.common import load_yaml,ensure_dir

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config/pipeline.yaml");a=p.parse_args();cfg=load_yaml(a.config);out=Path(cfg["project"]["output_root"]);df=pd.read_parquet(out/"weekly/brand_week_features.parquet");df["week_start"]=pd.to_datetime(df.week_start);L=cfg["forecast"]["context_length"]
 meta={"brand_id","week_start","date","target_date","target_gt_t_plus_1","geo","query","source_name","source_file"};features=[c for c in df.select_dtypes(include=[np.number]).columns if c not in meta and c!="target_gt_t_plus_1"]
 bounds=[("train",pd.Timestamp.min,pd.Timestamp(cfg["splits"]["train_end"])),("validation",pd.Timestamp(cfg["splits"]["train_end"]),pd.Timestamp(cfg["splits"]["validation_end"])),("test",pd.Timestamp(cfg["splits"]["validation_end"]),pd.Timestamp(cfg["splits"]["test_end"]))];dest=ensure_dir(out/"sequences")
 for name,lo,hi in bounds:
  xs=[];ys=[];brands=[];ends=[]
  for brand,g in df.sort_values("week_start").groupby("brand_id"):
   g=g.reset_index(drop=True)
   for i in range(L-1,len(g)):
    row=g.iloc[i]
    if not (row.week_start>lo and row.week_start<=hi) or pd.isna(row.target_gt_t_plus_1):continue
    x=g.iloc[i-L+1:i+1][features].astype(float).to_numpy(np.float32)
    if np.isfinite(x).all():xs.append(x);ys.append([row.target_gt_t_plus_1]);brands.append(brand);ends.append(str(row.week_start.date()))
  np.savez_compressed(dest/f"{name}.npz",X=np.asarray(xs,np.float32),y=np.asarray(ys,np.float32),brand_id=np.asarray(brands),input_end_date=np.asarray(ends),feature_names=np.asarray(features))
 long=df.rename(columns={"brand_id":"item_id","week_start":"timestamp","google_trends":"target"});long.to_parquet(dest/"forecast_long.parquet",index=False);print(dest)
if __name__=="__main__":main()
