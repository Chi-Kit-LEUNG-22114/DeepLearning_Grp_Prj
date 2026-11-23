#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np,pandas as pd,joblib
from brand_encoder.common import load_yaml,ensure_dir,week_start

def slope(s,w):
 return s.rolling(w,min_periods=w).apply(lambda y:np.polyfit(np.arange(w),y,1)[0],raw=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config/pipeline.yaml");a=p.parse_args();cfg=load_yaml(a.config);out=Path(cfg["project"]["output_root"]);w=pd.read_parquet(out/"weekly/brand_week_unreduced.parquet");art=out/"artifacts";rows=w[["brand_id","week_start"]].copy()
 for path in art.glob("*.joblib"):
  if path.name=="numeric_scaler.joblib":continue
  col=path.stem
  if col not in w:continue
  m=joblib.load(path);arr=[]
  for v in w[col]:arr.append(m.transform([np.asarray(v,np.float32)])[0] if isinstance(v,(list,np.ndarray)) else np.full(m.n_components_,np.nan))
  arr=np.vstack(arr)
  for i in range(arr.shape[1]):rows[f"{col}__pca_{i+1:03d}"]=arr[:,i]
 sc,num=joblib.load(art/"numeric_scaler.joblib");z=sc.transform(np.log1p(w[num].fillna(0)))
 for i,c in enumerate(num):rows[f"{c}__scaled"]=z[:,i]
 tr=pd.read_parquet(out/"canonical/trends.parquet");tr["week_start"]=week_start(tr.date,cfg["project"]["timezone"]);tr=tr.sort_values(["brand_id","week_start"])
 g=tr.groupby("brand_id")["google_trends"];tr["gt_diff_1"]=g.diff(1);tr["gt_diff_4"]=g.diff(4);tr["gt_pct_4"]=g.pct_change(4);tr["gt_mean_4"]=g.transform(lambda s:s.rolling(4,min_periods=1).mean());tr["gt_std_4"]=g.transform(lambda s:s.rolling(4,min_periods=2).std());tr["gt_slope_4"]=g.transform(lambda s:slope(s,4));tr["target_gt_t_plus_1"]=g.shift(-1);tr["target_date"]=tr.groupby("brand_id")["week_start"].shift(-1)
 rows=tr.merge(rows,on=["brand_id","week_start"],how="left");rows["social_data_available"]=rows.filter(like="post_count__scaled").notna().any(axis=1).astype(int);dest=ensure_dir(out/"weekly")/"brand_week_features.parquet";rows.to_parquet(dest,index=False);print(dest,rows.shape)
if __name__=="__main__":main()
