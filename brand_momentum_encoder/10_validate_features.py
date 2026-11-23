#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from brand_encoder.common import load_yaml,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config/pipeline.yaml");a=p.parse_args();cfg=load_yaml(a.config);out=Path(cfg["project"]["output_root"]);df=pd.read_parquet(out/"weekly/brand_week_features.parquet");issues=[]
 if df.duplicated(["brand_id","week_start"]).any():issues.append("duplicate brand-week rows")
 nums=df.select_dtypes(include=[np.number]);
 if np.isinf(nums.to_numpy()).any():issues.append("infinite numerical values")
 tgt=pd.to_datetime(df.target_date,errors="coerce");wk=pd.to_datetime(df.week_start,errors="coerce");bad=tgt.notna()&((tgt-wk).dt.days!=7)
 if bad.any():issues.append(f"{int(bad.sum())} labels are not exactly one week ahead")
 report={"rows":len(df),"brands":int(df.brand_id.nunique()),"weeks":int(df.week_start.nunique()),"numeric_features":len(nums.columns),"issues":issues,"missing_rate_top":df.isna().mean().sort_values(ascending=False).head(20).to_dict()};write_json(out/"feature_validation_report.json",report);print(json.dumps(report,indent=2,default=str))
if __name__=="__main__":main()
