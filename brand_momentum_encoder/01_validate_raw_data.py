#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from brand_encoder.common import load_yaml,ensure_dir,write_json,numeric
from brand_encoder.ingestion import ingest_all

def main():
 p=argparse.ArgumentParser();p.add_argument("--pipeline",default="config/pipeline.yaml");p.add_argument("--sources",default="config/sources.yaml");a=p.parse_args()
 cfg=load_yaml(a.pipeline); manifest=load_yaml(a.sources); root=Path(cfg["project"]["raw_root"]).parent; out=ensure_dir(cfg["project"]["output_root"])/"canonical";out.mkdir(parents=True,exist_ok=True)
 tables=ingest_all(manifest,root); issues=[]
 for kind,df in tables.items():
  idcol={"posts":"post_id","comments":"comment_id","images":"image_id","brands":"brand_id"}.get(kind)
  if idcol and idcol in df:
   for value in df.loc[df[idcol].duplicated(keep=False),idcol].astype(str).unique(): issues.append({"table":kind,"record_id":value,"issue":"duplicate_id"})
  if kind=="posts":
   df=numeric(df,["likes","comments_count","shares","views","impressions","followers"])
   bad=df[[c for c in ["likes","comments_count","shares","views","impressions","followers"] if c in df]].lt(0).any(axis=1)
   for value in df.loc[bad,"post_id"].astype(str): issues.append({"table":kind,"record_id":value,"issue":"negative_metric"})
  df.to_parquet(out/f"{kind}.parquet",index=False)
 if not tables["comments"].empty and not tables["posts"].empty:
  missing=~tables["comments"].post_id.astype(str).isin(set(tables["posts"].post_id.astype(str)))
  for v in tables["comments"].loc[missing,"comment_id"].astype(str): issues.append({"table":"comments","record_id":v,"issue":"missing_parent_post"})
 pd.DataFrame(issues).to_csv(out.parent/"data_quality_report.csv",index=False)
 write_json(out.parent/"data_quality_report.json",{"issues":issues,"counts":{k:len(v) for k,v in tables.items()}})
 print({k:len(v) for k,v in tables.items()},"issues",len(issues))
if __name__=="__main__":main()
