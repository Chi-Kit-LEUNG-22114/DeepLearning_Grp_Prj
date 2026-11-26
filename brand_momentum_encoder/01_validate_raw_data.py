#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from brand_encoder.common import load_yaml,ensure_dir,write_json,numeric
from brand_encoder.ingestion import ingest_all,ingest_embedding_json

def main():
 p=argparse.ArgumentParser();p.add_argument("--pipeline",default="config/pipeline.yaml");p.add_argument("--sources",default="config/sources.yaml");a=p.parse_args()
 cfg=load_yaml(a.pipeline); base=Path(a.pipeline).resolve().parent
 input_json=cfg["project"].get("input_json")
 if input_json:
  source=Path(input_json); source=source if source.is_absolute() else (base/source).resolve()
  image_root=Path(cfg["project"].get("raw_root","raw"))
  image_root=image_root if image_root.is_absolute() else (base/image_root).resolve()
  tables=ingest_embedding_json(source,image_root)
 else:
  manifest=load_yaml(a.sources); raw_root=Path(cfg["project"]["raw_root"])
  raw_root=raw_root if raw_root.is_absolute() else (base/raw_root).resolve()
  root=raw_root.parent
  tables=ingest_all(manifest,root)
 out=ensure_dir(cfg["project"]["output_root"])/"canonical";out.mkdir(parents=True,exist_ok=True); issues=[]
 for kind,df in tables.items():
  idcol={"posts":"post_id","comments":"comment_id","images":"image_id","brands":"brand_id"}.get(kind)
  if idcol and idcol in df:
   for value in df.loc[df[idcol].isna(),idcol].index.astype(str): issues.append({"table":kind,"record_id":value,"issue":"missing_id"})
   for value in df.loc[df[idcol].duplicated(keep=False),idcol].astype(str).unique(): issues.append({"table":kind,"record_id":value,"issue":"duplicate_id"})
  if kind=="posts":
   df=numeric(df,["likes","comments_count","shares","views","impressions","followers"])
   bad=df[[c for c in ["likes","comments_count","shares","views","impressions","followers"] if c in df]].lt(0).any(axis=1)
   for value in df.loc[bad,"post_id"].astype(str): issues.append({"table":kind,"record_id":value,"issue":"negative_metric"})
   if "created_at" in df:
    invalid=pd.to_datetime(df.created_at,utc=True,errors="coerce").isna()
    for value in df.loc[invalid,"post_id"].astype(str): issues.append({"table":kind,"record_id":value,"issue":"invalid_timestamp"})
   if "brand_id" in df:
    for value in df.loc[df.brand_id.isna(),"post_id"].astype(str): issues.append({"table":kind,"record_id":value,"issue":"missing_brand_id"})
  df.to_parquet(out/f"{kind}.parquet",index=False)
 if not tables["comments"].empty and not tables["posts"].empty:
  missing=~tables["comments"].post_id.astype(str).isin(set(tables["posts"].post_id.astype(str)))
  for v in tables["comments"].loc[missing,"comment_id"].astype(str): issues.append({"table":"comments","record_id":v,"issue":"missing_parent_post"})
 pd.DataFrame(issues,columns=["table","record_id","issue"]).to_csv(out.parent/"data_quality_report.csv",index=False)
 write_json(out.parent/"data_quality_report.json",{"issues":issues,"counts":{k:len(v) for k,v in tables.items()}})
 print({k:len(v) for k,v in tables.items()},"issues",len(issues))
 if issues: raise SystemExit(1)
if __name__=="__main__":main()
