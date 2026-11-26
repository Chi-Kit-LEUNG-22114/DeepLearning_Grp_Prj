#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from tqdm import tqdm
from brand_encoder.common import load_yaml,ensure_dir,select_device,file_sha256
from brand_encoder.encoders.image_encoder import ClipEncoder

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config/pipeline.yaml");a=p.parse_args();cfg=load_yaml(a.config)
 out=Path(cfg["project"]["output_root"]); images=pd.read_parquet(out/"canonical/images.parquet")
 root=Path(cfg["project"].get("raw_root","raw"));root=root if root.is_absolute() else (Path(a.config).resolve().parent/root).resolve()
 if images.empty:
  dest=ensure_dir(out/"embeddings")/"images.parquet"
  pd.DataFrame(columns=["image_id","post_id","image_path","encoder","available","sha256","error","embedding"]).to_parquet(dest,index=False)
  print(dest);return
 enc=ClipEncoder(cfg["models"]["image_encoder"],select_device(cfg["models"]["device"]),cfg["models"].get("offline",False)); rows=[]
 for _,r in tqdm(images.iterrows(),total=len(images)):
  path=Path(str(r.image_path)); path=path if path.is_absolute() else root/path
  if not path.exists():
   candidates=list(root.rglob(path.name)) if root.exists() else []
   if not candidates and str(r.post_id): candidates=list(root.rglob(f"*{r.post_id}*"))
   if candidates:path=candidates[0]
  base={"image_id":r.image_id,"post_id":r.post_id,"image_path":str(path),"encoder":enc.model_name}
  try:v=enc.image(path);base.update({"available":1,"sha256":file_sha256(path),"embedding":v.tolist()})
  except Exception as e:base.update({"available":0,"error":str(e),"embedding":np.zeros(enc.dimension,np.float32).tolist()})
  rows.append(base)
 dest=ensure_dir(out/"embeddings")/"images.parquet"
 columns=["image_id","post_id","image_path","encoder","available","sha256","error","embedding"]
 pd.DataFrame(rows,columns=columns).to_parquet(dest,index=False);print(dest)
if __name__=="__main__":main()
