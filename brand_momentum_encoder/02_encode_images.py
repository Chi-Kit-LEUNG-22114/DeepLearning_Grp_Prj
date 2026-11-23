#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from tqdm import tqdm
from brand_encoder.common import load_yaml,ensure_dir,select_device,file_sha256
from brand_encoder.encoders.image_encoder import ClipEncoder

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config/pipeline.yaml");a=p.parse_args();cfg=load_yaml(a.config)
 out=Path(cfg["project"]["output_root"]); images=pd.read_parquet(out/"canonical/images.parquet"); root=Path(cfg["project"]["raw_root"]).parent
 enc=ClipEncoder(cfg["models"]["image_encoder"],select_device(cfg["models"]["device"])); rows=[]
 for _,r in tqdm(images.iterrows(),total=len(images)):
  path=Path(str(r.image_path)); path=path if path.is_absolute() else root/path
  base={"image_id":r.image_id,"post_id":r.post_id,"image_path":str(path),"encoder":enc.model_name}
  try:v=enc.image(path);base.update({"available":1,"sha256":file_sha256(path),"embedding":v.tolist()})
  except Exception as e:base.update({"available":0,"error":str(e),"embedding":np.zeros(enc.dimension,np.float32).tolist()})
  rows.append(base)
 dest=ensure_dir(out/"embeddings")/"images.parquet";pd.DataFrame(rows).to_parquet(dest,index=False);print(dest)
if __name__=="__main__":main()
