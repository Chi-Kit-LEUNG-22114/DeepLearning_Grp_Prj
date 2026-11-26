#!/usr/bin/env python3
import argparse,subprocess,sys
from pathlib import Path
SCRIPTS=["01_validate_raw_data.py","02_encode_images.py","03_encode_text.py","04_encode_comments.py","05_build_post_vectors.py","06_pool_brand_weeks.py","07_fit_feature_reducers.py","08_build_weekly_vectors.py","09_build_forecasting_sequences.py","10_validate_features.py"]
def main():
 root=Path(__file__).resolve().parent
 p=argparse.ArgumentParser();p.add_argument("--pipeline",default=str(root/"config/pipeline.yaml"));p.add_argument("--sources",default=str(root/"config/sources.yaml"));a=p.parse_args()
 pipeline=Path(a.pipeline).resolve();sources=Path(a.sources).resolve()
 for s in SCRIPTS:
  script=root/s
  cmd=[sys.executable,str(script),"--config",str(pipeline)] if not s.startswith("01_") else [sys.executable,str(script),"--pipeline",str(pipeline),"--sources",str(sources)]
  print("RUN",*cmd);subprocess.run(cmd,check=True,cwd=root)
if __name__=="__main__":main()
