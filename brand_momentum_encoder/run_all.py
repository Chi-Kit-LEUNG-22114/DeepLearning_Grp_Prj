#!/usr/bin/env python3
import argparse,subprocess,sys
SCRIPTS=["01_validate_raw_data.py","02_encode_images.py","03_encode_text.py","04_encode_comments.py","05_build_post_vectors.py","06_pool_brand_weeks.py","07_fit_feature_reducers.py","08_build_weekly_vectors.py","09_build_forecasting_sequences.py","10_validate_features.py"]
def main():
 p=argparse.ArgumentParser();p.add_argument("--pipeline",default="config/pipeline.yaml");p.add_argument("--sources",default="config/sources.yaml");a=p.parse_args()
 for s in SCRIPTS:
  cmd=[sys.executable,s,"--config",a.pipeline] if not s.startswith("01_") else [sys.executable,s,"--pipeline",a.pipeline,"--sources",a.sources]
  print("RUN",*cmd);subprocess.run(cmd,check=True)
if __name__=="__main__":main()
