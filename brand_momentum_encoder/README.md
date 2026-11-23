# Brand Momentum Multimodal Encoder

Read `ENCODING_PLAN.md` first.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp config/sources.example.yaml config/sources.yaml
```

Edit `config/sources.yaml` to map scattered source columns to canonical names. Then run:

```bash
python 01_validate_raw_data.py --pipeline config/pipeline.yaml --sources config/sources.yaml
python 02_encode_images.py --config config/pipeline.yaml
python 03_encode_text.py --config config/pipeline.yaml
python 04_encode_comments.py --config config/pipeline.yaml
python 05_build_post_vectors.py --config config/pipeline.yaml
python 06_pool_brand_weeks.py --config config/pipeline.yaml
python 07_fit_feature_reducers.py --config config/pipeline.yaml
python 08_build_weekly_vectors.py --config config/pipeline.yaml
python 09_build_forecasting_sequences.py --config config/pipeline.yaml
python 10_validate_features.py --config config/pipeline.yaml
```

Or use `python run_all.py` after creating `config/sources.yaml`.

## Important

The code is an approved Phase 1 scaffold. Actual source mappings, timestamp conventions, engagement cutoffs, and source-specific parsing must be updated after inspecting real files. CLIP and BGE-M3 embedding spaces are intentionally kept separate.
