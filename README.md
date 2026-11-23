# IMPORTANT
embedding_json_package contains the aggregated JSON file brand_embedding_input.json from the other 30 brands.zip

However, the embedding package and brand_momentum_encode are not aligned for encoding the information. Check the script before run.

# YouTube brand data to embedding JSON

## Run

```bash
python build_embedding_json.py \
  --shortlist shortlist_30_b.csv \
  --videos videos.csv \
  --comments comments.csv \
  --json-output brand_embedding_input.json \
  --excel-output brand_embedding_conversion.xlsx
```

## JSON hierarchy

```text
brands[]
  brand_id
  brand_name
  youtube_channel
  google_trends
  periods[]
    week_start
    week_end
    weekly_metrics
    videos[]
      video_id
      title
      metrics
      thumbnail
      comments[]
```

## Encoding and cleaning

- Input files are read with `utf-8-sig`, which accepts UTF-8 with or without a BOM.
- JSON is written as UTF-8 with `ensure_ascii=False`.
- Text uses Unicode NFC normalization.
- Emoji, invisible formatting marks, control characters, surrogates, private-use, and unassigned characters are removed.
- Multilingual letters are retained.
- URLs and user mentions in comment text are replaced by `[URL]` and `[USER]`.
- Source identifiers are preserved except for surrounding or invisible control characters.
- `brand_id` is a deterministic ASCII snake-case key; `brand_name` retains the cleaned display name.
- Missing counts stay `null`; they are not silently changed to zero.

## Period

Weeks are Monday through Sunday using `Asia/Hong_Kong` time.


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
