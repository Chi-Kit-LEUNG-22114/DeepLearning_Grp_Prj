# Brand Momentum Weekly Encoder

This pipeline turns `../embedding_json_package/build_embedding_input.json` into
fixed-width weekly brand vectors. One output row is one brand × Monday-start
week. It encodes titles/comments with BGE, titles with CLIP's text tower, and
available thumbnails with CLIP's image tower. CLIP image/text blocks and BGE
blocks are pooled separately before concatenation.

## End-to-end workflow

Install and run the encoder against the existing aggregated JSON:

```bash
cd ../brand_momentum_encoder
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
```

The first embedding run downloads the configured BGE-M3 and CLIP models.
Set `models.offline: true` and make the model weights available locally for
offline operation.

## Main outputs

- `output/weekly/brand_week_features.parquet`: fixed-width weekly input vectors
  plus identifiers and optional Trends labels.
- `output/sequences/weekly_input_windows.npz`: unlabeled fixed-shape history
  windows for transfer learning (`X`, brand IDs, end dates, feature names).
- `output/sequences/{train,validation,test}.npz`: supervised windows only when
  a valid weekly Trends series is provided.
- `output/artifacts/`: training-period-only PCA/scaling artifacts.
- `output/feature_validation_report.json`: final data-quality checks.

The supplied JSON has no numeric Google Trends observations or target values.
This run leaves those fields untouched and produces unlabeled weekly vectors
and windows only. Comments after the post's input-week Sunday are excluded.
Video likes/views/comment counts are retained and summarized (sum, mean,
median, min, max, p90, and observed-count) as requested. Their snapshot time is
also retained as a late-snapshot count per week; most historical video counts
were retrieved after their publication week, so treat those count channels as
descriptive/test features—not valid historical forecasting covariates—until
time-matched snapshots are available.

Weekly embedding blocks use mean, engagement-weighted mean, standard deviation,
elementwise maximum, and elementwise minimum pooling. The 29 locally available
thumbnails are matched by video ID from `project.raw_root`; posts without
images retain explicit availability indicators.

The original unlabeled test output used an all-weeks fallback for sparse image
and comment PCA. That output must not be used for supervised evaluation. The
current config disables this fallback: sparse blocks are omitted unless they
have enough training-date vectors. The transfer project's `prepare_data.py`
passes its shared 70:10:20 training cutoff into `07_fit_feature_reducers.py`,
then runs `08_build_weekly_vectors.py`. Thus PCA and the encoder's numeric
scalers are fitted by this encoder package at the same cutoff used by both
transfer models.

## Configuration and alternate inputs

`config/pipeline.yaml` points to the aggregated JSON and defines encoders,
pooling, train-only reducers, and chronological split boundaries. The ingestion
adapter also supports source manifests through `--sources config/sources.yaml`
when `project.input_json` is removed. Edit `input_json` to use a different
aggregated JSON.

Weeks are Monday through Sunday in `Asia/Hong_Kong`. The pipeline does not
train or fine-tune the downstream forecasting model.
