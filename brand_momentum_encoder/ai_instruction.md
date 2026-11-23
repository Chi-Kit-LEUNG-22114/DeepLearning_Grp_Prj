# Build the Brand Momentum Encoding Pipeline

## Objective

Build a modular Python pipeline that converts `brand_embedding_input_with_comments.json` into fixed-length weekly multimodal vectors for brand-momentum forecasting.

The input hierarchy is:

```text
brands
└── periods
    └── videos
        └── comments
```

One output row must represent **one brand during one Monday-to-Sunday week**.

## 1. Text preparation

Before embedding:

- Read and write files with UTF-8.
- Apply Unicode NFC normalization.
- Remove emoji, variation selectors, zero-width characters, directional controls, and invalid control characters.
- Preserve legitimate multilingual text and punctuation.
- Normalize repeated whitespace.
- Keep missing text as `null`.
- Do not translate, stem, or remove stop words.

## 2. Text embeddings

Use frozen `BAAI/bge-m3` embeddings for:

- `videos[].title`
- `comments[].text`

Do not encode blank text. Use a zero vector and availability indicator instead.

Create:

```text
title_embedding
comment_embedding
title_available
comment_text_available
```

Pool comments in two stages:

1. Pool comment embeddings within each video.
2. Pool video-level comment embeddings within each brand-week.

Use:

- mean;
- like-weighted mean;
- standard deviation.

Comment weight:

\[
w_j=1+\log(1+\text{like_count}_j)
\]

Use the top-level `comments[]` array for batch encoding and join comments to videos with `video_id`. Do not encode duplicate nested copies.

## Comment handling

Not every video has collected comments. Every video must still receive a fixed-size comment feature block.

For videos with usable comment text:

1. Encode each unique `comments[].text` with frozen `BAAI/bge-m3`.
2. Pool comment embeddings within the video using:
   - mean;
   - like-weighted mean;
   - standard deviation.
3. Use comment weight:

\[
w_j=1+\log(1+\text{like_count}_j)
\]

For videos without usable comment text:

- use zero vectors for all comment embedding blocks;
- set availability indicators explicitly;
- do not interpret zero vectors as neutral sentiment.

Create these video-level fields:

```text
comment_records_available
comment_text_available
comment_count_in_file
comment_text_count
reported_comment_count
comment_capture_ratio
comment_data_status
comment_mean_embedding
comment_weighted_embedding
comment_std_embedding
```

Set `comment_data_status` using:

```text
reported = 0 and collected = 0
    → no_reported_comments

reported > 0 and collected = 0
    → not_collected

reported > 0 and collected / reported >= 0.8
    → complete_or_near_complete

reported > 0 and 0 < collected / reported < 0.8
    → partial

reported is missing
    → unknown
```

At weekly level:

- pool semantic comment vectors using only videos with usable comment text;
- do not include missing-comment zero vectors in semantic means;
- retain zero vectors only as fixed-shape placeholders;
- create separate comment-coverage features.

Required weekly coverage fields:

```text
weekly_video_count
weekly_videos_with_comments
weekly_videos_with_comment_text
weekly_comment_coverage
weekly_comment_text_coverage
weekly_comment_records
weekly_reported_comments
weekly_comment_capture_ratio
```

This must distinguish:

- no comments reported;
- comments reported but not collected;
- comment records with blank text;
- partially collected comments;
- usable comment text.


## 3. Image embeddings

Use the image tower from:

```text
openai/clip-vit-base-patch32
```

Encode:

- `videos[].thumbnail.local_path`
- `youtube_channel.channel_thumbnail_path`

For missing images:

```text
image_embedding = zero vector
image_available = 0
```

Use the matching CLIP text tower to encode each video title only for image-title alignment:

\[
alignment_i=
\cos(
e^{CLIP-image}_i,
e^{CLIP-title}_i
)
\]

Do not compare or average CLIP embeddings with BGE-M3 embeddings. The embedding spaces are incompatible and must remain separate.

## 4. Numerical features

Do not semantically embed numerical values.

Transform these with `log1p` and training-fitted scaling:

```text
view_count
like_count
comment_count_reported
comments_in_file
duration_seconds
tags_count
weekly video_count
weekly view_count_total
weekly like_count_total
weekly comments_in_file_total
```

Retain missing-value indicators. Do not convert unavailable values into genuine zeros.

## 5. Grouping keys

Use only for joining and grouping:

```text
brand_id
week_start
week_end
video_id
comment_id
parent_id
channel_id
published_at_utc
```

Do not include IDs as ordinary numerical model inputs.

## 6. Weekly pooling

For each brand-week, pool title, comment, and image vectors separately using:

- mean;
- engagement-weighted mean;
- standard deviation.

Video weight:

\[
w_i=
1+\log(
1+\text{likes}_i+
2\text{comments}_i+
0.001\text{views}_i
)
\]

Keep separate weekly feature blocks:

```text
image_features
title_features
comment_features
engagement_features
availability_features
calendar_features
```

Do not L2-normalize the complete concatenated vector.

## 7. Dimensionality reduction

Fit separate PCA models for:

- CLIP image embeddings;
- BGE-M3 title embeddings;
- BGE-M3 comment embeddings.

Fit PCA and numerical scalers using training dates only.

Never fit preprocessing objects on validation or test data.

## 8. Required outputs

Create:

```text
output/post_embeddings.parquet
output/comment_embeddings.parquet
output/video_comment_vectors.parquet
output/brand_week_vectors.parquet
output/feature_dictionary.csv
output/artifacts/
```

`brand_week_vectors.parquet` must contain one row per `brand_id` and `week_start`.

Also create forecasting sequences:

```text
X shape: [samples, context_weeks, feature_channels]
y shape: [samples, forecast_horizon]
```

Use next-week Google Trends as the future target when weekly Trends data become available:

\[
y_{b,t}=GT_{b,t+1}
\]

## 9. Engineering requirements

- Use Python 3.11.
- Use PyTorch, Transformers, Sentence Transformers, pandas, NumPy, PyArrow, scikit-learn, and Pillow.
- Support CPU and CUDA.
- Process embeddings in configurable batches.
- Cache embeddings using content hashes.
- Preserve model name and revision in output metadata.
- Use reproducible random seeds.
- Provide command-line interfaces.
- Add clear logging and row-level error reports.
- Never silently discard failed records.

Implement the pipeline as small modules rather than one script. Include a README, configuration file, and pytest tests for Unicode cleaning, missing text, missing images, duplicate IDs, weekly pooling, PCA leakage, and output shapes.