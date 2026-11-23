# Multimodal Brand-Week Content Encoding Plan

## 1. Purpose

This Phase 1 package converts scattered raw files containing brand-related images, YouTube thumbnails, titles, captions, descriptions, comments, hashtags/tags, engagement metrics, and Google Trends observations into a fixed-length numerical representation for each brand and Monday-to-Sunday week.

The output is model-neutral and can later feed IBM Granite Tiny Time Mixer, Chronos, Temporal Fusion Transformer, MOMENT, LightGBM, CatBoost, or a custom PyTorch forecaster.

The supervised target may be created without manual annotation:

\[
y_{b,t}=GT_{b,t+1}
\]

where `GT` is Google Trends, `b` is a brand, and `t` is a weekly observation period.

## 2. Approved design

### 2.1 Unit of analysis

One output row represents:

> one brand × one Monday-to-Sunday week

A week may contain no posts, one post, or thousands of posts. Variable-length content is encoded per item and then pooled into fixed-length weekly blocks.

### 2.2 Hierarchy

1. Discover scattered source files.
2. Normalize source-specific columns into canonical tables.
3. Validate IDs, timestamps, paths, types, and foreign keys.
4. Encode images/thumbnails.
5. Encode titles, captions/descriptions, tags, and comments.
6. Create one fixed-length vector per post.
7. Pool all post vectors for each brand-week.
8. Fit dimensionality reducers and scalers on training dates only.
9. Create compact weekly vectors and forecasting sequences.
10. Validate absence of temporal leakage.

## 3. Critical rule: embedding spaces are not automatically compatible

The system does **not** compare or average arbitrary embeddings from unrelated pretrained models.

- CLIP image embeddings and CLIP text embeddings share a trained contrastive space. Cosine similarity between these two is valid.
- BGE-M3 text embeddings occupy a different semantic space. A BGE-M3 vector must not be directly averaged with, subtracted from, or cosine-compared with a CLIP vector.
- Image and BGE-M3 text blocks remain separate through encoding and pooling.
- Separate PCA or projection artifacts are fitted per encoder and modality.
- The reduced modality blocks are concatenated only after modality-specific processing.
- If a learned shared space is required later, train explicit projection adapters with a contrastive or supervised objective. Phase 1 does not assume such compatibility.

The weekly state is therefore block-structured:

\[
U_{b,t}=[G_{b,t}; I_{b,t}; T_{b,t}; C_{b,t}; H_{b,t}; E_{b,t}; A_{b,t}]
\]

where:

- `G`: Google Trends and temporal features
- `I`: pooled CLIP image features
- `T`: pooled BGE-M3 title/caption features
- `C`: pooled BGE-M3 comment features
- `H`: pooled BGE-M3 tag features
- `E`: engagement and volume features
- `A`: availability and quality indicators

## 4. Pretrained models

Default encoders are configurable.

### Image and image-text alignment

- Default: `openai/clip-vit-base-patch32`
- Use CLIP image embeddings for images and thumbnails.
- Use the matching CLIP text tower only for image-caption alignment.
- L2-normalize CLIP embeddings before cosine similarity.

### General text semantics

- Default: `BAAI/bge-m3`
- Encode titles, captions/descriptions, comments, and tags separately.
- Preserve the encoder identity and revision in every artifact.
- Never mix raw BGE-M3 coordinates with CLIP coordinates.

### Optional sentiment

A configurable multilingual sentiment model may generate probabilities. Sentiment inference is optional and must be treated as model-generated features, not ground truth.

## 5. Scattered-data intake

Raw data are not yet finalized. The ingestion layer is deliberately general.

### Supported source types

- CSV
- TSV
- JSON
- JSON Lines
- Parquet
- local image trees

### Discovery

The raw-data manifest accepts glob patterns and source-specific column mappings. Example:

```yaml
sources:
  - name: youtube_posts
    kind: posts
    paths: ["raw/youtube/**/*.csv"]
    mapping:
      video_id: post_id
      channel_brand: brand_id
      published_at: created_at
      title: title
      description: caption
      thumbnail_path: image_path
      view_count: views
      like_count: likes
      comment_count: comments_count
  - name: instagram_posts
    kind: posts
    paths: ["raw/instagram/**/posts.csv"]
    mapping:
      shortcode: post_id
      owner_username: brand_id
      uploaded_at_utc: created_at
      caption: caption
      image_files: image_path
```

Unknown columns are retained in source snapshots but excluded from canonical outputs unless mapped.

### Canonical tables

The ingestion layer creates:

- `posts`: one row per post/video
- `comments`: one row per comment
- `images`: one row per image/thumbnail
- `trends`: one row per brand-date
- `brands`: one row per brand

Required canonical identifiers are generated deterministically when source IDs are absent, using source name plus a hash of stable fields.

## 6. Human preparation requirements

The minimum human work is:

1. Place all raw files under one raw directory. Existing nested folders may remain unchanged.
2. Do not rename or merge raw files manually unless necessary.
3. Create `config/sources.yaml` and add one entry per source family.
4. Map source columns to canonical names.
5. Confirm timezone, brand mapping, and week convention.
6. Keep images locally and map image paths; URLs may remain metadata but are not downloaded by Phase 1.
7. Preserve original text. Do not remove emojis, hashtags, punctuation, or negation.
8. Record the engagement observation cutoff where available.
9. Do not mix lifetime engagement with 24-hour or 7-day engagement without an explicit column.
10. Run validation before encoding.

When actual files arrive, update only the source manifest and, if necessary, source adapters. Encoder and aggregation logic should not need rewriting.

## 7. Text preprocessing

For titles, captions, descriptions, tags, and comments:

- Unicode normalization
- HTML entity decoding
- replace URLs with `[URL]`
- replace user mentions with `[USER]`
- preserve emojis and hashtags
- collapse repeated whitespace
- preserve punctuation and negation
- recognize deleted/removed/null strings as missing
- chunk text that exceeds the encoder limit
- pool chunk embeddings into one item embedding

Text fields remain separate because titles, captions, comments, and tags have different functions and distributions.

## 8. Image preprocessing

For JPEG, PNG, and WebP:

- verify file existence and readability
- apply EXIF orientation
- convert to RGB
- use the selected model processor for resize and normalization
- do not use random augmentation during feature extraction
- hash image bytes for caching
- represent missing images with a zero vector plus `image_available=0`

Multiple images in one post are encoded individually and mean-pooled to one post-image vector. Image count and intra-post visual dispersion are retained.

## 9. Comment hierarchy

Comments use two-level pooling.

### Within a post

1. Encode each comment with the text encoder.
2. Create mean, engagement-weighted mean, and standard-deviation summaries.
3. Retain count, sentiment, reply-depth, and comment-velocity statistics.

Comment weights default to:

\[
w_j=1+\log(1+likes_j)
\]

### Across the week

Pool the post-level comment vectors across all posts for the brand-week. This prevents a post with thousands of comments from becoming thousands of separate weekly observations.

## 10. Weekly pooling

For each modality, available item embeddings can be summarized by:

- mean
- engagement-weighted mean
- standard deviation
- optional elementwise maximum
- optional elementwise minimum
- optional 10th and 90th percentiles

Default engagement weight:

\[
w_i=1+\log(1+likes_i+2\,comments_i+3\,shares_i)
\]

The coefficients are configurable because platform interactions have different meanings.

Use both unweighted and weighted means:

- unweighted mean represents average conversation
- weighted mean represents conversation receiving more traction

## 11. Numerical features

### Google Trends

Retain:

- current score
- lags 1, 2, 4, 8, 13, 26, 52
- differences 1 and 4
- 4-week percentage change
- rolling means 4 and 13
- rolling standard deviations 4 and 13
- slopes 4 and 13
- acceleration

Only trailing windows are permitted.

### Engagement

Retain:

- `log1p` totals for likes, comments, shares, views, impressions, and post count
- median and 90th percentile post engagement
- engagement rate by impressions/views/followers when denominators exist
- top-10%-post engagement concentration
- platform mix
- brand-owned and paid proportions when known

Numerical features are standardized with training-period statistics. The entire concatenated vector is not L2-normalized.

## 12. Dimensionality reduction

Separate PCA models are fitted for separate encoder/modality blocks:

- CLIP images
- BGE-M3 titles
- BGE-M3 captions/descriptions
- BGE-M3 comments
- BGE-M3 tags

PCA, scalers, clipping thresholds, imputers, and categorical vocabularies are fitted on training dates only.

Default compact dimensions:

| Block | Default components |
|---|---:|
| Images | 16 |
| Titles | 8 |
| Captions/descriptions | 16 |
| Comments | 16 |
| Tags | 8 |

These are starting values and must be revised after real-data volume and explained variance are known.

## 13. Forecasting output

For a context length `L`, each sample is:

\[
X_{b,t}=[U_{b,t-L+1},\ldots,U_{b,t}]
\]

with label:

\[
y_{b,t}=GT_{b,t+1}
\]

Generic tensor shape:

- `X`: `[batch, context_length, channels]`
- `y`: `[batch, horizon]`

Long-form output contains:

- `brand_id`
- `timestamp`
- `target`
- historical numerical covariates
- static brand attributes

## 14. Temporal leakage controls

- common chronological train/validation/test boundaries
- comments after the input-week cutoff are excluded
- engagement after the configured measurement cutoff is excluded
- no centered rolling windows
- no random row split
- preprocessing artifacts fitted on training dates only
- next-week labels verified by calendar continuity, not blind row shift
- raw ingestion snapshots are immutable

## 15. Scripts

1. `01_validate_raw_data.py` — discover, canonicalize, validate, and report scattered raw files
2. `02_encode_images.py` — CLIP image/thumbnail embeddings
3. `03_encode_text.py` — separate title, caption, and tag BGE-M3 embeddings plus CLIP caption embeddings for alignment
4. `04_encode_comments.py` — comment encoding and within-post pooling
5. `05_build_post_vectors.py` — join post-level modality blocks
6. `06_pool_brand_weeks.py` — variable posts to one brand-week representation
7. `07_fit_feature_reducers.py` — training-only PCA/scaling artifacts
8. `08_build_weekly_vectors.py` — compact final weekly feature table
9. `09_build_forecasting_sequences.py` — arrays and long-form forecasting data
10. `10_validate_features.py` — leakage, shape, continuity, and quality checks

## 16. Phase 1 limitations

- Raw input mappings will need revision after actual data inspection.
- Phase 1 does not claim that CLIP and BGE-M3 share a latent space.
- Phase 1 does not train end-to-end cross-modal attention.
- Phase 1 does not download remote images.
- Phase 1 does not train the downstream forecasting model.
- Pretrained model outputs can contain domain bias; model-generated sentiment/topics are not labels.
- Google Trends normalization must use consistent geography, category, search type, and extraction protocol.

## 17. Definition of done

Phase 1 is complete when:

- scattered files can be mapped by configuration
- canonical tables are produced with provenance
- invalid rows are reported rather than silently discarded
- embeddings are cached and reproducible
- unrelated embedding spaces remain isolated
- multiple posts become one fixed-length brand-week vector
- weeks without posts remain represented
- train-only preprocessing is enforced
- sequences and long-form outputs can be consumed by downstream forecasters
- all final features have a feature dictionary and run manifest
