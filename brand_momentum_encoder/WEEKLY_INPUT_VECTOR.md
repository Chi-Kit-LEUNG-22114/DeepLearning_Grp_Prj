# What One Weekly Input Vector Contains

## Current run at a glance

The encoder writes **one row per brand per Monday-start week**. The current
weekly feature table has **5,024 rows**: 32 brands across 157 calendar weeks.
Each row contains **393 numeric input features**.

The row is conceptually:

```text
[brand_id, week_start | 393 numeric features]
```

`brand_id` and `week_start` are metadata, not numeric model channels. The
unlabeled sequence file stores each history window as
`X[brand, 52 weeks, 393 features]`; the current output has 3,392 such windows.

## Important: label and Google Trends are not in this vector

The input JSON has no numeric weekly Google Trends values and no available
target-label values. The encoder did **not** insert either one, and it did not
rewrite those JSON sections. In particular, the current vector does **not**
start with a label or a Google Trends value:

```text
target label:       not present
Google Trends:      not present
daily search rate:  not present
```

The supervised `train`, `validation`, and `test` NPZ files consequently have
zero labelled samples. `weekly_input_windows.npz` contains the unlabeled `X`
windows.

## Exact feature order and occupied dimensions

The ranges below are **1-based positions** in the 393-value feature vector.
They match `output/artifacts/feature_manifest.json` and the saved NPZ
`feature_names`.

| Positions | Block | Width |
|---:|---|---:|
| 1–80 | BGE comment vectors pooled from per-post comment means | 80 |
| 81–160 | BGE comment vectors pooled from per-post comment-weighted means | 80 |
| 161–224 | CLIP image/thumbnail embeddings | 64 |
| 225–264 | BGE title embeddings | 40 |
| 265–344 | CLIP title embeddings | 80 |
| 345–383 | Engagement/count numerical summaries | 39 |
| 384 | `log1p(post_count)` | 1 |
| 385–389 | Modality availability flags | 5 |
| 390–392 | Image–text alignment summaries | 3 |
| 393 | Social data available flag | 1 |
| **Total** |  | **393** |

### Embedding block layout

Each embedding block is ordered by pooling statistic as shown below; PCA
components within each statistic are numbered from 001 to the block's
configured width.

| Positions | Embedding block | Statistic order and reduced width |
|---:|---|---|
| 1–80 | `comment_mean` (BGE, 1,024 raw dimensions) | max × 16, mean × 16, min × 16, std × 16, engagement-weighted mean × 16 |
| 81–160 | `comment_weighted_mean` (BGE, 1,024 raw dimensions) | max × 16, mean × 16, min × 16, std × 16, engagement-weighted mean × 16 |
| 161–224 | `image_embedding` (CLIP, 512 raw dimensions) | max × 16, mean × 16, min × 16, engagement-weighted mean × 16 |
| 225–264 | `title_bge` (BGE, 1,024 raw dimensions) | max × 8, mean × 8, min × 8, std × 8, engagement-weighted mean × 8 |
| 265–344 | `title_clip` (CLIP, 512 raw dimensions) | max × 16, mean × 16, min × 16, std × 16, engagement-weighted mean × 16 |

The image `std` block is not present: its pooled values had zero variance in
this data and the reducer was skipped. Only 29 thumbnail images were available;
weeks without an image have zeroed reduced image channels plus availability
flags.

The vector column names make this explicit. For example,
`title_clip__max__pca_001` is component 1 of the PCA-reduced elementwise maximum
of the week's CLIP title embeddings; `title_clip__max__pca_016` is component 16
of that same block.

### Count and numerical block (positions 345–383)

These scalar features are each log-transformed with `log1p` and standardized
using training-period statistics. Within each metric, the order is
`total`, `observed_posts`, `mean`, `median`, `min`, `max`, `p90`.

| Positions | Feature(s) |
|---:|---|
| 345 | `post_count__scaled` |
| 346 | `engagement_weight_total__scaled` |
| 347 | `metrics_snapshot_late_posts__scaled` |
| 348 | `comments_encoded_total__scaled` |
| 349–355 | Likes: total, observed-post count, mean, median, min, max, p90 |
| 356–362 | Reported comment counts: total, observed-post count, mean, median, min, max, p90 |
| 363–369 | Shares: total, observed-post count, mean, median, min, max, p90 |
| 370–376 | Views: total, observed-post count, mean, median, min, max, p90 |
| 377–383 | Impressions: total, observed-post count, mean, median, min, max, p90 |

Shares and impressions are unavailable in this input; their channels remain
present but are zero-valued after preprocessing. Like, reported-comment, and
view counts are retained as requested.

### Availability and alignment (positions 385–393)

- 385–389: image, title, caption, tag, and comment availability, respectively
  (`0` or `1`).
- 390–391: mean and standard deviation of CLIP image–title cosine alignment.
- 392: alignment available (`0` or `1`).
- 393: at least one social post exists in the brand-week (`0` or `1`).

## Dimensionality reduction used

**PCA is used independently for each embedding modality and pooling statistic**;
the model does not combine CLIP and BGE coordinates in one PCA. The current
reductions are:

- CLIP image: 512 → 16 per retained pooling statistic.
- CLIP title: 512 → 16 per statistic.
- BGE title: 1,024 → 8 per statistic.
- BGE comments: 1,024 → 16 per statistic.

The original unlabeled test table used a configured all-weeks fallback for
sparse thumbnail/comment reducers. That historical table is not suitable for
supervised evaluation. The current config disables the fallback. The transfer
project computes the shared chronological training cutoff, invokes this
encoder's reducer and weekly-vector stages with that cutoff, and then consumes
the resulting reduced table. Sparse blocks with insufficient training-date
examples are omitted rather than fitted on validation/test weeks.

## Known duplication and timing caveats

- Comment likes are not historical snapshots in this input, so comments are
  pooled with unit weights. Consequently, per-post `comment_mean` and
  `comment_weighted_mean` are identical, and the two 80-channel comment blocks
  are duplicate representations. They are kept in this test output to reflect
  the current pipeline structure.
- Video metrics are retained even though 4,069 of 4,102 metric snapshots were
  recorded after their video's publication week. The late-snapshot count is
  included for visibility. These count channels are useful for this requested
  encoding test, but should not be treated as historically available
  forecasting inputs until time-aligned snapshots exist.
- Captions and tags are absent from this input, so their availability flags are
  zero and they contribute no embedding block.

For the authoritative generated order, see
`output/artifacts/feature_manifest.json`. The PCA dimensions and fit scope are
recorded in `output/artifacts/reducer_manifest.json`.
