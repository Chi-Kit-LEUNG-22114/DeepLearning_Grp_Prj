# Multimodal Brand Momentum Transfer Learning

This project compares two pretrained time-series backbones—IBM Granite TinyTimeMixer (TTM) and Chronos-2—on one task:

> Given the recent weekly multimodal input for a brand, predict its provided future Brand Momentum Index (BMI) as one continuous scalar.

This is **regression**, not classification. No BMI formula is implemented here. BMI values must be supplied in a label file and paired with their input cutoff. The target may use any numeric scale; the target scaler is fitted on training labels and its inverse is applied to saved predictions.

Training and inference are deliberately **not run** during project setup. The current encoder output has no BMI labels, so there are no supervised samples to train on yet.

## Project layout

```text
brand_momentum_transfer_learning/
├── README.md
├── requirements.txt
├── config/default.yaml
├── data/labels/                 # Put the BMI label table here
├── src/brand_momentum_transfer/
│   ├── data.py                  # Join, chronological split, train-only scaling, windows
│   ├── training.py              # Shared train/evaluate/checkpoint loop
│   ├── inference.py             # Reuse checkpoints on new weekly feature tables
│   └── models/
│       ├── common.py            # Per-modality adapters and scalar regression head
│       ├── ttm.py               # TTM r2.1 transfer-learning model
│       └── chronos2.py          # Chronos-2 encoder transfer-learning model
├── prepare_data.py
├── train_compare.py
├── predict.py
└── outputs/
    ├── shared/                  # One immutable set of splits and preprocessing
    ├── models/ttm/              # TTM checkpoint and predictions
    ├── models/chronos2/         # Chronos-2 checkpoint and predictions
    └── comparison/              # Side-by-side metrics
```

## 1. Expected input data

### Weekly feature data

The transfer project consumes the **reduced** encoder output:

```text
../brand_momentum_encoder/output/weekly/brand_week_features.parquet
```

PCA stays entirely in `brand_momentum_encoder` (`07_fit_feature_reducers.py` and `08_build_weekly_vectors.py`). The transfer code does not read unreduced embeddings, fit PCA, or project embedding blocks. It consumes the encoder's final feature table, including its brand/week calendar and reduced modality blocks.

The encoder currently reports 393 channels; the final count can be lower if a modality has too few training examples and its encoder-side reducer is omitted. `brand_id` and `week_start` are metadata, not model features. The encoder's reducer manifest records each fitted block and fit cutoff.

The project does **not** use the encoder's old `target_gt_t_plus_1` label. BMI is the sole supervised target.

### BMI labels

Create `data/labels/bmi_labels.csv` (or Parquet) with these required columns:

| Column | Meaning |
|---|---|
| `brand_id` | Same brand key as the weekly feature table |
| `input_week_start` | Monday-start week for the end of the input history |
| `bmi` | Numeric future BMI value associated with that input cutoff |

`target_period` is optional but recommended. It records the future period represented by the BMI label for auditing; split assignment remains based on `input_week_start` because the horizon can vary or be unknown. Each label must still be paired with its input cutoff. Duplicate `(brand_id, input_week_start)` labels are rejected.

If the source uses different names, change the column names in `config/default.yaml`. If BMI is only available as a brand-level value with no input-week pairing, the labels need to be paired before this training pipeline can form valid examples.

## 2. Shared 70:10:20 comparison split

The split is created **once** from the BMI labels by `prepare_data.py` and reused by both models. By default, that command also calls the encoder's existing PCA and vector-building scripts—not a transfer-side PCA implementation:

1. Calculate the chronological split and its training cutoff from the labels.
2. Call `brand_momentum_encoder/07_fit_feature_reducers.py --train-end <shared-train-end>`.
3. Call `brand_momentum_encoder/08_build_weekly_vectors.py` to refresh the final reduced feature table.
4. Validate the encoder manifests' fit scope/cutoff, then create shared model arrays.

The encoder config disables its all-weeks unlabeled PCA fallback. Sparse embedding blocks are omitted rather than fitted using validation/test weeks. The existing `brand_week_unreduced.parquet` must already be available; the preparation command does not rerun image/text encoders or raw-data ingestion.

The date partition sorts unique `input_week_start` dates: earliest 70% train, next 10% validation, final 20% test. `target_period`, when supplied, is retained for auditing but does not change split assignment because its horizon can vary or be unknown. The same cutoffs apply to every brand and both models. Each sample uses the 52 consecutive reduced weekly vectors ending at its input cutoff; earlier history may be shared across a boundary, as in forward prediction.

The date boundaries, labelled sample IDs, feature order, scaler statistics, and split basis are saved under `outputs/shared/`. Encoder PCA/scaler manifests must match the shared training cutoff or preparation stops. Neither model independently resplits or refits preprocessing.

Transfer-side preprocessing is limited to:

- Training-only median imputation and standardization of the encoder's already-reduced feature columns.
- BMI: training-only mean/standard-deviation scaling; predictions are inverse-scaled before reporting.
- Feature columns with no variation in the training data are retained but safely scaled.
- Non-finite values, duplicate labels, missing context weeks, and empty train/validation/test splits produce explicit errors.

## 3. Multimodal input handling

The encoder's reduced channels are grouped by name into separate blocks:

- BGE comment embeddings
- CLIP image embeddings
- BGE title embeddings
- CLIP title embeddings
- remaining numeric engagement/count inputs
- availability, image–text alignment, and other quality indicators

Each present block has its own learned adapter. The resulting low-width latent series are combined with the temporal backbone. CLIP and BGE coordinates remain separate as produced by the encoder; the transfer project does not average or compare those spaces directly.

Absent-modality and quality flags are retained. Empty/constant blocks are handled from the feature manifest rather than assumed to exist.

### Leakage protection

The encoder notes that most engagement snapshots were collected after publication. Therefore, the default configuration excludes the late-snapshot indicator, engagement-weight totals, likes, reported comments, shares, views, and impressions from model inputs. It also excludes engagement-weighted embedding summaries and the `comment_weighted_mean` block because their weights may use post-publication metrics. Those features can only be enabled after confirming their measurements/weights were available by the applicable input cutoff. Unweighted embedding summaries, post count, and availability/alignment features remain eligible.

The saved feature manifest lists the exact columns used. Review it before training, especially after changing the upstream encoder output.

## 4. Model comparison

### TTM branch

- Base: `ibm-granite/granite-timeseries-ttm-r2`, revision `52-16-ft-r2.1`.
- This r2.1 branch matches the encoder's 52-week context and provides a 16-step TTM representation.
- Modality adapters produce a small set of latent weekly channels. The adapted series pass through the pretrained TTM.
- TTM's backbone representations and forecast features feed a new scalar BMI regression head.
- The input-channel adaptation may require newly initialized channel-dependent weights; the checkpoint loader reports mismatched weights. The shared pretrained mixer/temporal weights are retained where dimensions permit.

### Chronos-2 branch

- Base: `autogluon/chronos-2`.
- Separate modality adapters produce compact latent weekly series.
- The Chronos-2 encoder processes the latent series jointly as related variates; a learned pooling/regression head maps its contextual embeddings to one BMI value.
- The custom head is necessary because the project target is one BMI scalar with a supplied label pairing, not an unspecified sequence of quantile forecasts.

These are independent model directories and checkpoints. They share the exact same labelled samples, split boundaries, feature selection, and preprocessing artifacts.

## 5. Training and model persistence

From this directory, create the shared labelled dataset:

```bash
python prepare_data.py --config config/default.yaml
```

Train and compare both branches:

```bash
python train_compare.py --config config/default.yaml
```

Training uses the validation set for early stopping/model selection. The test set is evaluated after restoring the best validation checkpoint. Metrics include MAE, RMSE, R², Pearson correlation, and Spearman correlation in the original BMI scale.

Each model directory stores:

```text
outputs/models/<model>/
├── best/
│   ├── model.safetensors
│   ├── model_config.json
│   ├── preprocessing.json
│   ├── encoder_artifacts/
│   └── training_summary.json
├── validation_predictions.csv
├── test_predictions.csv
└── metrics.json
```

`model.safetensors` includes the adapted pretrained model and learned regression layers. Metadata records the exact Hugging Face model/revision, feature order, block membership, input context, target scaler, encoder reducer manifest, and training configuration. The encoder's reducer/scaler artifacts are copied into each model checkpoint for reproducibility; PCA remains implemented and executed only by the encoder project. After training, complete model weights and backbone config are stored locally for reload. Training requires internet access on the first run unless the pretrained weights are already cached.

`outputs/comparison/metrics.csv` is written only after both model runs complete.

## 6. Prediction with a saved model

After training, first build the reduced weekly table with the same saved encoder artifacts, then predict:

```bash
python predict.py \
  --model-dir outputs/models/ttm/best \
  --features ../brand_momentum_encoder/output/weekly/brand_week_features.parquet \
  --output outputs/models/ttm/new_predictions.csv
```

Replace `ttm` with `chronos2` to load the other model. The input must be the final reduced encoder table, and the encoder's saved PCA artifacts must match those copied under the selected checkpoint's `encoder_artifacts/`.

## 7. Configuration

Edit `config/default.yaml` to set:

- Feature and label locations/column names.
- Context length and split ratios.
- Leakage exclusion patterns.
- Training epochs, batch size, early stopping, seed, and device.
- Model IDs/revisions and adapter width.

Both comparison models always use the same config-generated shared split. Use the same seed and training budget for a fair comparison. Model-specific learning rates can differ to accommodate their scale, and are recorded in metadata.

## 8. Data sufficiency and interpretation

The current 32-brand / 157-week encoder snapshot yields overlapping context windows, not thousands of independent observations. Actual labelled sample counts can be much smaller after requiring BMI labels, complete 52-week histories, and chronological split dates. Chronos-2 is comparatively large, so start with frozen-backbone adapter training and a conservative batch size; only increase fine-tuning if there are enough independent labelled dates.

No classifier is trained. If BMI tiers are later required, derive them from the scalar predictions using the agreed thresholds as a separate reporting step. They are not used to replace the continuous BMI objective.

## 9. Reproducibility notes

- Do not use random row splitting.
- Do not fit scalers/reducers on validation or test dates.
- Do not use vectors from an encoder run whose reducer manifest reports `all_observed_weeks_unlabeled_fallback`; rerun encoder stages 07–08 with the shared cutoff instead.
- Do not include post-cutoff engagement snapshots.
- Keep a `target_period` where available to audit the future label association.
- Preserve the generated shared split artifacts when comparing additional model variants.
- Training, model downloads, and inference are intentionally not triggered by project creation.
