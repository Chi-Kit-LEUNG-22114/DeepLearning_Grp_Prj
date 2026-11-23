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
