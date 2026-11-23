#!/usr/bin/env python3
"""Convert YouTube brand, video, and comment CSV files into one UTF-8 JSON.

The JSON is the final normalized input before multimodal/text embedding.
It preserves multilingual letters while removing emoji, invisible formatting marks,
control characters, and malformed spacing from model-facing text fields.

Expected input files:
  shortlist_30_b.csv
  videos.csv
  comments.csv

Outputs:
  brand_embedding_input.json
  brand_embedding_conversion.xlsx
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

SCHEMA_VERSION = "1.0.0"
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_.-]+")
SPACE_RE = re.compile(r"[ \t\f\v]+")
NEWLINE_RE = re.compile(r"\r\n?|\n+")

# Broad emoji/symbol ranges. Multilingual letters and ordinary punctuation remain.
EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),  # pictographs, transport, flags, supplemental symbols
    (0x2600, 0x27BF),    # miscellaneous symbols and dingbats
    (0x2300, 0x23FF),    # technical symbols often rendered as emoji
    (0x2B00, 0x2BFF),    # arrows and symbols
    (0xFE00, 0xFE0F),    # variation selectors
    (0x1F1E6, 0x1F1FF),  # regional indicator symbols
    (0xE0020, 0xE007F),  # tag characters used in emoji flags
)


def is_emoji_or_format(ch: str) -> bool:
    cp = ord(ch)
    if any(lo <= cp <= hi for lo, hi in EMOJI_RANGES):
        return True
    # Remove invisible formatting/control/surrogate/private-use/unassigned characters.
    return unicodedata.category(ch) in {"Cc", "Cf", "Cs", "Co", "Cn"}


def clean_text(value: Any, *, replace_urls: bool = False, replace_mentions: bool = False) -> str | None:
    """NFC-normalize, remove emoji/weird characters, and normalize whitespace.

    Chinese, Japanese, Korean, Cyrillic, accented Latin, and other valid letters
    are preserved. The function returns None for missing or empty model text.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = unicodedata.normalize("NFC", str(value))
    if replace_urls:
        text = URL_RE.sub("[URL]", text)
    if replace_mentions:
        text = MENTION_RE.sub("[USER]", text)
    text = "".join(ch for ch in text if not is_emoji_or_format(ch))
    text = NEWLINE_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text or None


def clean_identifier(value: Any) -> str | None:
    """Preserve source IDs exactly except surrounding whitespace/control marks."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = unicodedata.normalize("NFC", str(value)).strip()
    text = "".join(ch for ch in text if unicodedata.category(ch) not in {"Cc", "Cf", "Cs"})
    return text or None


def brand_id(name: str) -> str:
    """Create a deterministic ASCII machine key while retaining brand_name separately."""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    value = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    return value or "brand_unknown"


def nullable_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def nullable_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def iso_utc(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed.isoformat().replace("+00:00", "Z")


def week_fields(utc_value: Any, timezone_name: str) -> tuple[str, str, str]:
    ts = pd.to_datetime(utc_value, utc=True, errors="raise").tz_convert(timezone_name)
    start = (ts - pd.Timedelta(days=ts.weekday())).normalize()
    end = start + pd.Timedelta(days=6)
    return start.date().isoformat(), end.date().isoformat(), ts.isoformat()


def read_csv_utf8(path: Path) -> pd.DataFrame:
    # utf-8-sig also reads ordinary UTF-8 and safely strips a future BOM if present.
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=True, low_memory=False)


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def comment_record(row: pd.Series) -> dict[str, Any]:
    return {
        "comment_id": clean_identifier(row.get("comment_id")),
        "parent_comment_id": clean_identifier(row.get("parent_id")),
        "is_reply": bool(clean_identifier(row.get("parent_id"))),
        "author_display_name": clean_text(row.get("author_display_name")),
        "text": clean_text(row.get("text"), replace_urls=True, replace_mentions=True),
        "text_available": bool(clean_text(row.get("text"), replace_urls=True, replace_mentions=True)),
        "like_count": nullable_int(row.get("like_count")),
        "reply_count": nullable_int(row.get("reply_count")),
        "published_at_utc": iso_utc(row.get("published_at")),
        "updated_at_utc": iso_utc(row.get("updated_at")),
        "retrieved_at_hkt": clean_text(row.get("retrieved_at_hkt")),
    }


def video_record(row: pd.Series, video_comments: list[dict[str, Any]], shortlist_row: pd.Series | None,
                 timezone_name: str) -> tuple[dict[str, Any], str, str]:
    week_start, week_end, published_hkt = week_fields(row["published_at"], timezone_name)
    vid = clean_identifier(row.get("video_id"))
    representative = bool(shortlist_row is not None and vid == clean_identifier(shortlist_row.get("video_id")))
    thumbnail_url = clean_text(shortlist_row.get("video_thumbnail_url")) if representative else None
    thumbnail_path = clean_text(shortlist_row.get("video_thumbnail_path")) if representative else None
    record = {
        "video_id": vid,
        "video_url": clean_text(row.get("video_url")),
        "title": clean_text(row.get("title"), replace_urls=True),
        "title_available": bool(clean_text(row.get("title"), replace_urls=True)),
        "published_at_utc": iso_utc(row.get("published_at")),
        "published_at_hkt": published_hkt,
        "duration_iso": clean_text(row.get("duration_iso")),
        "duration_seconds": nullable_int(row.get("duration_seconds")),
        "metrics": {
            "view_count": nullable_int(row.get("view_count")),
            "like_count": nullable_int(row.get("like_count")),
            "comment_count_reported": nullable_int(row.get("comment_count")),
            "comments_in_file": len(video_comments),
        },
        "attributes": {
            "privacy_status": clean_text(row.get("privacy_status")),
            "made_for_kids": nullable_bool(row.get("made_for_kids")),
            "category_id": clean_identifier(row.get("category_id")),
            "tags_count": nullable_int(row.get("tags_count")),
        },
        "thumbnail": {
            "is_representative_shortlist_thumbnail": representative,
            "url": thumbnail_url,
            "local_path": thumbnail_path,
        },
        "retrieved_at_hkt": clean_text(row.get("retrieved_at_hkt")),
        "comments": video_comments,
    }
    return record, week_start, week_end


def aggregate_metrics(videos: list[dict[str, Any]]) -> dict[str, Any]:
    def total(path: tuple[str, ...]) -> int:
        vals = []
        for video in videos:
            value: Any = video
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                vals.append(int(value))
        return sum(vals)
    return {
        "video_count": len(videos),
        "view_count_total": total(("metrics", "view_count")),
        "like_count_total": total(("metrics", "like_count")),
        "comment_count_reported_total": total(("metrics", "comment_count_reported")),
        "comments_in_file_total": sum(len(v["comments"]) for v in videos),
    }


def add_table(ws, rows: list[dict[str, Any]], name: str) -> None:
    if not rows:
        ws.append(["No records"])
        return
    headers = list(rows[0].keys())
    ws.append(headers)
    for row in rows:
        ws.append([row.get(h) for h in headers])
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="top")
    ws.freeze_panes = "A2"
    ws.sheet_view.showGridLines = False
    ref = f"A1:{get_column_letter(len(headers))}{len(rows)+1}"
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False)
    ws.add_table(table)
    for idx, header in enumerate(headers, 1):
        width = min(55, max(12, len(header) + 2))
        if any(token in header for token in ("text", "title", "notes", "url", "path")):
            width = 45
        ws.column_dimensions[get_column_letter(idx)].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def create_workbook(path: Path, payload: dict[str, Any], shortlist: pd.DataFrame, videos: pd.DataFrame,
                    comments: pd.DataFrame) -> None:
    wb = Workbook()
    wb.remove(wb.active)

    brand_rows = []
    period_rows = []
    video_rows = []
    comment_rows = []
    for brand in payload["brands"]:
        brand_rows.append({
            "brand_id": brand["brand_id"], "brand_name": brand["brand_name"],
            "channel_id": brand["youtube_channel"]["channel_id"],
            "youtube_handle": brand["youtube_channel"]["handle"],
            "subscriber_count": brand["youtube_channel"]["subscriber_count"],
            "locked_trends_query": brand["google_trends"]["locked_query"],
            "period_count": len(brand["periods"]),
        })
        for period in brand["periods"]:
            period_rows.append({"brand_id": brand["brand_id"], "brand_name": brand["brand_name"],
                                "week_start": period["week_start"], "week_end": period["week_end"],
                                **period["weekly_metrics"]})
            for video in period["videos"]:
                video_rows.append({
                    "brand_id": brand["brand_id"], "week_start": period["week_start"],
                    "video_id": video["video_id"], "title": video["title"],
                    "published_at_utc": video["published_at_utc"],
                    "view_count": video["metrics"]["view_count"],
                    "like_count": video["metrics"]["like_count"],
                    "comment_count_reported": video["metrics"]["comment_count_reported"],
                    "comments_in_file": video["metrics"]["comments_in_file"],
                    "thumbnail_path": video["thumbnail"]["local_path"],
                })
                for comment in video["comments"]:
                    comment_rows.append({
                        "brand_id": brand["brand_id"], "week_start": period["week_start"],
                        "video_id": video["video_id"], "comment_id": comment["comment_id"],
                        "parent_comment_id": comment["parent_comment_id"], "is_reply": comment["is_reply"],
                        "text": comment["text"], "text_available": comment["text_available"],
                        "like_count": comment["like_count"], "published_at_utc": comment["published_at_utc"],
                    })

    conversion_rows = [
        {"source_file": "shortlist_30_b.csv", "source_key": "brand", "json_key": "brands[].brand_name", "rule": "NFC clean; emoji and format controls removed"},
        {"source_file": "shortlist_30_b.csv", "source_key": "brand", "json_key": "brands[].brand_id", "rule": "deterministic lowercase ASCII snake_case"},
        {"source_file": "videos.csv", "source_key": "video_id", "json_key": "periods[].videos[].video_id", "rule": "preserve source ID"},
        {"source_file": "videos.csv", "source_key": "title", "json_key": "periods[].videos[].title", "rule": "NFC; remove emoji/control; normalize whitespace"},
        {"source_file": "videos.csv", "source_key": "view_count", "json_key": "periods[].videos[].metrics.view_count", "rule": "integer or null"},
        {"source_file": "videos.csv", "source_key": "like_count", "json_key": "periods[].videos[].metrics.like_count", "rule": "integer or null; missing not converted to zero"},
        {"source_file": "comments.csv", "source_key": "comment_id", "json_key": "videos[].comments[].comment_id", "rule": "preserve source ID"},
        {"source_file": "comments.csv", "source_key": "parent_id", "json_key": "videos[].comments[].parent_comment_id", "rule": "null for top-level; preserve source parent ID"},
        {"source_file": "comments.csv", "source_key": "text", "json_key": "videos[].comments[].text", "rule": "NFC; replace URLs/users; remove emoji/control; normalize whitespace"},
        {"source_file": "derived", "source_key": "published_at", "json_key": "periods[].week_start/week_end", "rule": "Monday-Sunday in Asia/Hong_Kong"},
    ]

    for title, rows, table_name in [
        ("Brands", brand_rows, "BrandsTable"), ("Weekly Periods", period_rows, "PeriodsTable"),
        ("Videos", video_rows, "VideosTable"), ("Comments", comment_rows, "CommentsTable"),
        ("Conversion Map", conversion_rows, "ConversionMapTable")]:
        ws = wb.create_sheet(title)
        add_table(ws, rows, table_name)

    summary = wb.create_sheet("Summary", 0)
    summary.sheet_view.showGridLines = False
    summary["A1"] = "YouTube Brand JSON Conversion"
    summary["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    summary["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    summary.merge_cells("A1:D1")
    items = [
        ("Schema version", payload["schema_version"]), ("Encoding", payload["encoding"]),
        ("Brand records", len(payload["brands"])), ("Source video rows", len(videos)),
        ("Source comment rows", len(comments)), ("Week convention", payload["week_definition"]),
        ("Text cleaning", "NFC; emoji and invisible/control characters removed; multilingual letters retained"),
        ("JSON structure", "brand -> period -> video -> comments"),
    ]
    for i, (label, value) in enumerate(items, 3):
        summary.cell(i, 1, label).font = Font(bold=True, color="666666")
        summary.cell(i, 2, value)
    summary.column_dimensions["A"].width = 28
    summary.column_dimensions["B"].width = 95
    summary.freeze_panes = "A3"
    wb.save(path)


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    shortlist = read_csv_utf8(args.shortlist)
    videos = read_csv_utf8(args.videos)
    comments = read_csv_utf8(args.comments)

    require_columns(shortlist, {"brand", "channel_id", "youtube_handle", "locked_trends_query"}, "shortlist")
    require_columns(videos, {"brand", "channel_id", "video_id", "title", "published_at", "view_count"}, "videos")
    require_columns(comments, {"brand", "video_id", "comment_id", "parent_id", "text", "published_at"}, "comments")

    if shortlist["brand"].duplicated().any():
        raise ValueError("shortlist contains duplicate brand rows")
    if videos["video_id"].duplicated().any():
        raise ValueError("videos contains duplicate video_id rows")
    if comments["comment_id"].duplicated().any():
        raise ValueError("comments contains duplicate comment_id rows")

    comments_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in comments.iterrows():
        comments_by_video[clean_identifier(row["video_id"])].append(comment_record(row))
    for records in comments_by_video.values():
        records.sort(key=lambda x: (x["published_at_utc"] or "", x["comment_id"] or ""))

    shortlist_map = {clean_text(r["brand"]): r for _, r in shortlist.iterrows()}
    videos_by_brand: dict[str, list[pd.Series]] = defaultdict(list)
    for _, row in videos.iterrows():
        videos_by_brand[clean_text(row["brand"])].append(row)

    brands_json = []
    for _, master in shortlist.sort_values("brand").iterrows():
        display = clean_text(master["brand"])
        bid = brand_id(display)
        periods: dict[str, dict[str, Any]] = {}
        for row in sorted(videos_by_brand.get(display, []), key=lambda r: str(r["published_at"])):
            vid = clean_identifier(row["video_id"])
            record, start, end = video_record(row, comments_by_video.get(vid, []), master, args.timezone)
            bucket = periods.setdefault(start, {"week_start": start, "week_end": end, "videos": []})
            bucket["videos"].append(record)
        period_list = []
        for start in sorted(periods):
            period = periods[start]
            period["videos"].sort(key=lambda x: (x["published_at_utc"] or "", x["video_id"] or ""))
            period["weekly_metrics"] = aggregate_metrics(period["videos"])
            period_list.append(period)

        brands_json.append({
            "brand_id": bid,
            "brand_name": display,
            "youtube_channel": {
                "channel_id": clean_identifier(master.get("channel_id")),
                "handle": clean_text(master.get("youtube_handle")),
                "url": clean_text(master.get("youtube_url")),
                "subscriber_count": nullable_int(master.get("subscriber_count")),
                "channel_thumbnail_url": clean_text(master.get("channel_thumbnail_url")),
                "channel_thumbnail_path": clean_text(master.get("channel_thumbnail_path")),
            },
            "google_trends": {
                "locked_query": clean_text(master.get("locked_trends_query")),
                "status": clean_text(master.get("trends_status")),
            },
            "source_notes": clean_text(master.get("notes"), replace_urls=False),
            "periods": period_list,
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "encoding": "UTF-8",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "week_definition": f"Monday-Sunday; timezone={args.timezone}",
        "text_cleaning": {
            "unicode_normalization": "NFC",
            "emoji_removed": True,
            "format_and_control_characters_removed": True,
            "multilingual_letters_preserved": True,
            "urls_in_comment_text": "replaced with [URL]",
            "user_mentions_in_comment_text": "replaced with [USER]",
        },
        "hierarchy": "brands -> periods -> videos -> comments",
        "brands": brands_json,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    create_workbook(args.excel_output, payload, shortlist, videos, comments)
    return args.json_output, args.excel_output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shortlist", type=Path, default=Path("shortlist_30_b.csv"))
    p.add_argument("--videos", type=Path, default=Path("videos.csv"))
    p.add_argument("--comments", type=Path, default=Path("comments.csv"))
    p.add_argument("--json-output", type=Path, default=Path("brand_embedding_input.json"))
    p.add_argument("--excel-output", type=Path, default=Path("brand_embedding_conversion.xlsx"))
    p.add_argument("--timezone", default="Asia/Hong_Kong")
    return p.parse_args()

if __name__ == "__main__":
    json_path, excel_path = build(parse_args())
    print(f"Wrote {json_path}")
    print(f"Wrote {excel_path}")
