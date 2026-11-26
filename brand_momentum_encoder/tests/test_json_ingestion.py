import json
from pathlib import Path

from brand_encoder.ingestion import ingest_embedding_json


def test_aggregated_json_flattens_to_canonical_tables(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps({
        "brands": [{
            "brand_id": "sample_brand",
            "brand_name": "Sample Brand",
            "youtube_channel": {"subscriber_count": 42},
            "google_trends": {"locked_query": "Sample Brand", "status": "pass"},
            "periods": [{
                "week_start": "2025-01-06",
                "videos": [{
                    "video_id": "video-1",
                    "title": "A title",
                    "published_at_utc": "2025-01-08T12:00:00Z",
                    "retrieved_at_hkt": "2026-01-01 12:00:00 HKT",
                    "metrics": {"view_count": 100, "like_count": 4, "comment_count_reported": 2},
                    "thumbnail": {"local_path": "/not/available.jpg"},
                    "comments": [{
                        "comment_id": "comment-1",
                        "parent_comment_id": None,
                        "text": "A comment",
                        "like_count": 12,
                        "published_at_utc": "2025-01-09T12:00:00Z",
                    }],
                }],
            }],
        }],
    }), encoding="utf-8")

    tables = ingest_embedding_json(source)

    assert len(tables["brands"]) == 1
    assert len(tables["posts"]) == 1
    assert tables["posts"].iloc[0]["brand_id"] == "sample_brand"
    assert tables["posts"].iloc[0]["measurement_at"] == "2026-01-01 12:00:00+08:00"
    assert len(tables["images"]) == 1
    assert len(tables["comments"]) == 1
    # The JSON has a Trends query, not a numeric observation: never fabricate it.
    assert tables["trends"].empty
    # Current comment likes are not historical observations.
    assert tables["comments"].iloc[0]["likes"] is None
