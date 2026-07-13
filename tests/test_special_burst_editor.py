from __future__ import annotations

import yaml

from grb_project.special_bursts import parse_time_segments_text, upsert_special_burst_config


def test_parse_time_segments_text_accepts_named_and_unnamed_rows() -> None:
    segments = parse_time_segments_text(
        """
        first 0.1 1.0
        1.0 2.5
        third,2.5,4.0
        # ignored comment
        """
    )

    assert segments == [
        {"name": "first", "start": 0.1, "stop": 1.0},
        {"name": "seg2", "start": 1.0, "stop": 2.5},
        {"name": "third", "start": 2.5, "stop": 4.0},
    ]


def test_parse_time_segments_text_rejects_invalid_ranges() -> None:
    try:
        parse_time_segments_text("bad 2.0 1.0")
    except ValueError as exc:
        assert "stop 必须大于 start" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("invalid range should fail")


def test_upsert_special_burst_config_adds_new_burst(tmp_path) -> None:
    yaml_path = tmp_path / "special_bursts.yaml"

    saved = upsert_special_burst_config(
        yaml_path,
        name="GRB260101A",
        bnname="bn260101001",
        active_interval="0-5",
        background_interval="-20--5,50-80",
        time_segments=[
            {"name": "seg1", "start": 0.0, "stop": 1.0},
            {"name": "seg2", "start": 1.0, "stop": 5.0},
        ],
    )

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert saved["name"] == "GRB260101A"
    assert data["special_bursts"] == [saved]


def test_upsert_special_burst_config_updates_existing_by_bnname(tmp_path) -> None:
    yaml_path = tmp_path / "special_bursts.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "special_bursts": [
                    {
                        "name": "GRB260101A",
                        "bnname": "bn260101001",
                        "time_segments": [{"name": "old", "start": 0.0, "stop": 1.0}],
                    },
                    {
                        "name": "GRB260102A",
                        "bnname": "bn260102001",
                        "time_segments": [{"name": "keep", "start": 0.0, "stop": 2.0}],
                    },
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    saved = upsert_special_burst_config(
        yaml_path,
        name="GRB260101B",
        bnname="bn260101001",
        time_segments=[{"name": "new", "start": 1.0, "stop": 3.0}],
    )

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["special_bursts"][0] == saved
    assert data["special_bursts"][1]["bnname"] == "bn260102001"
