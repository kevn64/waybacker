import pytest
from waybacker.utils import (
    normalize_url, is_wayback_url, parse_wayback_rewritten, in_allowed_scope
)

def test_normalize_url():
    assert normalize_url("/web/20200101/https://x.com", "https://web.archive.org") == "https://web.archive.org/web/20200101/https://x.com"

def test_is_wayback_url():
    assert is_wayback_url("https://web.archive.org/web/20240101/https://example.com/app.css")
    assert not is_wayback_url("https://example.com/app.css")

def test_parse_wayback_rewritten():
    ts, orig, mode = parse_wayback_rewritten("https://web.archive.org/web/20191228083558im_/https://i.example.com/a.png")
    assert ts == "20191228083558"
    assert orig == "https://i.example.com/a.png"
    assert mode.startswith("im_")

def test_in_allowed_scope():
    assert in_allowed_scope("https://cdn.ropd.info/a.png", "ropd.info", True, False)
    assert not in_allowed_scope("https://other.com/a.png", "ropd.info", True, False)