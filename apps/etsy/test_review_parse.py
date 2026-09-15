#!/usr/bin/env python3
"""Test shop-review count parsing against testdata/review_parse_cases.json.

Mirrors parseShopReviewCount in export_etsy_results.js (keep in sync).
Run: python3 test_review_parse.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "testdata" / "review_parse_cases.json"


def parse_count_token(token: str | None) -> int | None:
    if not token:
        return None
    s = re.sub(r"[,\s]+", "", token).lower()
    if s.endswith("k"):
        try:
            return round(float(s[:-1]) * 1000)
        except ValueError:
            return None
    if s.endswith("m"):
        try:
            return round(float(s[:-1]) * 1_000_000)
        except ValueError:
            return None
    if not re.fullmatch(r"\d+", s):
        return None
    return int(s)


def parse_shop_review_count(text: str, shop_name: str = "") -> int | None:
    if not text:
        return None
    t = text

    with_word = (
        re.search(r"(?:from\s+)?([\d,]+(?:\.\d+)?\s*[kKmM]?)\s*reviews?\b", t, re.I)
        or re.search(r"([\d.]+(?:\.\d+)?\s*[kKmM]?)\s*Bewertungen\b", t, re.I)
        or re.search(r"bei\s+([\d.]+)\s*Bewertungen\b", t, re.I)
    )
    if with_word:
        token = with_word.group(1)
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", token):
            token = token.replace(".", "")
        n = parse_count_token(token)
        if n is not None:
            return n

    stripped = re.sub(r"[\d.]+\s*(?:out of|/)\s*5(?:\s*stars?)?", " ", t, flags=re.I)
    stripped = re.sub(r"\bstar\s*seller\b", " ", stripped, flags=re.I)

    km = re.search(r"\(([\d,]+(?:\.\d+)?\s*[kKmM])\)", stripped, re.I)
    if km:
        n = parse_count_token(km.group(1))
        if n is not None:
            return n

    if shop_name:
        escaped = re.escape(shop_name)
        near = re.search(
            escaped + r"\s*\(([\d,]+(?:\.\d+)?\s*[kKmM]?)\)", stripped, re.I
        )
        if near:
            n = parse_count_token(near.group(1))
            if n is not None:
                return n

    for m in re.finditer(r"\(([\d,]+(?:\.\d+)?\s*[kKmM]?)\)", stripped, re.I):
        if "%" in m.group(0):
            continue
        n = parse_count_token(m.group(1))
        if n is not None and n >= 6:
            return n

    bare = re.search(r"\b([\d]+(?:\.\d+)?\s*[kKmM])\b", stripped, re.I)
    if bare:
        n = parse_count_token(bare.group(1))
        if n is not None and n >= 100:
            return n

    return None


def main() -> None:
    cases = json.loads(CASES.read_text())["cases"]
    failed = 0
    for c in cases:
        got = parse_shop_review_count(c["text"], c.get("shop") or "")
        expect = c["expect"]
        if got != expect:
            failed += 1
            print(f"FAIL {c['name']}: got {got}, expect {expect}")
        else:
            print(f"ok   {c['name']}")
    if failed:
        raise SystemExit(f"{failed} failed")
    print(f"All {len(cases)} cases passed")


if __name__ == "__main__":
    main()
