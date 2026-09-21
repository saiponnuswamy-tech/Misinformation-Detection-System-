"""Run with:  python -m pytest -q   (or plain: python tests/test_detector.py)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detector  # noqa: E402

FAKE = ("SHOCKING!! Doctors say this miracle drink CURES diabetes in 3 days. "
        "Big pharma doesn't want you to know. 100% guaranteed, no side effects. "
        "Forward this to everyone before it is deleted!!!")

REAL = ("According to a statement published by the Ministry of Health on 12 March 2024, "
        "1,240 cases were recorded last month. Data from https://www.who.int/data "
        "suggests the figure may fall as vaccination continues.")

SCAM = ("URGENT: double your money in 30 days with guaranteed returns, completely risk free. "
        "Only 50 limited slots. Register at http://fast-cash-india.xyz and claim your reward NOW!!")


def test_fake_scores_high():
    r = detector.analyse(FAKE)
    assert r["risk_score"] > 65
    assert r["band_key"] == "high"


def test_credible_scores_low():
    r = detector.analyse(REAL)
    assert r["risk_score"] <= 30
    assert r["band_key"] == "low"


def test_scam_flags_financial_bait():
    codes = {w["code"] for w in detector.analyse(SCAM)["warnings"]}
    assert "FINANCIAL_BAIT" in codes
    assert "SHADY_DOMAIN" in codes


def test_claims_are_extracted():
    claims = detector.analyse(FAKE)["claims"]
    assert claims and "diabetes" in claims[0]["text"].lower()


def test_opinion_is_not_a_claim():
    r = detector.analyse("I think the new policy is a bad idea and I feel it will fail.")
    assert r["claims"] == []


def test_short_input_rejected():
    assert "error" in detector.analyse("fake news")


def test_score_is_bounded():
    for text in (FAKE * 5, REAL, SCAM, "a " * 400):
        r = detector.analyse(text)
        if "risk_score" in r:
            assert 0 <= r["risk_score"] <= 100


def test_trusted_link_lowers_score():
    base = "The department confirmed 1,240 cases were recorded last month."
    with_link = base + " See https://www.who.int/data for the figures."
    assert detector.analyse(with_link)["risk_score"] <= detector.analyse(base)["risk_score"]


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"PASS  {name}")
            except AssertionError:
                failed += 1
                print(f"FAIL  {name}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
