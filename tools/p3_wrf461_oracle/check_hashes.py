"""Compare a complete P3 replay to pinned WRF state and net-tendency hashes."""
import argparse
import json
from pathlib import Path
from hash_receipt import receipt


def check(expected, got):
    bad = []
    words = 0
    for name, case in expected["cases"].items():
        if not case["in_claim"]:
            continue
        candidate = got["cases"].get(name)
        if candidate is None:
            bad.append(dict(case=name, error="missing case"))
            continue
        for field, ref in case["fields"].items():
            value = candidate["fields"].get(field)
            words += ref["words"]
            if value is None or any(value[key] != ref[key] for key in
                                    ("sha256", "tendency_sha256", "words")):
                bad.append(dict(case=name, field=field))
    if not words:
        bad.append(dict(error="no oracle words checked"))
    return dict(passed=not bad, state_words=words,
                derived_tendency_words=words, mismatches=bad,
                max_ulp=0 if not bad else None)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("expected",type=Path)
    ap.add_argument("fixtures",type=Path)
    ap.add_argument("candidate",type=Path)
    ap.add_argument("--out",type=Path)
    a=ap.parse_args()
    result=check(json.loads(a.expected.read_text()), receipt(a.fixtures,a.candidate))
    result["expected_manifest"]=str(a.expected)
    result["candidate"]=str(a.candidate)
    text=json.dumps(result,indent=1)
    print(text)
    if a.out: a.out.write_text(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
