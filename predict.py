# -*- coding: utf-8 -*-
"""저장된 분류기를 로드해 상호명 CSV에 카테고리를 예측한다(재학습 없음, 수초).

train.py --save-model 로 저장한 모델을 쓴다.

입력 CSV: merchant 또는 raw_merchant 컬럼.
출력   : 입력 컬럼 + clf_pred(32-cat) + clf_p(확률). --raw 면 상권 중분류 원라벨도.

사용:
    python train.py --zip DATA.zip --level middle --save-model model.joblib   # 1회
    python predict.py --model model.joblib --input merchants.csv -o out.csv    # 이후 반복(수초)
"""
import argparse
import csv

import joblib
from sklearn.feature_extraction.text import HashingVectorizer

import mapping


def load_model(path):
    bundle = joblib.load(path)
    vec = HashingVectorizer(**bundle["vec_params"])
    return vec, bundle["clf"], bundle["level"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="train.py --save-model 로 저장한 파일")
    ap.add_argument("--input", required=True, help="merchant/raw_merchant 컬럼 CSV")
    ap.add_argument("-o", "--output", help="결과 CSV(생략 시 stdout 요약)")
    ap.add_argument("--raw", action="store_true", help="상권 원라벨도 출력(중분류 모델)")
    args = ap.parse_args()

    vec, clf, level = load_model(args.model)
    rows = list(csv.DictReader(open(args.input, encoding="utf-8")))
    col = "merchant" if rows and "merchant" in rows[0] else "raw_merchant"
    merchants = [(r.get(col) or "").strip() for r in rows]

    proba = clf.predict_proba(vec.transform(merchants))
    labels = clf.classes_
    for r, pr in zip(rows, proba):
        i = pr.argmax()
        raw = labels[i]
        r["clf_p"] = f"{pr[i]:.2f}"
        r["clf_pred"] = mapping.to_32(raw) if level == "middle" else raw
        if args.raw:
            r["clf_raw"] = raw

    if args.output:
        fields = list(rows[0].keys())
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.output} ({len(rows)} rows)")
    else:
        for r in rows:
            print(f"  '{r[col]}' -> {r['clf_pred']} (p={r['clf_p']})")


if __name__ == "__main__":
    main()
