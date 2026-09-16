# -*- coding: utf-8 -*-
"""소상공인 상가(상권)정보로 '상호명 -> 상권업종' 분류기를 학습/평가한다.

- zip을 압축 해제하지 않고 스트리밍으로 읽는다(17개 시도 CSV).
- 상호명 단위로 dedup(같은 이름이 train/test 양쪽에 걸쳐 정확도가 부풀려지는 leakage 방지).
- char n-gram(2~4) + 선형(SGD log-loss). in-domain accuracy / macro-F1 보고.
- --map32 : 중분류 예측을 ktc4-pusan-4 32-cat으로 매핑한 뒤 성능 재측정.
- --predict FILE : 임의의 CSV(raw_merchant 컬럼)에 예측을 찍어본다(도메인 시프트 확인용).

사용:
    python train.py --zip "소상공인시장진흥공단_상가(상권)정보_YYYYMM.zip"
    python train.py --zip DATA.zip --level middle --map32
    python train.py --zip DATA.zip --level middle --map32 --predict my_card.csv

데이터 출처: 공공데이터포털 '소상공인시장진흥공단_상가(상권)정보' (17개 시도 CSV zip).
"""
from __future__ import annotations

import argparse
import csv
import io
import time
import zipfile
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

import mapping

# 상가정보 CSV 컬럼 인덱스(0-based)
COL_NAME = 1   # 상호명
COL_MAJOR = 4  # 상권업종대분류명
COL_MIDDLE = 6  # 상권업종중분류명

csv.field_size_limit(10 ** 7)


def stream_rows(zip_path: str, label_col: int):
    """zip 안 17개 CSV를 스트리밍으로 (상호명, 라벨) 산출."""
    with zipfile.ZipFile(zip_path) as z:
        for name in [n for n in z.namelist() if n.endswith(".csv")]:
            with z.open(name) as fh:
                rd = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", newline=""))
                next(rd, None)  # header
                for row in rd:
                    if len(row) <= label_col:
                        continue
                    merchant = (row[COL_NAME] or "").strip()
                    label = (row[label_col] or "").strip()
                    if merchant and label:
                        yield merchant, label


def build_dataset(zip_path: str, label_col: int):
    """상호명 단위 majority 라벨로 dedup."""
    name_label = defaultdict(Counter)
    total = 0
    for merchant, label in stream_rows(zip_path, label_col):
        name_label[merchant][label] += 1
        total += 1
    X = np.array(list(name_label.keys()), dtype=object)
    y = np.array([c.most_common(1)[0][0] for c in name_label.values()], dtype=object)
    return X, y, total


def make_pipeline():
    vec = HashingVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        n_features=2 ** 20, alternate_sign=False, norm="l2",
    )
    clf = SGDClassifier(
        loss="log_loss", alpha=1e-6, max_iter=20, tol=1e-4,
        n_jobs=-1, random_state=42,
    )
    return vec, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="상가정보 zip 경로")
    ap.add_argument("--level", choices=["major", "middle"], default="middle",
                    help="상권업종 대분류(major, 10) / 중분류(middle, 75)")
    ap.add_argument("--map32", action="store_true",
                    help="중분류 예측을 32-cat으로 매핑해 성능 재측정(level=middle 전용)")
    ap.add_argument("--predict", help="예측을 찍어볼 CSV(raw_merchant 컬럼)")
    ap.add_argument("--min-count", type=int, default=10,
                    help="stratify를 위한 클래스 최소 표본 수")
    args = ap.parse_args()

    label_col = COL_MAJOR if args.level == "major" else COL_MIDDLE
    t0 = time.time()

    X, y, total = build_dataset(args.zip, label_col)
    print(f"[읽기] 유효행 {total:,} / 고유 상호명 {len(X):,} / {time.time()-t0:.0f}s", flush=True)

    dist = Counter(y)
    keep = np.array([dist[l] >= args.min_count for l in y])
    X, y = X[keep], y[keep]
    print(f"[필터] {len(y):,} names / {len(set(y))} classes (min_count={args.min_count})")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[split] train {len(Xtr):,} / test {len(Xte):,}", flush=True)

    vec, clf = make_pipeline()
    clf.fit(vec.transform(Xtr), ytr)
    print(f"[train] done {time.time()-t0:.0f}s", flush=True)

    pred = clf.predict(vec.transform(Xte))
    print(f"\n===== [{args.level}] in-domain 성능 =====")
    print(f"  classes    : {len(set(y))}")
    print(f"  accuracy   : {accuracy_score(yte, pred):.4f}")
    print(f"  macro-F1   : {f1_score(yte, pred, average='macro'):.4f}")
    print(f"  weighted-F1: {f1_score(yte, pred, average='weighted'):.4f}")

    if args.map32 and args.level == "middle":
        yte32 = np.array([mapping.to_32(l) for l in yte])
        pred32 = np.array([mapping.to_32(l) for l in pred])
        print(f"\n===== [중분류 -> 32-cat 매핑 후] =====")
        print(f"  32cat accuracy   : {accuracy_score(yte32, pred32):.4f}")
        print(f"  32cat macro-F1   : {f1_score(yte32, pred32, average='macro'):.4f}")
        print(f"  32cat weighted-F1: {f1_score(yte32, pred32, average='weighted'):.4f}")
        rep = classification_report(yte32, pred32, digits=3, zero_division=0,
                                    output_dict=True)
        rows = [(k, v["f1-score"], v["support"]) for k, v in rep.items()
                if k not in ("accuracy", "macro avg", "weighted avg")]
        print("\n[32-cat 별 F1 (support 순)]")
        for k, f1, s in sorted(rows, key=lambda r: -r[2]):
            print(f"  {k:<12} F1={f1:.3f}  n={int(s):,}")
        print(f"\n[도달 가능 32-cat] {len(mapping.reachable())}종")
        print(f"[도달 불가(공개 상가데이터에 없음)] {mapping.unreachable()}")

    if args.predict:
        merchants = []
        with open(args.predict, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m = (r.get("raw_merchant") or "").strip()
                if m:
                    merchants.append(m)
        uniq = list(dict.fromkeys(merchants))
        proba = clf.predict_proba(vec.transform(uniq))
        labels = clf.classes_
        print(f"\n===== [예측: {args.predict}] {len(uniq)}종 =====")
        for j, m in enumerate(uniq):
            i = proba[j].argmax()
            mid = labels[i]
            line = f"  '{m}' -> {mid} (p={proba[j, i]:.2f})"
            if args.map32 and args.level == "middle":
                line += f" -> 32cat={mapping.to_32(mid)}"
            print(line)

    print(f"\n[완료] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
