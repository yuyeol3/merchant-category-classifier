# -*- coding: utf-8 -*-
"""전략별(규칙-only / 분류기-only / 하이브리드) 카테고리 분류 성능을 객관 비교한다.

입력 CSV 컬럼:
    merchant     원본 카드 상호 문자열
    gold         정답 32-cat
    rule_pred    규칙 파이프라인 예측(빈칸/uncertain = 미해소 -> review)
    clf_pred     분류기 예측(32-cat)
    clf_p        분류기 top-1 확률(선택; 임계값 스윕용)
    ambiguous    1이면 정답이 애매한 케이스(집계에서 분리)

지표:
    coverage           자동 분류 비율(review로 안 넘긴 비율)
    acc_on_covered     자동 분류한 것 중 정확도(정밀도)
    acc_overall        전체 대비 정확도(미해소=오답 취급)
    residual_acc       규칙이 놓친(uncertain) 건에서 분류기 정확도 = 분류기의 순수 기여

하이브리드 = 규칙 우선, 규칙 uncertain이면 분류기.

사용: python evaluate.py predictions.csv
"""
import csv
import sys
from collections import Counter


def is_covered(pred):
    return bool(pred) and pred.strip().lower() not in ("", "uncertain", "none")


def arm_metrics(rows, pred_key):
    n = len(rows)
    covered = [r for r in rows if is_covered(r.get(pred_key))]
    correct_cov = [r for r in covered if r[pred_key].strip() == r["gold"].strip()]
    correct_all = correct_cov  # 미커버는 오답 취급
    return {
        "n": n,
        "coverage": len(covered) / n if n else 0.0,
        "acc_on_covered": len(correct_cov) / len(covered) if covered else 0.0,
        "acc_overall": len(correct_all) / n if n else 0.0,
    }


def hybrid_pred(r):
    rp = r.get("rule_pred", "")
    return rp if is_covered(rp) else r.get("clf_pred", "")


def residual_acc(rows):
    """규칙이 uncertain인 건에서 분류기 정확도(분류기의 순수 기여)."""
    miss = [r for r in rows if not is_covered(r.get("rule_pred"))]
    if not miss:
        return None, 0
    ok = sum(1 for r in miss if r.get("clf_pred", "").strip() == r["gold"].strip())
    return ok / len(miss), len(miss)


def threshold_sweep(rows):
    """clf_p가 있으면 임계값별 (자동채택 커버리지, 그 구간 정확도)."""
    have = [r for r in rows if r.get("clf_p")]
    if len(have) < 30:
        return None
    out = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        auto = [r for r in have if float(r["clf_p"]) >= th]
        if not auto:
            continue
        ok = sum(1 for r in auto if r["clf_pred"].strip() == r["gold"].strip())
        out.append((th, len(auto) / len(have), ok / len(auto)))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "predictions_example.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    clear = [r for r in rows if str(r.get("ambiguous", "")).strip() != "1"]

    def block(title, data):
        print(f"\n===== {title} (n={len(data)}) =====")
        for name, key in [("규칙-only", "rule_pred"), ("분류기-only", "clf_pred")]:
            m = arm_metrics(data, key)
            print(f"  {name:<12} coverage={m['coverage']:.2f}  "
                  f"acc_on_covered={m['acc_on_covered']:.2f}  acc_overall={m['acc_overall']:.2f}")
        # 하이브리드
        for r in data:
            r["_hybrid"] = hybrid_pred(r)
        mh = arm_metrics(data, "_hybrid")
        print(f"  {'하이브리드':<12} coverage={mh['coverage']:.2f}  "
              f"acc_on_covered={mh['acc_on_covered']:.2f}  acc_overall={mh['acc_overall']:.2f}")
        ra, k = residual_acc(data)
        if ra is not None:
            print(f"  → 분류기 순수 기여: 규칙이 놓친 {k}건 중 정확도 {ra:.2f}")

    block("전체(애매 포함)", rows)
    block("명확한 케이스만(애매 제외)", clear)

    sweep = threshold_sweep(rows)
    if sweep:
        print("\n[분류기 confidence 임계값 스윕]")
        print("  th    auto커버리지  그구간정확도")
        for th, cov, acc in sweep:
            print(f"  {th:.1f}   {cov:.2f}        {acc:.2f}")
    else:
        print("\n[임계값 스윕] clf_p 라벨 30건 미만 → 생략(표본 부족)")

    print(f"\n[주의] n={len(rows)}. 결정을 내리려면 라벨 상호가 수백 개 필요"
          f"(정확도 ±5%p @95%CI ≈ 250건). 지금은 하네스 동작 확인용 seed다.")


if __name__ == "__main__":
    main()
