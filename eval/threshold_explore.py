# -*- coding: utf-8 -*-
"""confidence cutoff 탐색.

하이브리드: 규칙이 커버하면 규칙(결정적). 규칙 miss면 분류기 예측을 쓰되
clf_p >= th 일 때만 자동채택, 아니면 review.

threshold th 를 촘촘히 훑으며:
  - residual(규칙 miss)에서 자동채택 수 / 그 정밀도
  - 전체 파이프라인 자동화율 / 자동채택 정밀도(규칙+분류기)
정밀도 목표(0.90/0.95/0.99)를 만족하는 최소 th(=최대 커버리지)를 찾는다.
"""
import csv, sys


def covered(p):
    return bool(p) and p.strip().lower() not in ("", "uncertain", "none")


def analyze(rows, title):
    n = len(rows)
    rule_cov = [r for r in rows if covered(r["rule_pred"])]
    residual = [r for r in rows if not covered(r["rule_pred"])]
    rule_ok = sum(1 for r in rule_cov if r["rule_pred"].strip() == r["gold"].strip())

    print(f"\n===== {title} (n={n}) =====")
    print(f"규칙 커버 {len(rule_cov)} (정밀도 {rule_ok/len(rule_cov):.2f}) | "
          f"residual(규칙miss) {len(residual)}")
    print("\n th    │ residual자동 │ residual정밀도 │ 전체자동화 │ 전체자동정밀도 │ review")
    print(" ──────┼──────────────┼────────────────┼────────────┼────────────────┼───────")
    grid = [i / 100 for i in range(30, 100, 5)]
    for th in grid:
        auto_res = [r for r in residual if float(r["clf_p"]) >= th]
        auto_res_ok = sum(1 for r in auto_res if r["clf_pred"].strip() == r["gold"].strip())
        # 전체 자동채택 = 규칙커버(항상) + residual 자동채택
        auto_total = len(rule_cov) + len(auto_res)
        auto_total_ok = rule_ok + auto_res_ok
        review = len(residual) - len(auto_res)
        rp = f"{auto_res_ok/len(auto_res):.2f}" if auto_res else "  - "
        print(f" {th:.2f}  │  {len(auto_res):>2}/{len(residual):<2}       │     {rp}       │  "
              f"{auto_total/n:.2f}      │     {auto_total_ok/auto_total:.2f}       │  {review}")

    # 정밀도 목표별 최소 th (전체 자동채택 기준)
    print("\n[정밀도 목표별 최소 th (전체 자동채택 정밀도 기준) — 커버리지 최대]")
    for target in (0.90, 0.95, 0.99):
        best = None
        for th in [i / 100 for i in range(0, 101)]:
            auto_res = [r for r in residual if float(r["clf_p"]) >= th]
            auto_res_ok = sum(1 for r in auto_res if r["clf_pred"].strip() == r["gold"].strip())
            at = len(rule_cov) + len(auto_res)
            atk = rule_ok + auto_res_ok
            prec = atk / at if at else 0
            if prec >= target:
                best = (th, at / n, prec)
                break
        if best:
            print(f"  정밀도≥{target:.2f}: th={best[0]:.2f} → 자동화 {best[1]:.0%}, 실제정밀도 {best[2]:.2f}")
        else:
            print(f"  정밀도≥{target:.2f}: 달성 불가(현 데이터)")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "predictions_92.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    clear = [r for r in rows if str(r.get("ambiguous", "")).strip() != "1"]
    analyze(clear, "명확한 케이스만(애매 제외)")
    analyze(rows, "전체(애매 포함)")
    print(f"\n[주의] residual 표본이 수십 건뿐 → 각 th 칸이 소수 샘플. "
          f"곡선 모양·knee는 참고, 정확한 최적 th 확정엔 라벨 수백 개 필요.")


if __name__ == "__main__":
    main()
