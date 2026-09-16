# eval — 규칙 vs 분류기, 객관적 결정 하네스

"규칙만 vs 규칙+분류기(하이브리드) 중 뭐가 나은가"를 **같은 라벨셋에서 같은 지표로** 재서
숫자로 결정하기 위한 평가 틀.

## 세 전략

| 전략 | 정의 |
|---|---|
| 규칙-only | `tools/keyword_rules.py` 파이프라인. uncertain은 미해소(review) |
| 분류기-only | 상가데이터 학습 분류기(char n-gram + SGD → 32-cat 매핑) |
| 하이브리드 | 규칙 우선, 규칙이 uncertain일 때만 분류기 |

## 지표 (`evaluate.py`)

- **coverage**: 자동 분류 비율(review로 안 넘긴 비율)
- **acc_on_covered**: 자동 분류한 것의 정확도(정밀도)
- **acc_overall**: 전체 대비 정확도(미해소=오답)
- **residual_acc**: 규칙이 놓친 건에서 분류기 정확도 = **분류기의 순수 기여**
- (라벨 30건↑ + `clf_p` 있으면) confidence 임계값 스윕: 자동채택 커버리지 vs 그 구간 정확도

## 결정 규칙 (제안)

세무 판정 특성상 **정밀도(자동채택한 게 맞을 것)**가 커버리지보다 중요하다. 그래서:

> **분류기를 채택한다** ⇔ 하이브리드가 규칙-only 대비
> ① `acc_overall`을 유의하게 올리고(커버리지 회복),
> ② `acc_on_covered`(정밀도)를 목표선(예: ≥0.90) 아래로 떨어뜨리지 않는다.
> 정밀도가 떨어지면 → confidence 임계값을 올려 저확신은 review로 보낸다(스윕에서 지점 선택).

즉 "분류기의 residual_acc가 충분히 높아 자동화 이득이 review 비용보다 큰가"를 본다.

## 필요한 표본 수

정확도 ~0.8을 **±5%p (95% CI)**로 재려면 라벨 **약 250개 고유 상호**.
거친 go/no-go만이면 100~150. 지금 seed는 **9개** — 결정 불가, 하네스 동작 확인용.

## seed 결과 (n=9 실제 카드셋 — 집계만, 행 데이터는 비공개)

> 아래 수치는 로컬의 실제 카드 상호 9개(gold는 초안 라벨, 검증 필요)로 낸 것이다.
> 행 단위 CSV는 개인 소비내역이라 저장소에 커밋하지 않는다(`predictions_seed.csv`는 gitignore).
> 저장소의 `predictions_example.csv`는 **합성 예시**로, 하네스 실행 방법을 보이기 위한 것이다.

```
명확한 케이스만(애매 3건 제외, n=6):
  규칙-only    coverage=0.67  acc_on_covered=1.00  acc_overall=0.67
  분류기-only   coverage=1.00  acc_on_covered=0.83  acc_overall=0.83
  하이브리드     coverage=1.00  acc_on_covered=0.83  acc_overall=0.83
  → 분류기 순수 기여: 규칙이 놓친 2건 중 정확도 0.50
```

읽는 법(방향만, 통계적 의미 없음): 규칙은 커버한 건 100% 정확하지만 1/3을 review로 넘긴다.
분류기가 그 residual을 메워 커버리지를 100%로 올리지만, 메운 것 중 절반만 맞는다.
→ **이 residual_acc가 결정의 핵심 숫자**이고, 이걸 신뢰하려면 표본을 250개로 늘려야 한다.

## 라벨 늘려서 진짜 숫자 뽑기

1. `data/`에 실제 카드 이용내역을 모아 고유 상호를 추출한다(개인정보 — 저장소에 커밋 금지).
2. 각 상호에 gold 32-cat을 붙인다(`labels_template.csv` 형식).
3. 규칙 예측 채우기: ktc4-pusan-4 저장소에서
   `keyword_rules.pipeline()`을 상호별로 돌려 `rule_pred` 컬럼을 채운다.
4. 분류기 예측 채우기: `python ../train.py --zip DATA.zip --level middle --map32 --predict labels.csv`
   결과의 `clf_pred`/`clf_p`를 채운다.
5. `python evaluate.py labels.csv` → 세 전략 지표 비교.

## 파일

- `evaluate.py` — 예측 CSV → 전략별 지표 + 임계값 스윕.
- `threshold_explore.py` — 하이브리드 confidence cutoff 촘촘히 탐색(정밀도 목표별 최적 th).
- `predictions_example.csv` — 합성 예시(3-way 예측 채워짐). `python evaluate.py`로 바로 실행.
- `labels_template.csv` — 라벨링 시작 틀.
