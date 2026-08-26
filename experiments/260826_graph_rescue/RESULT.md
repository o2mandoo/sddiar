# UNKNOWN graph rescue 실험 결과

상태: `REJECTED / DEFAULT_OFF / REVIEW_REQUIRED`

고정된 `ANCHOR_ONLY + bounded kNN` 정책으로 기존 `UNKNOWN`만 구제했다.
주 판정은 상류 assignment calibration도 끈 strict 기본값으로 실행했으며,
Clova timing과 화자 label은 계산 종료 뒤 proxy 평가에만 사용했다.

## 결과

- 기존 할당 변경: `0초`
- 신규 구제: `316개`, `320.256초`
- reference timeline coverage: `33.78% → 43.88%`
- turn coverage: `69.23% → 84.62%`
- turn accuracy given covered: `92.36% → 93.75%`
- 전체 assigned accuracy: `99.05% → 98.00%`
- worst-speaker assigned accuracy: `94.54% → 92.85%`
- 신규 구제 구간 proxy precision: `94.46%`
- graph 및 proxy scoring: `0.481초`

coverage와 turn 지표는 올랐지만 신규 구제 precision과 worst-speaker gate를
통과하지 못했다. 따라서 운영 기본값으로 승격하지 않고 후보를 기각한다.
같은 파일에서 k, margin, posterior threshold를 다시 조정하지 않는다.

이 결과는 graph propagation 자체가 불가능하다는 뜻이 아니다. 독립 Korean
RTTM/UEM에서 새 정책을 사전 고정해 재검증할 수 있도록 default-off 하네스만
남긴다. 현재 결과는 단일 녹음의 Clova timing proxy이며 DER 또는 release 품질
근거가 아니다.
