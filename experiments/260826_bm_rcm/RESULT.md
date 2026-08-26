# BM-RCM v2 장시간 음성 A/B

상태: `HIGH_PRECISION_CHALLENGER / DEFAULT_OFF / REVIEW_REQUIRED`

H2 stable anchor의 독립 block 분포만 사용해 baseline `UNKNOWN` 연속 구간을
BM-RCM prediction set으로 평가했다. Clova timing은 계산 종료 뒤 scoring에만
사용했다.

| 지표 | baseline | BM-RCM |
|---|---:|---:|
| timeline coverage | 33.78% | 34.93% |
| assigned accuracy | 99.05% | 99.08% |
| worst-speaker accuracy | 94.54% | 95.30% |
| worst-speaker coverage | 13.75% | 16.11% |
| UNKNOWN rate | 27.96% | 25.51% |
| turn accuracy | 92.36% | 94.48% |

44개 candidate run 중 17개만 singleton으로 선택했고 36.512초를 새로
귀속했다. 신규 구간 proxy precision은 99.76%, 기존 assigned 변경은 0초였다.

품질 방향은 graph rescue보다 명확히 좋았다. 다만 추가 wall 6.843초로 현재
host의 상대 overhead가 22.56%이고, UNKNOWN 상대 감소 8.78%와 worst-speaker
coverage 증가는 2.36%p라 사전 gate를 모두 통과하지는 못했다. 따라서
high-precision opt-in challenger로 유지하고 production 기본값으로 승격하지
않는다.

conformal finite-sample 성질은 anchor pseudo-label의 정확성과 block
exchangeability를 전제로 하므로 이 단일 녹음에서 release 보장을 주장하지
않는다. 독립 RTTM/UEM과 실제 Xeon 1CPU 검증이 필요하다.
