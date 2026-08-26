# Bounded global gain v2 안정성 결과

상태: `ACCEPT_OPT_IN_API / DEFAULT_OFF / REVIEW_REQUIRED`

동일한 네 perturbation 결과를 gain 미사용과 gain v2 사용으로 비교했다.
speaker label permutation 뒤 source-time duration 지표만 계산했다.

| 지표 | baseline | gain v2 |
|---|---:|---:|
| H1/H2 변경률 | 75.00% | 25.00% |
| 원본 할당 유지율 | 77.26% | 90.25% |
| speaker flip | 22.74% | 9.75% |
| speech IoU | 76.08% | 88.38% |
| boundary F1 | 74.21% | 90.72% |

-12 dB와 8 kHz+-12 dB는 H1에서 H2로 복구했다. 정상 원음과 8 kHz
변형은 exact no-op이었다. SNR 20 dB 잡음에는 gain이 적용되지 않았고 H1
상태가 그대로 남았다.

따라서 gain v2는 명시적인 opt-in library API로 채택한다. production 기본값과
자동 denoise router로는 승격하지 않는다. 이 결과는 한 녹음에서 파생된
일관성 증거이며 DER 또는 독립 정확도 증거가 아니다.
