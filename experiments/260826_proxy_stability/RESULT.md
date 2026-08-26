# 화자분리 perturbation 안정성 감사

상태: `REVIEW_REQUIRED / RELEASE_AUTHORITY_NONE`

원본 결과와 8 kHz 재표본화, -12 dB, SNR 20 dB 잡음, 8 kHz+-12 dB
변형을 source-time 기준으로 비교했다. label permutation 뒤의 duration 지표만
사용했으며 음원, 전사문, embedding은 저장하지 않았다.

## 결과

| 변형 | H1/H2 변경 | 원본 할당 유지 | speaker flip | speech IoU | boundary F1 |
|---|---:|---:|---:|---:|---:|
| 8 kHz→16 kHz | 아니오 | 93.36% | 6.64% | 90.46% | 90.57% |
| -12 dB | 예 | 72.37% | 27.63% | 71.75% | 63.93% |
| SNR 20 dB | 예 | 75.45% | 24.55% | 74.89% | 83.16% |
| 8 kHz+-12 dB | 예 | 67.86% | 32.14% | 67.23% | 59.20% |

재표본화 단독은 비교적 안정적이지만 저음량과 잡음에서는 세 변형이 H2에서
H1로 바뀌었다. 평균 원본 할당 유지율은 77.26%, speaker flip은 22.74%다.
따라서 정상 음성의 속도보다 저음량·잡음 robustness가 다음 품질 병목이다.

이 결과만으로 global gain이나 RNNoise를 기본값으로 승격하지 않는다. 기존
gain/RNNoise challenger를 독립 녹음과 RTTM/UEM에서 재검증해야 한다. 이 감사는
일관성과 context sensitivity를 측정할 뿐 정확도 또는 DER 증거가 아니다.
