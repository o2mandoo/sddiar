# 260824 음성 — Clova timing proxy 비교 결과

## 기준과 범위

- 입력: local converted PCM WAV, SHA-256 prefix `d78b53216085`
- 길이: 3,171.732초, 16 kHz mono PCM WAV
- reference: Clova Note export의 speaker turn timestamp를 `REF_00`/`REF_01`로 pseudonym 처리
- reference turn 수: 208
- 이 결과는 Clova transcript의 단어 정확도 평가가 아니다. STT/alignment는 연결하지 않았고, 실제 pretrained speaker embedding은 연결했다. 평가는 timestamp/speaker-label proxy와의 화자분리 비교다.

## 처리 가능성

- WAVE_FORMAT_EXTENSIBLE PCM도 처리하도록 WAV decoder를 보완했다.
- 실제 변환 파일을 1,000,000-frame bounded decoder chunk 51개로 끝까지 읽었다.
- 50,747,712 frame, source duration 3,171,732,000 µs가 모두 일치했다.
- canonical 16 kHz processing stream도 51개 chunk로 전체 순회했다.

## 비교 결과

| 실험 | reference 경계 사용 | 결과 |
|---|---:|---|
| forced acoustic 2-cluster | 사용 | turn cluster accuracy **55.29%**, pair precision **50.13%**, pair recall **92.06%** |
| 현재 H1/H2 core + 같은 acoustic feature | 사용 | `H1_CONFIRMED` — 두 화자를 안전하게 확정하지 못하고 한 화자 가설 선택 |
| non-oracle Energy VAD + 500 ms window | 미사용 | time coverage **50.84%**, 전체 시간 speaker-mapped accuracy **35.76%**, predicted-speech 조건부 accuracy **70.35%**, UNKNOWN **49.18%** |
| non-oracle turn proxy | scoring 시에만 사용 | turn coverage **95.67%**, covered-turn accuracy **55.28%** |

## 실제 pretrained ONNX CPU 최종 결과

개발용 Silero VAD ONNX와 WeSpeaker VoxCeleb ResNet34 ONNX를 CPUExecutionProvider로 연결했다. 최종 frontend는 `kaldi-native-fbank 1.22.3`이며 WeSpeaker 공식 ONNX 설정인 16 kHz, 80-bin, 25/10 ms, Hamming, dither 0, snip-edges, utterance CMN을 사용한다. Clova reference 앞 60%에서 assignment distance를 선택하고 뒤 40%를 holdout으로 유지했다. 이 split은 같은 녹음·같은 두 화자이므로 release holdout을 대체하지 않는다.

TorchAudio 2.10.0의 `torchaudio.compliance.kaldi.fbank`와 직접 비교했다. 합성 2초와 승인된 실제 2초 구간 모두 `(198, 80)` frame이 일치했고, feature MAE는 각각 `7.77e-6`, `7.57e-6`, max absolute error는 `1.53e-4`, `2.73e-4`, resulting WeSpeaker embedding cosine은 둘 다 `1.0`이었다. 개발 parity tolerance `atol=3e-4`를 통과했다.

| 항목 | 결과 |
|---|---:|
| H1/H2 | `H2_CONFIRMED` |
| complete merge | **0** |
| H2 centroid separation | **0.752** |
| H2 label / centroid stability | **1.000 / 1.000** |
| holdout turn coverage | **70.42%** |
| holdout covered-turn accuracy | **94.00%** |
| holdout reference timeline coverage | **39.42%** |
| holdout assigned-time speaker accuracy | **99.74%** |
| holdout UNKNOWN rate on output speech | **13.26%** |
| holdout end-to-end time accuracy | **39.31%** |
| full-file turn coverage / accuracy | **76.92% / 92.50%** |
| full-file assigned-time speaker accuracy | **98.96%** |
| full-file reference timeline coverage / end-to-end accuracy | **40.43% / 40.01%** |
| VAD speech ratio | **46.88%** |
| 3,171.732초 처리 wall time | **50.56초** |
| CPU RTF | **0.01594** |
| peak process RSS | **218.62 MB** |
| 3회 RTF p95 / RSS max | **0.01619 / 220.02 MB** |

Quality status는 여전히 `REVIEW_REQUIRED`다. 이유는 단일 녹음 Clova proxy calibration일 뿐 signed multi-recording calibration profile, OSD 검증, independent release holdout이 없기 때문이다.

실제 library CLI도 같은 설정으로 1,009개 redacted span(`SPEAKER_00` 150, `SPEAKER_01` 420, `UNKNOWN` 439)을 생성했다. source path, transcript, audio sample, embedding, centroid는 결과에 포함되지 않는다.

Canonical artifact는 `onnx_full_kaldi_native.json`, `diarization_output_kaldi_native.json`, `runtime_profile_macos_arm64_kaldi_native.json`, `fbank_parity_torchaudio.json`이다. `onnx_full_calibrated_split.json` 등 이전 파일은 NumPy approximation 단계의 이력이며 최종 수치로 사용하지 않는다.

## 1.00 CPU goal 후속 결과

Docker Linux arm64에서 cpuset에는 18개 CPU가 보이지만 `cpu.max=100000/100000`으로 정확히 1.00 CPU-equivalent를 적용했다. network none, ORT/OpenMP/BLAS 1 thread 조건이다.

| 항목 | 기존 FP32 | PCM fast path + H2 single pass |
|---|---:|---:|
| wall / RTF | 120.844초 / 0.03810 | **85.210초 / 0.02687** |
| peak RSS | 180.28MB | **154.45MB** |
| quota utilization | 99.63% | 99.49% |
| span/H2/metric | canonical 동일 | canonical 동일 |

개선 후 동일 음성 밀도의 60분 단순 외삽은 약 96.7초다. 이는 ARM quota proxy이며 실제 Xeon Gold 6230R 성능 증거는 아니다. Linux x86_64 wheelhouse는 emulated clean install을 통과했고 실제 cgroup-v1 preflight를 별도로 제공한다.

Clova timing은 turn 끝을 다음 turn 시작으로 두어 침묵까지 화자 시간으로 채운다. 따라서 40%대 수치는 실제 speech recall/DER가 아니다. system-detected speech 안의 baseline speaker assignment coverage는 **86.22%**다. 반면 화자별 불균형은 실제 개선 대상이다.

| proxy 화자 | assigned rate | assigned accuracy | UNKNOWN rate |
|---|---:|---:|---:|
| `REF_00` | 22.27% | 95.50% | 33.21% |
| `REF_01` | 52.38% | 99.93% | 6.15% |

공식 Silero hysteresis(`0.5/0.35`, 100ms silence, 30ms pad) + ResNet 후보는 full/holdout coverage를 **42.95%/42.45%**, turn coverage를 **79.81%/74.65%**로 높였다. 1-CPU RTF는 `0.02892`였다. 그러나 holdout worst-speaker accuracy가 `94.18%`라 95% gate를 넘지 못해 default-off 상태다.

추가 A/B는 모두 fail-closed됐다.

- MICRO cost/soft decoder 21-arm: calibration accuracy 미달
- MICRO-only 150-frame embedding: valid 66개 복구 후에도 centroid distance 실패
- ResNet authority + CAM++ rescue: coverage/accuracy/fairness/CPU 동시 gate 실패
- per-cluster ceiling 36-arm: coverage 44.05% 후보가 accuracy와 worst-speaker를 훼손
- ResNet34 Q1/Q1b INT8: embedding/pairwise parity 실패
- CAM++ 단독: 1-CPU RTF `0.01231`로 빠르지만 calibration worst-speaker accuracy 94.15%로 미달
- exact-length batch4: full RTF 이득 없이 RSS가 154.45→225.91MB로 증가
- pyannote segmentation3 INT8: FP32 argmax parity 0.67~0.74, 더 느려 거부

현재 상태는 `STOP_SAME_FILE_THRESHOLD_TUNING_DATA_BLOCKER`다. 같은 파일의 threshold를 더 조정하지 않으며, 다음 품질 판정에는 독립 Korean speech mask/RTTM/overlap annotation이 필요하다. 상세 근거는 `research/1CPU_SOL_RESEARCH_SYNTHESIS.md`와 `GOAL_1CPU_PRODUCTION.md`에 정리했다.

## 개선 A/B 판단

| 후보 | 결과 | 판단 |
|---|---|---|
| 기존 NumPy log-mel 근사 | exact FBank 대비 real-segment embedding cosine `0.958`; full end-to-end `39.86%` | release frontend에서 제거 |
| strict Kaldi-compatible FBank + 3초 tracklet | full end-to-end `40.01%`, holdout covered-turn accuracy `94.00%`, RTF `0.01594` | **기본값 유지** |
| 1.5초 tracklet 직접 분할 | H2 outlier `25.26%`, `UNCERTAIN_1_OR_2`, 전부 `UNKNOWN` | safety gate가 거부; 미채택 |
| 3초 tracklet + 1.5/0.75초 subsegment embedding 집계 | holdout turn coverage `71.83%`, accuracy `92.16%`, RTF `0.02578` | 정확도·CPU trade-off 열화; 기본 off |

## 해석

1. 단순 RMS/ZCR/roughness/crest 같은 handcrafted acoustic feature는 실제 speaker embedding을 대체하지 못한다.
2. 특히 Clova turn 경계를 이미 아는 쉬운 조건에서도 forced 2-cluster가 약 55%이므로, 이 특징만으로 speaker-aware summary를 열면 위험하다.
3. 보수적 H1/H2 core는 strict embedding에서는 `H2_CONFIRMED`를 냈지만, 1.5초 direct split A/B에서는 outlier 증가를 감지해 전부 `UNKNOWN`으로 보류했다. 안전 규칙이 품질보다 먼저 동작한 결과다.
4. non-oracle baseline의 70.35%는 Energy VAD가 speech라고 판단한 절반가량의 시간에만 한정된 값이다. 전체 시간축 기준 35.76%가 실제 coverage를 반영하는 더 보수적인 값이다.

## 결론

현재 library wheel이 WAV input, 시간축, VAD, Kaldi-compatible FBank, pretrained embedding, H1/H2, UNKNOWN 정책을 CPU에서 끝까지 실행하고 redacted span을 만드는 것은 확인했다. 단순 handcrafted feature는 실패했지만 실제 speaker embedding을 연결하자 두 화자 합침을 피하면서 high-precision selective attribution이 가능했다. 다만 전체 reference 시간의 약 40%만 speaker로 확정하므로 현재 장점은 높은 선택적 정확도이고, coverage는 계속 개선해야 한다.

다음 유효한 gate는 여러 실제 녹음으로 calibration/holdout을 분리하고, WeSpeaker feature parity와 OSD를 검증한 뒤 diarization-only와 word-speaker mapping을 각각 측정하는 것이다.

## 0.4.0 robustness·runtime 후속

동일 source에서 label-independent 8k, -12dB, noise 20dB challenge를 만들었다.
global gain v2는 1.25배 deadband로 정상/canonical을 exact no-op 처리하면서
-12dB와 8k+-12dB의 H1 merge를 H2로 복원했다. noise 20dB는 gain/CAM++로
복원되지 않았고, RNNoise+ResNet experimental lane에서만 H2가 됐다. 이 결과는
파생 단일 artifact이므로 모두 default off다.

설치된 최종 0.4.0 wheel을 1 CPU/256MiB/network-none/read-only container에서
한 persistent session으로 두 번 실행했다. cold/warm RTF는 `0.02639/0.02643`,
timeline digest는 두 pass 및 0.3 canonical과 동일했다. warm resident 증가는
`3.466%`, cgroup peak는 `248.4MiB`였다. 실제 Xeon/cgroup-v1 증거는 아니다.

Clova transcript를 enterprise proxy로 둔 STT 5분 비교에서는 Whisper turbo Q5
CER `15.90%`가 가장 낮았지만 wall `629.91초`였고, SenseVoice INT8은
`8.13초`였지만 CER `24.35%`였다. CPU-only drop-in 대체는 거부하고 기존 STT에
sddiar speaker attribution을 결합하는 경로를 우선한다.
