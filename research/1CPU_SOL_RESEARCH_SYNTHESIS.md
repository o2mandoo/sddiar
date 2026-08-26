# 1 CPU 화자분리 독립 Sol 연구 종합

기준일: 2026-08-26  
상태: `GOAL_ACTIVE / LOCAL_040_REPEAT_PASS / STOP_SAME_FILE_THRESHOLD_TUNING`

서로 결과를 공유하지 않은 최신 Sol 연구자 3명이 각각 품질 알고리즘, 1-CPU runtime, 모델·양자화를 검토했다. 아래는 이후 실제 구현·전체 파일 A/B까지 반영한 Terra 종합 판단이다.

## 공통 결론

1. 1 CPU 처리속도는 이미 충분하다. M5 수치를 Xeon으로 환산하지 않아도 정확한 local cgroup quota에서 52분 파일을 85~92초에 처리한다.
2. `40% coverage`는 실제 speech recall이 아니다. Clova timing이 침묵까지 다음 turn 화자로 채워 wall-clock 전체를 분모로 쓰기 때문이다.
3. 실제 system-detected speech 안의 baseline assigned coverage는 86.22%, conditional accuracy는 98.96%다.
4. 남은 핵심 문제는 조용한 화자 불균형, 짧은 발화 centroid distance, SCD/OSD 부재, 독립 Korean annotation 부재다.
5. 같은 파일의 threshold를 더 조정하면 과적합이다. 실제 speech mask/RTTM/overlap annotation이 다음 품질 gate다.

## 1-CPU 성능 결과

환경: Docker Linux arm64, `cpu.max=100000/100000`, cpuset에는 18 CPU 노출, network none, ORT/OpenMP/BLAS 1 thread.

| 후보 | wall / RTF | RSS | 결과 |
|---|---:|---:|---|
| B0 기존 FP32 | 120.844초 / 0.03810 | 180.28MB | 기준 |
| C1 NumPy PCM + H2 single pass | 85.210초 / 0.02687 | 154.45MB | **채택**; 품질 동일 |
| temporal VAD + ResNet | 약 91.72초 / 0.02892 | 154.47MB | 품질 challenger |
| CAM++ 단독 | 약 39.03초 / 0.01231 | 174.45MB | 빠르지만 fairness gate 실패 |
| exact-length batch4 | 0.02692 | 225.91MB | 이득 없음, 기본 off |

C1의 60분 단순 외삽은 약 96.7초다. 실제 Xeon 6230R cgroup-v1 성능은 별도 승인 gate다.

## 품질 분해

Baseline:

- H2 separation `0.7519`, stability 약 `1.0`, complete merge `0`
- VAD speech `1,487.040초`
- assigned `1,282.096초`, UNKNOWN `204.944초`
- MICRO UNKNOWN `119.008초`, non-MICRO UNKNOWN `85.936초`
- `REF_00` assigned rate/accuracy/UNKNOWN: `22.27% / 95.50% / 33.21%`
- `REF_01`: `52.38% / 99.93% / 6.15%`

즉 평균 정확도가 높은 동시에 조용한 proxy 화자가 크게 불리하다. worst-speaker metric을 calibration 필수 조건으로 추가했다.

## 실제 품질 A/B

| 실험 | 결과 | 판단 |
|---|---|---|
| E1 MICRO cost + strict soft decoder 21-arm | 최고 후보도 calibration accuracy 미달 | baseline fail-closed |
| E2 MICRO-only 150-frame resize | valid embedding 66개 증가, 355 MICRO 모두 distance gate 실패 | 미채택 |
| E3 ResNet authority + CAM++ UNKNOWN rescue | 43.224초 구조 가능, calibration accuracy/fairness/CPU gate 실패 | 미채택 |
| E4 temporal ResNet per-cluster ceiling 36-arm | coverage 44.05% 후보는 accuracy/worst-speaker 붕괴 | 미채택, same-file tuning 종료 |
| 공식 Silero temporal hysteresis | full/holdout coverage 42.95/42.45%, turn 품질 개선 | 가장 유효한 challenger; 독립 VAD/RTTM 필요 |
| CAM++ 단독 | turn·coverage·CPU 개선, calibration worst-speaker 94.15% | ResNet 대체 거부 |

Temporal ResNet은 전체 worst-speaker accuracy `95.20%`를 지켰지만 holdout worst-speaker가 `94.18%`라 production gate를 넘지 못했다.

## 모델·양자화 결과

| 후보 | 크기/속도 | parity | 판단 |
|---|---|---|---|
| ResNet34 FP32 | 26.5MB | canonical | 기준·fallback 유지 |
| Q1 static QDQ INT8 | 6.77MB, embedding 약 18% 빠름 | cosine p01 0.966, pair error p95 0.0379 | 거부 |
| Q1b first Conv/last Gemm FP32 | 10.7MB, 약 14% 빠름 | cosine p01 0.970, pair error p95 0.0342 | 거부, quant tuning 종료 |
| CAM++ FP32 | 29.3MB, 3초 inference 약 2.45배 빠름 | model-specific H2 valid | fairness로 단독 채택 거부 |
| pyannote segmentation3 FP32 | 5.99MB | evidence contract valid | annotated SCD/OSD calibration 후보 |
| pyannote segmentation3 INT8 | 1.54MB | argmax 0.67~0.74, 느림 | 거부 |

INT8 artifact는 모두 `production_approved=false`, runtime ineligible 상태로 이력만 보존한다.

## 채택된 구조 변경

- cgroup-aware central ORT CPU session factory, quota 초과 thread fail-closed
- CLI/library 기본 ORT thread 1
- bounded NumPy PCM16 fast path
- H1/H2 single evaluation
- source-time cgroup quota/throttle/stage timing
- fairness-aware proxy evaluator와 span timeline digest
- official Silero temporal core/halo postprocessor는 default-off candidate
- pyannote FP32는 redacted evidence-only adapter; split/overlap assignment 권한 없음

## 다음 품질 데이터 최소 조건

같은 파일 threshold tuning은 중단한다. 다음 실행에는 최소 다음이 필요하다.

- 파일·세션 단위로 분리된 실제 1인/2인 Korean 녹음
- source-time speech mask와 speaker RTTM
- overlap annotation, 0.25초 DER collar policy
- MM/FF/MF, 8/16 kHz, near/far/noisy subgroup
- calibration/holdout speaker·session leakage 없음
- 조용한 화자와 1초 이하 turn을 의도적으로 포함

이 데이터가 들어오면 temporal VAD, cluster radius, CAM++ 보조 증거, pyannote SCD/OSD를 처음부터 같은 gate로 재평가한다. 현재 파일의 추가 threshold 조정은 하지 않는다.

## 0.4.0 후속 반증과 보강

독립 completion audit에서 빈 unsigned calibration PASS, 1µs 가짜 second label로
merge 회피, Windows `resource` import, 실행 불가능한 ORT no-telemetry recipe,
source/wheel 불일치를 찾았다. 다음을 수정했다.

- release-trusted `VerifiedCalibrationBinding` 외에는 QualityGate PASS 금지
- UEM-aware DER/JER·duration-floor merge/false-H2·SCD/OSD·subgroup scorer
- Windows import-safe RSS, ORT `--build_dir --no_telemetry`, production evidence gate
- current source를 포함한 `0.4.0` wheel 및 five-target development metadata 재생성

저음량·잡음 challenge 결과:

| 변형 | 기본 | challenger | 판단 |
|---|---|---|---|
| -12dB | H1 | global gain H2 | 독립 데이터 전 default off |
| 8k+-12dB | H1 | global gain H2 | 독립 데이터 전 default off |
| noise 20dB | H1 | RNNoise+ResNet H2 | timebase/native 승인 전 experimental |
| 정상 8k/canonical | H2 | gain exact no-op | 회귀 없음 |

RNNoise는 20dB noise proxy에서 assigned rate `42.37%`, assigned accuracy
`98.23%`, worst-speaker accuracy `95.10%`였지만 single derived artifact다. upstream
delay/flush를 반영한 adapter와 offline build plan은 추가했으나 source-time
authority와 native five-target 증거가 없어 release 권한은 없다.

## 0.4.0 persistent runtime

1 CPU/256MiB/network-none/read-only Linux arm64 container에서 3,171.732초 입력을
한 session으로 두 번 처리했다.

- cold/warm RTF `0.02667/0.02728`
- timeline digest 두 pass 및 0.3 canonical exact
- warm resident 증가 `2.874%`
- cgroup peak `249.1MiB`, throttle `0.0288%/0.0362%`

초기 warm gate가 누적 `memory.peak`를 resident로 잘못 비교해 false failure를
냈다. peak는 256MiB hard cap에만, warm leak는 current RSS/PSS/cgroup current에만
쓰도록 고치고 regression test 후 strict 10% gate를 재통과했다.

## STT 대체 결론

화자분리만의 5분/10분 환산은 약 8초/16초지만 end-to-end에서는 STT가
지배한다. 동일 5분 proxy에서 Whisper turbo Q5는 CER `15.90%`, wall
`629.91초`; SenseVoice INT8은 CER `24.35%`, decode `8.13초`였다. 정확도와
속도를 동시에 만족한 CPU-only 후보가 없어, 현재 운영 권고는 기존 STT를
유지하고 sddiar의 화자·단어 귀속을 추가하는 것이다. 구체 수치는
`research/CPU_STT_REPLACEMENT.md`에 분리했다.
