# Goal — 1 CPU 운영형 화자분리

상태: `ACTIVE / LOCAL_040_REPEAT_PASS / STT_DROP_IN_REPLACEMENT_REJECTED / EXTERNAL_RELEASE_GATES`  
기준일: 2026-08-26  
대상: Intel Xeon Gold 6230R급 x86_64, cgroup v1 `100000/100000`, 1.00 CPU-equivalent, 폐쇄망

## 최종 목표

1~2인 장시간 대화를 1 CPU에서 안전하게 처리하고, 불확실성을 숨기지 않으면서 실제 speech의 화자 귀속률과 짧은 교대·조용한 화자·중첩 구간 품질을 운영 가능한 수준으로 만든다. 기존 STT word timeline 또는 승인된 local STT를 결합해 화자별 대본까지 반환하되, diarization 실패 시 중립 대본을 보존한다. M5 Max 성능은 개발 참고값일 뿐이며 최종 승인은 실제 Xeon cgroup에서 수행한다.

## 고정 원칙

- complete merge와 false second-speaker를 평균 점수로 상쇄하지 않는다.
- `UNKNOWN`을 무조건 채워 coverage를 높이지 않는다.
- VAD, embedding, H1/H2, decoder, SCD/OSD 변경은 각각 분리 A/B한다.
- calibration과 holdout은 파일·세션 단위로 분리한다.
- QualityGate의 speaker-aware `PASS_*`는 서명과 provenance를 검증한 `VerifiedCalibrationBinding`에서만 허용한다. profile ID만 있는 객체, raw/legacy profile, unsigned profile은 모두 `REVIEW_REQUIRED`다.
- verified binding은 비어 있지 않은 threshold/model hash, dataset manifest/scorer/config SHA-256, 지원 sample rate, approver, signer, calibration version과 split/selection provenance를 요구하고 현재 profile/model/sample-rate/config 진단에 묶는다. 개발용 digest verifier는 release PASS 권한을 만들지 않는다.
- runtime은 network/download/fallback 없이 hash-verified local artifact만 사용한다.
- CPU 절감은 품질 모델·SCD/OSD 예산으로 재투자하고 안전 gate를 제거하는 데 쓰지 않는다.

## QualityGate calibration authority contract

- `CalibrationProfile`과 기존 `CalibrationBinding`은 호환용 data/inspection 객체다. `valid=True`여도 QualityGate의 PASS 권한은 없다.
- PASS 권한은 release-trusted signature verifier가 canonical payload를 검증해 만든 immutable `VerifiedCalibrationBinding`에만 있다. constructor 직접 호출, duck-typed profile ID, subclass, 개발용 digest verifier는 거부한다.
- signed payload에는 schema/calibration version, nonempty threshold와 exact model SHA-256 map, dataset manifest/scorer/config SHA-256, supported source rates, approver/signer, annotation·pipeline·model-pack·selection/safety provenance가 들어간다.
- binding 생성 시 current model hash map, source rate, config hash를 exact match하고, QualityGate 호출 시 diagnostics의 calibration profile/model/rate/config identity를 다시 exact match한다. 하나라도 없으면 `Q_CALIBRATION_UNBOUND`, 다르면 해당 mismatch reason으로 `REVIEW_REQUIRED`다.
- `confirmed_hard_out_of_profile`은 negative evidence이므로 calibration 없이도 speaker-neutral `UNSUPPORTED`를 낼 수 있다. speaker-aware `PASS_HIGH`/`PASS_STANDARD` 예외는 없다.

## 현재 1-CPU 기준선

| 항목 | B0 FP32 | C1 PCM fast path + H2 single pass |
|---|---:|---:|
| 3,171.732초 wall | 120.844초 | 85.210초 |
| RTF | 0.038100 | 0.026866 |
| peak process RSS | 180.28MB | 154.45MB |
| quota utilization | 99.63% | 99.49% |
| throttled wall ratio | 0.0104% | 0.0038% |
| 결과/metric | canonical과 동일 | B0와 동일 |

설치된 최종 `0.4.0` wheel과 persistent session으로 두 번 처리한 strict gate는 cold/warm wall `83.715/83.820초`, RTF `0.02639/0.02643`, process-tree RSS `159.53/165.06MB`, warm resident 증가 `3.466%`였다. timeline digest는 두 pass 및 0.3 canonical과 같았다. cgroup peak는 `248.4MiB`로 256MiB 제한을 통과했지만 headroom이 약 7.6MiB라 dedicated worker가 필수다. 이는 Linux arm64 Docker quota proxy이며 Xeon 성능 증거가 아니다.

## 현재 품질 병목

- Clova proxy는 다음 turn 시작까지의 침묵도 화자 시간으로 포함하므로 40.43% timeline coverage를 VAD recall이나 DER로 해석하지 않는다.
- system-detected speech 1,487.040초 중 1,282.096초, 즉 86.22%가 speaker로 귀속됐다.
- `UNKNOWN` 204.944초 중 MICRO가 119.008초다. 현재 358개 deferred MICRO가 모두 `UNKNOWN_SHORT`가 된다.
- 조용한 proxy 화자 `REF_00`의 assigned rate는 22.27%, `REF_01`은 52.38%다. UNKNOWN rate도 33.21% 대 6.15%로 불균형하다.
- 실제 SCD/OSD가 없어 연속 발화 내 speaker change와 overlap은 아직 검증되지 않았다.
- 독립 annotation 없이 같은 녹음의 proxy를 더 조정할 수 없다.

## robustness challenger 결과

- global gain v2는 RMS `<0.01`, 최대 4배, peak 0.99, activation 1.25배 deadband다.
- -12dB와 8k+-12dB 변형은 H1→H2로 복원했고 canonical/정상 8k는 exact no-op이었다.
- noise 20dB는 gain, temporal VAD, CAM++만으로 복원되지 않았다.
- RNNoise+ResNet은 noise 20dB에서 H1→H2, assigned rate `42.37%`, worst-speaker accuracy `95.10%`를 보였다.
- RNNoise source-time/resampler/native five-target evidence가 미승인이라 default off이며 release 권한이 없다.

## 단일 파일 개발 회귀 gate

이 gate는 현재 파일에 대한 파괴 방지용이며 release 품질 주장이 아니다.

- `H2_CONFIRMED`, complete merge `0`
- H2 separation `>= 0.714`, outlier ratio `<= 0.088`, stability `>= 0.99`
- assigned-time accuracy `>= 98.5%`
- covered-turn accuracy `>= 93%`
- decoder 개선 후보: full proxy coverage `>= 43%`, holdout coverage `>= 42%`, holdout assigned accuracy `>= 99%`
- 기존 assigned label 변경 시간 `<= 0.5%`
- FP32 계산 최적화는 span timeline digest와 metric이 기준선과 정확히 같아야 한다.
- 품질 개선의 1-CPU RTF는 직전 기준의 `1.05x` 이하여야 한다.

## 독립 Korean release gate

- 알려진 2인 파일마다 complete merge `0`
- 1인 파일의 false second-speaker duration `<= 1%`
- attributed non-overlap speech coverage `>= 85%`
- assigned-time speaker accuracy `>= 95%`
- turn coverage / covered-turn accuracy `>= 90% / 95%`
- DER `<= 15%`, JER `<= 25%` (`0.25초` collar, overlap 포함)
- SCD F1 `>= 0.75` (`±0.5초`), OSD precision/recall `>= 0.75/0.60`
- MM/FF/MF, 8/16 kHz, near/far/noisy subgroup가 macro보다 `5%p` 넘게 낮으면 실패
- worst-speaker assigned coverage와 accuracy를 별도 보고하고 평균으로 숨기지 않는다.

## 실제 Xeon 1-CPU 성능 gate

- cgroup v1 quota/period `100000/100000`, shares `1024`, cpuset `0-103` 확인
- 모든 ORT session `CPUExecutionProvider`, intra/inter `1/1`, sequential, spinning off
- worker당 audio job 동시성 `1`
- full quality stack RTF p95 `<= 0.25`, max `<= 0.35`; hard completion ceiling `<= 1.0`
- peak process-tree RSS `<= 256MiB`, 10분→60분 baseline 증가 `<= 10%`
- throttled wall ratio `<= 1%`, OOM/fallback/output corruption `0`
- cold/warm, cgroup CPU, process-tree RSS/PSS, stage wall/CPU, I/O를 함께 기록
- full-hour 5회 이하는 median/max로 보고하고, release p95는 20~30회 뒤 주장한다.

## STT 통합과 대체 gate

- `ProductionOrchestrator`는 supplied word timeline과 hash-bound local STT를 지원한다.
- STT 성공 후 diarization 실패 시 모든 word를 `UNKNOWN`으로 내려 중립 대본을 보존한다.
- whisper.cpp adapter는 exact binary/model hash, thread/process 1, no GPU/network/fallback, strict JSON을 강제한다.
- 첫 5분 proxy에서 turbo Q5는 CER `15.90%`지만 wall `629.91초`, SenseVoice INT8은 wall `8.13초`지만 CER `24.35%`였다.
- CPU-only 후보 중 기존 large-v3 품질과 빠른 SLA를 동시에 만족한 후보는 없다.
- 현재 권고는 기존 STT를 유지하고 sddiar word-speaker attribution을 추가하는 것이다.
- local STT 승격은 현재 GPU large-v3 대비 CER `+1%p`, SA-WER `+2%p` 이내, timestamp calibration, actual Xeon gate를 모두 요구한다.

## 실행 순서

1. 안전한 계산 최적화: cgroup-aware ORT factory, ndarray PCM, H2 single pass, persistent session
2. frozen-H2 decoder ablation: MICRO abstention/soft emission/transition
3. ResNet34 static QDQ INT8 candidate와 FP32 parity
4. Silero hysteresis core/halo 및 v6 sequence A/B
5. centroid-conditioned selective SCD와 approved OSD evidence
6. 독립 Korean calibration/holdout
7. 실제 Xeon cgroup-v1 및 Linux x86_64 폐쇄망 bundle 검증
8. 기존 GPU large-v3 output anchor와 local STT non-inferiority 검증

완료는 위 release 품질·성능 gate와 production artifact 승인까지 모두 통과했을 때만 선언한다.

같은 녹음의 decoder/MICRO/CAM++/per-cluster threshold 탐색은 모두 calibration gate에서 fail-closed됐으며 추가 same-file tuning은 중단했다. 로컬에서 해결할 evaluation/runtime/package 결함은 보강했지만, 목표 완료에는 독립 Korean speech mask/RTTM/overlap/word-time annotation, 실제 Xeon context, native OS별 실행과 승인된 release artifact가 필요하다.
