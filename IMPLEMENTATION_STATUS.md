# 구현 현황 — P0~P6 개발 구현

기준 문서: `SDD_offline_cpu_speaker_diarization_v1_ko.md`

## 구현됨

- source-time contract, deterministic artifact ID, time-warp, word provenance
- strict UNKNOWN/OVERLAP word attribution과 summary-safe Quality Gate
- protected overlap을 별도로 보존하는 tracklet builder
- H1/H2 constrained clustering, leave-block-out stability, temporal interleaving guard
- MICRO 보류 및 strict 재평가, stable/recent centroid drift guard, tracklet-level Viterbi finalization
- model-pack hash/signature/runtime/platform fail-closed reference verifier
- session/split leakage, merge risk, word/MICRO safety evaluation primitives
- `EvidencePipeline`: local embedding provider를 주입해 위 구성요소를 whole-file 순서로 연결
- WAV PCM/WAVE_FORMAT_EXTENSIBLE frontend와 개발 fixture용 Energy VAD
- bounded WAV chunk decode/resample path, VAD/probe/approved SCD-OSD evidence segmentation seam
- pre-bundled ONNX/Silero embedding/VAD capability boundary 및 CPU-only fail-closed 검사
- calibration profile binding, RTF/RSS/repeated-run measurement helper
- four-target release layout/hash/static offline policy validator
- local idempotent worker, atomic result publishing, summary policy, injected GenOS port/stub
- canonical public-result serializer that rejects raw embedding/centroid state
- session-scoped human correction that can modify only UNKNOWN source intervals
- local SHA-verified Silero VAD와 WeSpeaker ResNet34 FP32 actual ONNX CPU runtime
- strict `kaldi-native-fbank` frontend: 16 kHz, 80-bin, 25/10 ms, Hamming, dither 0, snip-edges, utterance CMN
- TorchAudio Kaldi FBank direct golden: 합성/실제 2초 `(198,80)`, max abs `2.73e-4`, embedding cosine `1.0`
- `LocalOnnxDiarizer` 고수준 API와 `sddiar diarize` CLI, redacted/atomic JSON output
- macOS arm64 CPython 3.11 hash-locked development wheelhouse와 clean offline install
- development CycloneDX SBOM과 third-party notice inventory; unsigned/internal-review 상태 명시
- 3,171.732초 실제 WAV의 전체 실행, Clova timing proxy calibration/holdout, 반복 RTF/RSS profile
- upstream 1.5초 tracklet 및 1.5/0.75초 subsegment A/B; 안전/성능 열화로 기본값 미채택
- cgroup v1/v2 quota·throttling 계측과 중앙 ORT CPU budget factory; 1 CPU에서 threads>1 fail-closed
- bounded NumPy PCM16 fast path, H1/H2 single pass, persistent WeSpeaker session, audio job 직렬화
- 1-CPU Linux arm64 `--network=none` Docker benchmark와 실제 Xeon Gold 6230R cgroup-v1 preflight
- Linux arm64/x86_64 CPython 3.11 development wheelhouse/lock 및 x86_64 emulated clean install
- fairness-aware proxy scorer: worst-speaker accuracy/coverage/UNKNOWN gap을 calibration에 포함
- frozen-H2 decoder, MICRO-only fixed window, CAM++ rescue, per-cluster ceiling의 fail-closed A/B
- Silero v6.2.1 official hysteresis core/halo postprocessor(default off)
- pyannote segmentation3 FP32 redacted SCD/OSD evidence adapter; INT8 parity rejection, direct split 권한 없음
- ResNet34 Q1/Q1b static INT8 parity rejection과 CAM++ standalone fairness rejection 기록
- strict annotation intake: RTTM/UEM/word JSONL, hash/path/privacy/split/subgroup 검증
- UEM-aware DER/JER, duration-aware merge/false-H2, SCD/OSD, word/MICRO, bootstrap scorer
- `VerifiedCalibrationBinding` 외에는 speaker-aware PASS를 열 수 없는 signed quality authority
- Windows import-safe runtime과 Windows x64 wheelhouse, macOS Intel availability/source-build plan
- ORT `--no_telemetry`/`--build_dir` pinned build plan과 production evidence validator
- 저음량 global gain challenger: -12dB/8k+-12dB H1→H2 복원, 정상 입력 exact no-op
- RNNoise default-off challenger: noise 20dB H1→H2 proxy 복원, source-time authority 미승인
- `ProductionOrchestrator`: 8/16k canonical input, supplied/local STT, word mapping, neutral fallback
- hash-bound whisper.cpp local backend와 privacy-safe Korean CER/WER/SA-WER evaluation
- 0.4.0 persistent 1-CPU worker: 105.7분 처리, deterministic digest, warm memory gate

## production 전 아직 필요한 항목

- FFmpeg/MP3/FLAC production decoder 또는 명시적인 WAV/FLAC-only 운영 결정, approved resample profile
- 독립 Korean annotation을 이용한 pyannote FP32 SCD/OSD calibration과 overlap 성능 검증
- 현재 GPU large-v3 실제 출력과 동일 입력의 STT non-inferiority 비교
- local STT word start/end timestamp calibration과 signed scorer receipt
- audited `--no_telemetry` ORT build, model weight 법무 승인, signed manifest, SBOM, third-party notice
- Windows x64, macOS x64 target wheel/bundle과 실제 OS별 clean install·restart·API parity
- 남성-남성/여성-여성/남녀, 8/16 kHz, overlap, third-party, long-file multi-recording calibration/independent holdout
- 실제 Xeon Gold 6230R 1.00-CPU cgroup-v1의 process-tree RSS, cold/warm 반복, full-hour profile
- 실제 GenOS/service adapter, persistence와 manual correction UI

## 다음 구현 gate

1. 현재 macOS 개발 candidate의 WeSpeaker/Silero weight 사용·재배포 조건과 `kaldi-native-fbank` notice를 승인한다.
2. 여러 독립 Korean 파일의 speech mask/RTTM/overlap annotation을 calibration/holdout으로 분리하고 8 kHz·성별 subgroup을 채운다. 현재 파일의 threshold tuning은 중단했다.
3. 그 calibration으로 signed profile을 생성한 뒤 `UNKNOWN` coverage/precision, complete merge, DER/JER와 false PASS를 동결한다.
4. 최소 지원 CPU를 포함한 P4 반복 profile을 수행하고, 승인된 telemetry-free ORT로 네 target bundle을 만든다.
5. 네 OS의 network-disabled clean install/restart/API/schema parity를 통과한 뒤에만 P5 release를 승인한다.

## 현재 외부 gate 증빙

- `artifacts/dev`에는 해시 고정된 Silero v6.2.1, ResNet34/CAM++, rejected INT8, pyannote evidence model, `sddiar-0.4.0` wheel과 offline lock/SBOM/notice가 있다. `production_approved`는 명시적으로 `false`다.
- Linux arm64 1-CPU proxy image와 Linux x86_64 wheelhouse는 개발 검증을 통과했다. 공식 ORT이므로 production no-telemetry 승인을 대신하지 않는다.
- 동일 manifest는 `ModelPackVerifier(development_mode=True)`로 model hash/runtime/target을 통과하며, production mode에서는 서명이 없어 `MANIFEST_SIGNATURE_INVALID`로 거부된다.
- production signed model/release manifest, SBOM, notice, telemetry-free ORT와 four-target release root는 아직 없다.
- `scripts/verify_offline_release.py release --scan-source src/sddiar`의 `RELEASE_ROOT_MISSING`은 production release 부재를 fail-closed로 탐지한 정상 결과다.
- static zero-network scan은 `src/sddiar`, `scripts`, `bench/one_cpu` 모두 issue 0건이다. Clova timing artifact도 원본 filename 대신 content hash ID만 보존한다.

## 실행 검증

2026-08-26 기준 CPython 3.11 개발 runtime test 285개가 전부 통과했다. Windows `resource`/`_winapi` 부재 모의 import도 통과했다.

`sddiar-0.4.0-py3-none-any.whl` SHA-256은 `eda81dfe7ad265d2143ea465562bd9ee8d6646f774696e78f504f4e176fe5ea3`다. macOS arm64 fresh venv와 Linux x86_64 emulated hash-locked clean install·import를 통과했다.

정확한 1.00-CPU quota baseline은 wall `120.844초`, RTF `0.03810`, RSS `180.28MB`였다. NumPy PCM + H2 single pass 후 wall `85.210초`, RTF `0.02687`, RSS `154.45MB`로 29.5% 개선됐고 H2/span/metric은 동일하다. Silero temporal ResNet challenger는 RTF `0.02892`로 full/holdout coverage `42.95%/42.45%`와 turn 품질을 개선했으나 holdout worst-speaker accuracy `94.18%`로 95% gate를 넘지 못해 default-off다.

Clova proxy holdout에서는 turn coverage `70.42%`, covered-turn accuracy `94.00%`, reference timeline coverage `39.42%`, assigned-time speaker accuracy `99.74%`였다. 이는 동일 녹음 내부 split의 선택적 정확도이며 독립 release 품질 증거가 아니다. Quality Gate는 올바르게 `REVIEW_REQUIRED`를 반환한다.

동일 파일의 decoder/MICRO/CAM++/cluster-ceiling 총 A/B는 모두 fail-closed로 canonical을 유지했다. `STOP_SAME_FILE_THRESHOLD_TUNING_DATA_BLOCKER`가 현재 품질 상태이며 다음 threshold 선택에는 독립 annotation이 필요하다.

최종 0.4.0 wheel을 설치한 Linux arm64 1-CPU/256MiB container에서 같은 장시간 입력을 한 persistent session으로 두 번 처리했다. cold/warm RTF는 `0.02639/0.02643`, timeline digest는 서로 및 0.3 canonical과 동일했고, warm resident 증가는 `3.466%`, cgroup peak는 `248.4MiB`였다. 이는 cgroup-v2 local proxy이지 Xeon/cgroup-v1 release 증거가 아니다.

Whisper-only 대체 실험에서 5분 Clova proxy CER는 base Q5 `39.53%`, small Q5 `26.50%`, turbo Q5 `15.90%`, SenseVoice INT8 `24.35%`였다. turbo는 5분 처리에 `629.91초`, SenseVoice는 `8.13초`였다. 정확도와 속도를 동시에 만족한 CPU-only 후보가 없어, 현재 권고는 기존 STT를 유지하고 sddiar 화자 귀속을 추가하는 것이다.

```sh
PYTHONPATH=src python3.11 -m py_compile src/sddiar/*.py
PYTHONPATH=src python3.11 -m unittest discover -s tests -v
```
