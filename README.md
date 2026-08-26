# sddiar

`SDD-SDDIAR-V1-001`을 구현하는 폐쇄망 CPU 기반 1~2인 화자분리
라이브러리다. `0.4.0`은 실제 Silero VAD와 WeSpeaker ONNX 화자분리뿐 아니라
8/16 kHz 입력 정규화, 외부 또는 로컬 STT 단어 시간축, 화자별 대본,
fail-closed 품질 판정을 하나의 orchestration 경로로 연결한다.

저장소는 [o2mandoo/sddiar](https://github.com/o2mandoo/sddiar)에 공개돼 있다.
음성·대본·ONNX 모델·wheel은 Git에 올리지 않고 hash-locked 반입물로만
관리한다.

## 구현 범위

- PCM16 mono 8/16 kHz WAV, WAVE_FORMAT_EXTENSIBLE, bounded canonical decode
- Silero VAD와 WeSpeaker ResNet34 FP32 CPU ONNX inference
- 제한된 H1/H2 clustering, deferred MICRO, whole-file sequence finalization
- 강제 귀속 대신 `UNKNOWN`, protected `OVERLAP`, whole-word speaker mapping
- UEM-aware DER/JER, merge/false-H2, SCD/OSD, worst-speaker, subgroup scorer
- signed calibration만 speaker-aware `PASS_*`를 열 수 있는 Quality Gate
- hash-verified model pack/release/lock/SBOM/no-telemetry build evidence gate
- cgroup v1/v2 CPU·throttle와 process-tree RSS/PSS/I/O 반복 worker
- `ProductionOrchestrator`: supplied words 또는 hash-bound local STT를 결합
- diarization 실패 시 STT를 버리지 않고 speaker-neutral transcript 보존
- local whisper.cpp adapter: argv-only, thread 1, no GPU/network/fallback, strict JSON
- 저음량 global gain과 RNNoise noise enhancement challenger(default off)
- Windows-safe import, Linux x86_64/arm64 및 macOS arm64 wheelhouse
- Windows x64 development wheelhouse와 macOS Intel source-build plan

## 화자분리 성능

Linux arm64 Docker를 정확히 1.00 CPU-equivalent, 256MiB,
`--network=none`, read-only root로 제한했다. 3,171.732초 입력을 설치된 0.4.0
wheel과 하나의 persistent session으로 두 번 처리했다.

| 항목 | cold | warm |
|---|---:|---:|
| wall | 83.715초 | 83.820초 |
| RTF | 0.02639 | 0.02643 |
| process-tree RSS | 159.53MB | 165.06MB |
| throttled wall ratio | 0.3671% | 0.0000% |

두 pass의 timeline digest는 서로 같고 0.3 canonical과도 일치했다. warm
resident 증가는 3.466%, cgroup peak는 248.4MiB였다. 이 수치는 Apple
Silicon 위 Linux cgroup-v2 proxy이며 실제 Xeon 6230R/cgroup-v1 증거가
아니다. 256MiB 여유가 약 7.6MiB뿐이므로 dedicated worker를 전제로 한다.

5분/10분 화자분리 단순 환산은 약 8초/16초다. STT는 포함하지 않는다.

## Whisper-only 흐름 대체 판단

기능적으로는 대체 가능하다. STT word timestamp를 받아 화자분리와 결합하고,
실패 시에도 화자 중립 대본을 반환한다. 그러나 기존 GPU large-v3 수준의
정확도와 속도를 Xeon 1 CPU에서 동시에 대체한 STT 후보는 아직 없다.

첫 5분 Clova proxy 비교:

| CPU STT 후보 | wall | CER | 판단 |
|---|---:|---:|---|
| Whisper base Q5 beam5 | 43.76초 | 39.53% | 정확도 탈락 |
| Whisper small Q5 beam5 | 138.91초 | 26.50% | 정확도 탈락 |
| Whisper turbo Q5 beam5 | 629.91초 | 15.90% | 속도 탈락 |
| SenseVoice INT8, 25초 chunk | 8.13초 | 24.35% | 빠른 초안 후보 |

따라서 즉시 적용안은 **기존 STT를 유지하고 sddiar 화자분리·단어 귀속을
추가하는 것**이다. CPU-only STT는 느린 품질 profile과 빠른 초안 profile로
분리하되 현재 둘 다 production 승인 상태가 아니다. 상세 근거는
[CPU STT 대체 검토](research/CPU_STT_REPLACEMENT.md)에 있다.

## 폐쇄망 개발 설치

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install \
  --no-index \
  --find-links artifacts/dev/wheels \
  --require-hashes \
  -r artifacts/dev/requirements-macos-arm64-cp311.lock
```

`sddiar-0.4.0-py3-none-any.whl` SHA-256:

```text
eda81dfe7ad265d2143ea465562bd9ee8d6646f774696e78f504f4e176fe5ea3
```

macOS arm64 fresh venv와 Linux x86_64 emulated clean install을 통과했다.
포함된 공식 ONNX Runtime wheel은 개발용이다. production에는 별도 승인된
`--no_telemetry` build가 필요하다.

## 화자분리 CLI

```sh
sddiar diarize INPUT_16K_MONO_PCM.wav \
  --silero-model /approved/models/silero_vad.onnx \
  --silero-sha256 EXPECTED_SHA256 \
  --wespeaker-model /approved/models/voxceleb_resnet34.onnx \
  --wespeaker-sha256 EXPECTED_SHA256 \
  --threads 1 \
  --output diarization.json
```

`--auto-gain-normalization`과 `--silero-temporal-postprocess`는 독립 annotation
승인 전에는 default off다. 단일 파일에서 선택한 assignment threshold를 다른
파일의 release 기본값으로 사용하면 안 된다.

## 검증

```sh
PYTHONPATH=src python3.11 -m py_compile src/sddiar/*.py
PYTHONPATH=src python3.11 -m unittest discover -s tests -v
PYTHONPATH=src python3.11 scripts/verify_offline_release.py release \
  --production --scan-source src/sddiar
```

2026-08-26 기준 CPython 3.11 개발 runtime에서 285개 test가 통과했다.
`src/sddiar`, `scripts`, `bench/one_cpu` static zero-network scan은 issue 0건이다.
production release root가 아직 없으므로 production 검증은
`RELEASE_ROOT_MISSING`으로 fail-closed되는 것이 정상이다.

## 상태와 근거

- [1-CPU goal](GOAL_1CPU_PRODUCTION.md)
- [독립 Sol 연구 종합](research/1CPU_SOL_RESEARCH_SYNTHESIS.md)
- [CPU STT 대체 검토](research/CPU_STT_REPLACEMENT.md)
- [Xeon 실행 절차](docs/XEON_ONECPU_RUNBOOK.md)
- [annotation 반입 형식](docs/ANNOTATION_INTAKE.md)
- [RNNoise challenger](docs/RNNOISE_EXPERIMENTAL_LANE.md)
- [구현 현황](IMPLEMENTATION_STATUS.md)
- [실제 음성 proxy 결과](experiments/260824_clova_proxy/RESULT.md)
- [artifact intake gate](docs/ARTIFACT_INTAKE_GATE.md)
- [전체 SDD](SDD_offline_cpu_speaker_diarization_v1_ko.md)
