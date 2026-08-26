# 실제 runtime artifact intake gate

작성일: 2026-08-26

runtime 코드는 모델·wheel을 다운로드하지 않는다. macOS arm64와 Linux arm64/x86_64 CPython 3.11용 개발 candidate가 해시 고정되어 있지만 모두 `production_approved: false`다. 아래 production evidence가 승인되기 전에는 운영 품질이나 four-target offline release 성공을 주장하지 않는다.

## 반입 전 필수 항목

| 구분 | 필요한 artifact | 반입 전 확인 | 현재 상태 |
|---|---|---|---|
| ORT | Windows x64, Linux x64, macOS arm64, macOS x64용 CPython 3.11 CPU wheel | exact build ID/version/SHA-256, CPU EP only, no-telemetry build attestation, ABI·linked library report | macOS arm64, Linux arm64/x86_64 official 1.29.0 개발 wheel 반입·clean install; production no-telemetry, Windows/macOS x64 미완료 |
| VAD | approved Silero VAD ONNX와 input/state contract | model hash, weight terms, 8/16 kHz input/state reset contract, ORT compatibility | 개발 candidate 반입·stateful 전체 실행 완료; weight 내부 승인 미완료 |
| embedding | WeSpeaker ResNet34 FP32 ONNX development candidate | exact downloaded artifact hash, inspected I/O/opset/dimension, FBank contract, weight redistribution/attribution approval | 개발 candidate 반입·256-D CPU inference 완료; VoxCeleb weight 승인 미완료 |
| feature | `kaldi-native-fbank` target wheel | 80-bin/25 ms/10 ms/Hamming/dither 0 parity, SHA-256, Apache-2.0 notice, ABI matrix | macOS arm64와 Linux arm64/x86_64 1.22.3 반입; TorchAudio direct parity(max abs `2.73e-4`, embedding cosine `1.0`) 완료 |
| fallback comparison | existing pyannote baseline pack | existing pack hash, CPU runtime profile, gated model terms | 미반입/미연결 |
| decode | FFmpeg target binaries 또는 WAV/FLAC-only operational decision | build configuration, codec license/notice, hash, native dependencies | WAV PCM reference만 구현 |
| evaluation | supported/challenge/unsupported source-time annotations | split leakage audit, male-male/8 kHz/overlap/third-party coverage, handling approval | 실제 1개 녹음 Clova timing proxy와 synthetic harness만 완료; independent set 미완료 |
| STT/GenOS | internal STT word-time contract 및 GenOS service port | no public network route, auth/retention/idempotency agreement, test endpoint | protocol/stub만 구현 |

## STT OpenVINO 4-arm 추가 반입 묶음

60초 baseline/VAD A/B는 완료했지만 VAD가 품질 gate를 실패했다. 다음 후보인
OpenVINO encoder와 VAD+OpenVINO arm은 아래 항목을 하나의 Linux x86_64
candidate pack으로 반입한 뒤에만 실행한다.

- commit `371b5a7561823ab2bb32142d2751e35e7534727b`에서
  `WHISPER_OPENVINO=ON`, GPU/BLAS/Metal off로 빌드한 `whisper-cli`
- 동일 commit의 CPU-only binary와 두 build의 normalized CMake config/diff
- `large-v3-turbo` FP32 source checkpoint hash와 conversion script hash
- `ggml-large-v3-turbo-encoder-openvino.xml/.bin`의 개별 SHA-256
- 변환에 사용한 CPython 3.11 wheel lock 전체: OpenVINO, openai-whisper,
  torch, transformers 및 모든 전이 dependency
- 실행용 OpenVINO runtime shared-library tree hash, ABI/link report, SBOM,
  license/notice, no-network 재기동 증거
- 기존 turbo Q5와 Silero v6.2.0 GGML VAD hash
- exact 60초 clip hash와 redacted Clova proxy reference hash

OpenVINO external encoder는 `-t 1`만으로 내부 inference thread 1을 보장하지
않는다. 따라서 실제 Xeon Gold 6230R cgroup-v1 quota `100000/100000`에서
실행하거나, development host에서 `CPU seconds / wall seconds <= 1.05`를
통과한 trial만 1-CPU-equivalent로 인정한다. 두 반복의 text/timeline hash가
다르거나 external encoder load marker가 없으면 해당 arm 전체를 무효 처리한다.

## 확인한 외부 사실과 해석

- ONNX Runtime 공식 build는 telemetry가 기본 활성화이며, 공식 privacy 문서는 private build에서 `--no_telemetry`를 지원한다고 명시한다. 따라서 production model pack에는 회사 build 또는 동등한 audited vendor artifact가 필요하다. [ORT Privacy](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.29.0/docs/Privacy.md)
- WeSpeaker upstream은 runtime ONNX model을 제공하며, 문서는 VoxCeleb 기반 pretrained weight가 해당 데이터셋의 CC-BY-4.0 조건을 따른다고 설명한다. 이 사실은 개발 candidate 판단용이며, 사내 재배포/상업 이용 approval을 대체하지 않는다. [WeSpeaker pretrained models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- Silero VAD ONNX는 code/model artifact, exact input contract, weight terms를 **같은 pack intake**에서 검증해야 한다. repository code license만으로 production model-weight approval을 가정하지 않는다.

## 반입 절차

1. isolated ingress host에서 원본 URL, publisher revision, download timestamp, SHA-256을 기록한다.
2. 모델/ORT/FFmpeg artifact와 관련 config·external tensor·notice를 하나의 candidate pack에 넣는다.
3. `ModelPackVerifier`로 file hash, signature, path traversal, runtime/EP contract를 검증한다.
4. approved CPython 3.11 target host에서 golden I/O contract, CPU-only provider, restart, read-only installation을 검증한다.
5. egress/DNS 차단 clean environment에서 `scripts/verify_offline_release.py`와 actual target release validation을 실행한다.
6. P1 development fixture → P3 calibration → release holdout 순서로 평가한다. holdout 결과를 보고 threshold를 조정하지 않는다.

## 현재 개발 candidate

- manifest: `artifacts/dev/manifest.json`
- offline lock: `artifacts/dev/requirements-macos-arm64-cp311.lock`
- library wheel: `artifacts/dev/wheels/sddiar-0.4.0-py3-none-any.whl`
- development SBOM/notice: `artifacts/dev/sbom.cdx.json`, `artifacts/dev/THIRD_PARTY_NOTICES.md`
- Silero SHA-256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
- WeSpeaker SHA-256: `9fea6516d7ad6bf0a76c7689f5a49b65d330fad6dde96c91bb4435ffbfe056a1`
- strict FBank wheel SHA-256: `29452f2900e771086e9022dde17a92d191217ab3e34ca7dc361bd9be53e94fb4`

새 가상환경에서 `--no-index --require-hashes` 설치와 CLI import는 macOS arm64, Linux arm64/x86_64에서 성공했다. 전체 실제 파일은 macOS와 Linux arm64 1-CPU proxy에서 처리했고, Linux x86_64는 emulated install smoke만 수행했다.
Development manifest는 model hash/runtime/target 검증을 통과하지만, production verifier에서는 서명 부재로 의도적으로 거부된다.

## 반입 후 바로 실행할 명령

```sh
# macOS arm64 development lock clean install
python3.11 -m pip install --no-index --find-links artifacts/dev/wheels \
  --require-hashes -r artifacts/dev/requirements-macos-arm64-cp311.lock

# model pack hash/runtime validation (production caller supplies a signed manifest)
sddiar validate <signed-result-or-manifest-json>

# target release layout and static offline policy scan
PYTHONPATH=src python3.11 scripts/verify_offline_release.py release --scan-source src/sddiar

# reference regression suite
PYTHONPATH=src python3.11 -m unittest discover -s tests -v
```

반입 URL은 provenance metadata에만 존재한다. runtime과 release validator에는 네트워크·자동 다운로드 경로가 없으며, production pack은 위 잔여 승인 항목을 충족해야 한다.
