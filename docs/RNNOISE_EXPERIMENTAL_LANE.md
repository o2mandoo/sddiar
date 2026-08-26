# RNNoise 잡음 억제 실험 lane

상태: `EXPERIMENTAL / DEFAULT_OFF / PRODUCTION_APPROVED_FALSE`

이 lane은 20 dB noisy challenge 1건에서 관찰한 H1→H2 변화를 재현·검증하기 위한
전처리 A/B다. 운영 기본 경로, release QualityGate 권한, Xeon 성능 증거가 아니다.
기존 패키지 버전·wheel·기본 diarization 경로는 변경하지 않는다.

## 고정 provenance

- upstream: `https://gitlab.xiph.org/xiph/rnnoise.git`
- source commit: `70f1d256acd4b34a572f999a05c87bf00b67730d`
- `model_version` 및 model tar SHA-256:
  `0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37`
- source license: `BSD-3-Clause`
- 실험 mode: upstream `rnnoise_demo`, native scalar, raw signed 16-bit mono 48 kHz

BSD-3-Clause 확인은 source code 범위다. model tar의 반입·재배포 승인 범위를 code
license에서 자동 추론하지 않는다. model tar는 별도 hash·법무 검토 대상으로 남긴다.

## 오프라인 source build

[`scripts/build_rnnoise_offline.py`](../scripts/build_rnnoise_offline.py)는 build를 대신
다운로드하지 않는다. 다음 다섯 target의 argv plan과 단일 target attestation을 만든다.

- Windows x86_64
- Linux x86_64 / aarch64
- macOS x86_64 / arm64

각 target은 실제 target-native host에서 따로 build해야 한다. cross-build 결과나 한
host의 plan 생성은 네 플랫폼 native 검증이 아니다.

### 반입 gate

폐쇄망 진입 전에 아래를 별도 매체로 반입하고 SHA-256을 검증한다.

1. exact commit의 source archive와 clean checkout
2. recursive submodule status; submodule이 없으면 canonical empty-list SHA-256을 기록
3. pinned model tar
4. target별 compiler/sysroot 또는 SDK, Autoconf, Automake, Libtool, Make를 열거한
   hash-verified toolchain manifest
5. build 뒤 native `rnnoise_demo`, build log, linked-dependency report

upstream `autogen.sh`는 실행 금지다. 해당 스크립트가 `download_model.sh`를 호출해
model을 자동 다운로드하기 때문이다. 대신 다음 순서를 사용한다.

```text
python3.11 scripts/build_rnnoise_offline.py stage-model \
  --model-tar <imported/rnnoise_data-0a8755...tar.gz> \
  --source-root <imported/rnnoise-checkout>

autoreconf -isf
./configure --disable-doc --enable-examples --disable-x86-rtcd \
  --disable-shared --enable-static
make -j1 examples/rnnoise_demo
```

Windows는 target-native MSYS2/MinGW-w64 toolchain에서 helper가 출력하는
`sh ./configure --host=x86_64-w64-mingw32 ...` argv를 사용한다. build volume의
network/DNS 차단은 외부 OS/container gate로 증명해야 한다. helper의 정적 plan만으로
native executable의 무통신을 주장하지 않는다.

`--disable-x86-rtcd`는 x86 runtime dispatch만 끈다. compiler가 정의하는 SSE2/AVX 또는
ARM NEON compile-time path까지 scalar로 만드는 option은 아니다. 따라서 이 build plan은
`target_compiler_default_not_scalar_claim`으로 기록하며, 단일 local 실험의
`native_scalar` 관찰을 다섯 target build에 일반화하지 않는다.

```text
python3.11 scripts/build_rnnoise_offline.py plan --output rnnoise-plan.json

python3.11 scripts/build_rnnoise_offline.py attest \
  --target linux-x86_64 \
  --source-root <checkout> \
  --source-archive <source-archive> \
  --source-archive-sha256 <approved-ingress-sha256> \
  --model-tar <model-tar> \
  --toolchain-manifest <toolchain.json> \
  --toolchain-manifest-sha256 <sha256> \
  --build-log <build.log> \
  --build-log-sha256 <sha256> \
  --dependency-report <dependencies.json> \
  --dependency-report-sha256 <sha256> \
  --native-binary <rnnoise_demo> \
  --output rnnoise-attestation.json

python3.11 scripts/build_rnnoise_offline.py verify-attestation \
  rnnoise-attestation.json \
  --expected-sha256 <approved-attestation-file-sha256>
```

attestation은 local path를 저장하지 않고 모든 approval을 false/not-run으로 유지한다.
내부 input/output hash chain과 caller-supplied outer file SHA-256은 변조 검출용이며,
signature나 binary lineage의 독립 검증이 아니다. verifier 결과 scope도
`caller_hash_bound_structure_only / cryptographic_authenticity=not_verified`로 고정된다.
resampler는 RNNoise source build의 일부가 아니며, target별 native binary·source·license·
toolchain·SHA-256 attestation을 별도로 요구한다.
RNNoise와 resampler 실행 파일은 private job directory로 복사한 뒤 다시 hash하고 실행한다.
따라서 배포 binary는 system library 외 dependency를 자체 해결하는 static/self-contained
형태여야 하며, linked-dependency report에 누락된 DLL/dylib/so fallback이 없어야 한다.

## runtime seam

[`src/sddiar/rnnoise_experimental.py`](../src/sddiar/rnnoise_experimental.py)의
`ExperimentalRNNoisePreprocessor`는 context-managed prepared waveform 하나를 만든다.

```text
source PCM16 mono 16 kHz WAV
  -> fixed local resampler argv: exact N samples -> exact 3N samples at 48 kHz
  -> zero-pad to 480-sample boundary and append one 480-sample zero flush frame
  -> pinned local rnnoise_demo argv
  -> discard padded tail and trim to exact 3N; no prefix shift
  -> fixed local resampler argv: exact 3N -> exact N at 16 kHz
  -> private exact-length PCM16 mono WAV
  -> same prepared waveform for VAD chunks and embedding region reads
```

RNNoise는 한 frame 지연된 `delayed_X`를 합성한다. upstream demo가 기록하지 않는 첫
480-sample output은 source frame 0이 아니라 초기 all-zero delayed frame이다. adapter는
마지막 signal/pad frame을 출력시키기 위해 zero input frame 하나를 append하고, 결과의
padded tail만 자른다. 입력 sample 수 `N`, 출력 sample 수 `N`, source duration
microseconds가 하나라도 다르면 결과를 폐기한다. 이는 duration/sample-count 계약이다.

resampler group delay는 output 길이만으로 증명할 수 없다. enabled lane은 binary hash에
묶인 `rnnoise-resampler-timebase-proof` JSON에서 impulse/first-marker/last-marker shift 0과
tail frame case를 요구하지만, 현재 그 문서는 caller-hash-bound structural record일 뿐
독립 서명·review가 아니다. 따라서 `PreparedRNNoiseAudio.source_time_authorized`는
`False`, receipt의 mapping은
`SAMPLE_COUNT_PRESERVED_LATENCY_NOT_INDEPENDENTLY_VERIFIED`다. 이 상태의 waveform은
redacted proxy A/B에만 쓰고 source-time public span을 내보내면 안 된다. 독립 verifier와
real-binary impulse/chirp evidence가 추가되기 전에는 exact public timebase를 fail closed한다.

structural timebase record의 최소 필드는 다음과 같다.

- `artifacts.rnnoise_binary_sha256`, `artifacts.resampler_binary_sha256`
- `contract.argv_contract_version = raw-s16le-exact-v1`
- `source_sample_rate_hz=16000`, `rnnoise_sample_rate_hz=48000`
- demo warm-up/flush 각각 480 samples
- roundtrip impulse, first marker, last marker shift 각각 exact integer `0`
- duration cases `[1,159,160,161,479,480,481,16003]` 전체 exact
- `review.authority=CALLER_HASH_BOUND_EXPERIMENTAL`, `release_authority=none`

adapter는 이 record와 build attestation file 자체의 caller-approved SHA-256, 내부 artifact
hash 일치를 모두 검사한다. 그래도 독립 reviewer signature가 없으므로 source-time 권한은
만들지 않는다.

`PreparedRNNoiseAudio`는 두 소비 경로를 제공한다.

- `iter_chunks()` / `iter_chunks_numpy()`: VAD용 연속 waveform
- `read_mono_samples()` / `read_mono_samples_numpy()`: embedding region용 같은 waveform

prepared WAV의 digest나 임시 path를 public audio ID로 사용하면 안 된다. caller는
`PreparedRNNoiseAudio.source_sha256`의 원본 identity를 tracklet/idempotency에 유지하고,
enhancement policy/artifact hash는 별도 config provenance로 묶어야 한다. 현재
`LocalOnnxDiarizer.process(prepared.local_path)`만 감싸는 방식은 원본 audio identity를
derived WAV digest로 바꾸므로 허용하지 않는다.

### default-off 사용

```python
from sddiar.rnnoise_experimental import ExperimentalRNNoisePreprocessor

hook = ExperimentalRNNoisePreprocessor()  # enabled=False
with hook.prepare(source_path) as prepared:
    assert prepared.enhanced is False
    assert prepared.local_path == source_path
```

disabled path는 binary/root/hash 검증, runner 호출, temp 생성, write를 하지 않는다.

### 명시적 실험 사용

```python
from sddiar.rnnoise_experimental import (
    ExperimentalRNNoisePreprocessor,
    RNNoiseEnhancementPolicy,
)

hook = ExperimentalRNNoisePreprocessor(
    policy=RNNoiseEnhancementPolicy(enabled=True),
    rnnoise_binary=absolute_rnnoise_demo,
    rnnoise_binary_sha256=rnnoise_demo_sha256,
    resampler_binary=absolute_resampler,
    resampler_binary_sha256=resampler_sha256,
    rnnoise_build_attestation=absolute_build_attestation,
    rnnoise_build_attestation_sha256=build_attestation_sha256,
    timebase_proof=absolute_timebase_proof,
    timebase_proof_sha256=timebase_proof_sha256,
    artifact_root=approved_native_root,
    input_root=approved_input_root,
    work_root=private_quota_volume,
)
with hook.prepare(
    source_path,
    expected_source_sha256=source_sha256,
    expected_duration_us=source_duration_us,
) as prepared:
    assert prepared.source_time_authorized is False  # proxy A/B only
    vad_chunks = prepared.iter_chunks()
    region = prepared.read_mono_samples(start_us, end_us)
```

resampler executable은 adapter의 고정 contract를 구현해야 한다.

```text
<absolute-binary> raw-s16le
  --input <private-fixed-path>
  --output <private-fixed-path>
  --input-rate-hz <16000|48000>
  --output-rate-hz <48000|16000>
  --channels 1
  --exact-output-samples <integer>
```

PATH lookup, command string, shell, stdout/stderr capture, URL, runtime download, cache fallback은
없다. subprocess는 absolute executable을 argv list로만 실행한다. Python adapter가 network
API를 호출하지 않는 것과 native child의 egress 차단은 다른 증거다. 실제 실행은 반드시
network-disabled, filesystem-confined sandbox에서 검증한다.

위 shell/minimal-environment 설명은 adapter가 직접 만든 default
`SubprocessArgvRunner`에만 해당한다. test용 injected runner는 `NativeInvocation` argv tuple을
받지만 내부 실행 방식은 검증되지 않으므로 receipt에 `INJECTED_NOT_VERIFIED`로 남는다.

### 보안·자원 계약

- binary/input/work root는 명시적 absolute directory다.
- root escape, traversal, leaf/내부 symlink, non-regular file을 거부한다.
- source와 두 binary를 실행 전후 다시 hash한다.
- source는 no-follow descriptor에서 private snapshot으로 한 번 복사·hash한 뒤 그 snapshot만
  decode한다. path 교체 후 복원하는 race가 다른 PCM을 공급할 수 없다.
- process 전체에서 prepared context 동시성은 1이다.
- source bytes/duration, 예상 workspace peak, final output bytes, stage timeout, queue timeout을
  사전 제한한다.
- private job directory는 POSIX `0700`, files는 `0600`; 고정 basename만 허용한다.
- 예상하지 않은 file/FIFO/symlink/nested entry 또는 1-sample 차이도 fail closed한다.
- 성공·실패 뒤 derived waveform을 삭제한다. 실제 disk exhaustion 방지는 work volume quota가
  추가로 필요하다.
- receipt에는 hash prefix, sample/rate/duration, binary hash, exit code만 남기고 path, argv,
  environment, log, PCM, transcript를 남기지 않는다.
- receipt의 build lineage/timebase evidence는 각각 `CALLER_HASH_BOUND_STRUCTURAL_*`로
  표시하며 production/independent authority를 주장하지 않는다.

초기 A/B에서는 auto-gain과 RNNoise를 동시에 켜지 않는다. 켜진 enhancement는 calibration
provenance가 달라지므로 기존 `VerifiedCalibrationBinding`이 speaker-aware PASS 권한을 줄 수
없다.

## 실험 증거와 해석

redacted 수치는
[`experiments/260826_rnnoise_enhancement/evidence.json`](../experiments/260826_rnnoise_enhancement/evidence.json)에
있다. noisy 20 dB 단일 challenge proxy에서 fixed ResNet threshold 0.5가 H1에서 H2로
바뀌었다. 이는 후보를 보존할 이유이지 운영 승인 근거가 아니다.

현재 누락된 gate:

- 여러 독립 Korean recording과 subgroup holdout
- enhancement 전용 signed calibration binding
- real-binary impulse/chirp latency 및 tail parity
- hash-verified resampler source/build attestation
- 다섯 target native clean build/run
- 실제 Xeon cgroup-v1 성능·RSS 반복
- network-disabled child process 및 bounded-volume integration test

## 검증

```text
PYTHONPATH=src python3.11 -m unittest \
  tests.test_rnnoise_experimental tests.test_rnnoise_build_plan -v

PYTHONPATH=src python3.11 -m py_compile \
  src/sddiar/rnnoise_experimental.py scripts/build_rnnoise_offline.py
```
