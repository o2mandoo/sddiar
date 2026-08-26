# Xeon Gold 6230R / 1.00 CPU runbook

이 문서는 `genos-cs-356`과 같은 Linux x86_64 cgroup-v1 환경에 개발 bundle을 반입해 측정하는 절차다. 공식 ORT wheel을 쓰는 현재 bundle은 development verification용이며 production no-telemetry 승인을 대신하지 않는다.

## 1. 폐쇄망 설치

반입 대상:

- `artifacts/dev-linux-x86_64/wheels/`
- `artifacts/dev-linux-x86_64/requirements.lock`
- `artifacts/dev/models/silero_vad.onnx`
- `artifacts/dev/models/voxceleb_resnet34.onnx`
- `scripts/verify_xeon_onecpu_target.py`
- 필요한 경우 `bench/one_cpu/run_benchmark.py`, `src/`, `scripts/run_onnx_diarization_experiment.py`

```sh
python3.11 -m venv .venv-sddiar
.venv-sddiar/bin/python -m pip install \
  --no-index \
  --find-links artifacts/dev-linux-x86_64/wheels \
  --require-hashes \
  -r artifacts/dev-linux-x86_64/requirements.lock
```

## 2. 1-thread 환경 고정

아래 값은 Python/NumPy/ORT import 전에 설정한다.

```sh
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=PASSIVE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_BLOCKTIME=0
export MALLOC_ARENA_MAX=2
export ORT_DISABLE_TELEMETRY=1
```

## 3. strict preflight

```sh
.venv-sddiar/bin/python scripts/verify_xeon_onecpu_target.py \
  --selected-provider CPUExecutionProvider \
  --model artifacts/dev/models/silero_vad.onnx \
  --model-sha256 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3 \
  --model artifacts/dev/models/voxceleb_resnet34.onnx \
  --model-sha256 9fea6516d7ad6bf0a76c7689f5a49b65d330fad6dde96c91bb4435ffbfe056a1 \
  --wheel artifacts/dev-linux-x86_64/wheels/sddiar-0.3.0-py3-none-any.whl \
  --wheel-sha256 695e332892fdc16225a924cabde56e58d9de2906bfa95eed2db29f5207fdd631 \
  --lock artifacts/dev-linux-x86_64/requirements.lock \
  --lock-sha256 c473b7f1a55590c0cffc7fa675556a36ea3676a9f8b1a9eb384eafd328a2e401
```

다음이 모두 일치해야 한다.

- Linux `x86_64`, Intel Xeon Gold 6230R, AVX2/AVX-512 VNNI
- cgroup v1 quota/period `100000/100000`, effective CPU `1.0`
- `cpu.shares=1024`(읽을 수 있는 경우)
- CPython 3.11, ORT 1.29.0, selected provider CPU only
- 모든 artifact hash와 thread environment

## 4. 단계별 benchmark

먼저 60초, 300초, 900초를 실행하고 전체 파일로 확대한다. `run_benchmark.py`는 1.00 CPU가 아니면 중단한다.

```sh
PYTHONPATH=src:scripts .venv-sddiar/bin/python bench/one_cpu/run_benchmark.py \
  INPUT_16K_MONO_PCM.wav \
  --silero-model artifacts/dev/models/silero_vad.onnx \
  --silero-sha256 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3 \
  --wespeaker-model artifacts/dev/models/voxceleb_resnet34.onnx \
  --wespeaker-sha256 9fea6516d7ad6bf0a76c7689f5a49b65d330fad6dde96c91bb4435ffbfe056a1 \
  --threads 1 \
  --output xeon-benchmark.json
```

운영 performance gate는 full quality stack RTF p95 `<=0.25`, max `<=0.35`, process-tree RSS `<=256MiB`, throttled wall ratio `<=1%`다. 5회 이하는 p95라고 부르지 않고 median/max를 보고한다.

## 5. 품질 주의

현재 `0.5` assignment limit와 temporal VAD challenger는 한 녹음 proxy 결과다. 실제 target benchmark에서는 성능만 측정하고 threshold를 조정하지 않는다. 품질 승격은 별도의 Korean speech mask/RTTM/overlap calibration과 holdout에서 수행한다.
