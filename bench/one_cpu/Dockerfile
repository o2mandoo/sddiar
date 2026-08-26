FROM python@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

ENV PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OMP_THREAD_LIMIT=1 \
    OMP_DYNAMIC=FALSE \
    OMP_WAIT_POLICY=PASSIVE \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MKL_DYNAMIC=FALSE \
    BLIS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    KMP_BLOCKTIME=0 \
    MALLOC_ARENA_MAX=2 \
    ORT_DISABLE_TELEMETRY=1

COPY artifacts/dev-linux-aarch64/wheels /opt/sddiar-wheelhouse
COPY artifacts/dev-linux-aarch64/requirements.lock /opt/sddiar-requirements.lock
COPY bench/one_cpu/run_repeated_worker.py /opt/sddiar/bench/one_cpu/run_repeated_worker.py

RUN python -m pip install \
      --no-index \
      --find-links /opt/sddiar-wheelhouse \
      --require-hashes \
      -r /opt/sddiar-requirements.lock \
    && python -c "import kaldi_native_fbank, numpy, onnxruntime, sddiar; assert 'CPUExecutionProvider' in onnxruntime.get_available_providers()" \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin sddiar

USER 10001:10001
WORKDIR /workspace

ENTRYPOINT ["python"]
