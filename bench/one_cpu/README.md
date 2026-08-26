# Local 1-CPU quota proxy

This image is a Linux arm64 development proxy for the target Kubernetes quota shape. It verifies cgroup quota handling, thread discipline, bounded memory, offline execution, and long-file completion. It does **not** claim Intel Xeon 6230R ISA or clock parity.

Build from the repository root:

```sh
docker build --network=none -f bench/one_cpu/Dockerfile -t sddiar-onecpu:0.4.0 .
```

Runtime invariants:

- `--cpus=1` while the container may still see many logical CPUs
- `--network=none`
- ORT intra-op/inter-op threads set to 1 by the benchmark caller
- BLAS/OpenMP thread environment fixed to 1
- read-only root filesystem and read-only source/model/input mounts
- result directory is the only writable bind mount

The actual Intel x86_64 cgroup-v1 environment remains a separate release gate.

## Persistent-worker gate

The image contains the benchmark harness separately from the installed
`sddiar` wheel, so mounting the source tree cannot accidentally replace the
package under test. Run at least two passes to cover more than 60 minutes of
processed audio and to establish a warm-memory baseline:

```sh
docker run --rm --network=none --cpus=1 --memory=256m --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=32m \
  --mount type=bind,src=/local/input.wav,dst=/input/input.wav,readonly \
  --mount type=bind,src=/local/models,dst=/models,readonly \
  --mount type=bind,src=/local/results,dst=/results \
  sddiar-onecpu:0.4.0 \
  /opt/sddiar/bench/one_cpu/run_repeated_worker.py \
  /input/input.wav \
  --silero-model /models/silero_vad.onnx \
  --silero-sha256 EXPECTED_SHA256 \
  --wespeaker-model /models/voxceleb_resnet34.onnx \
  --wespeaker-sha256 EXPECTED_SHA256 \
  --assignment-distance-limit 0.5 \
  --repetitions 2 \
  --min-total-audio-minutes 60 \
  --evidence-mode local_proxy \
  --require-cgroup-version v2 \
  --output /results/repeated_worker.json
```

The harness fails closed on timeline digest drift, model/runtime fallback,
quota mismatch, RTF or throttle ceilings, process-tree/cgroup memory above
256 MiB, and warm-memory growth above the declared limit. Target evidence uses
`--evidence-mode target --require-cgroup-version v1` and additionally requires
cgroup CPU, process-tree RSS/PSS, and I/O counters.
