# ONNX Runtime 1.29.0 target-native build plan

The development wheel packs contain the official `onnxruntime==1.29.0`
PyPI artifacts where PyPI publishes a matching tag. Those wheels are not
evidence of a telemetry-free production binary. Production approval stays
false until a target-native source build and independent audit are complete.

## Pinned source and closed-network boundary

- Repository: `https://github.com/microsoft/onnxruntime.git`
- Ref: `v1.29.0`
- Resolved tag commit: `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`
- Checkout: `git clone --branch v1.29.0 --recurse-submodules --shallow-submodules ...`
- Record `git submodule status --recursive`; the build must use those checked-out
  SHAs and must not resolve floating refs after entering the closed network.
- Runtime: CPython 3.11, CPUExecutionProvider only, with the package's 1-CPU
  thread policy applied separately.

The reproducibility pins used by the v1.29.0 upstream CI are CMake 3.31.8
(SHA-256
`99cc9c63ae49f21253efb5921de2ba84ce136018abf08632c92c060ba91d552e0f6acc214e9ba8123dee0cf6d1cf089ca389e321879fd9d719a60d975bcffcc8`)
and vcpkg 2025.08.27 (SHA-256
`9a4b32849792e13bee1d24726f073b3881acae4165206ddf1a6378e44a4ddd05b3ee93f55ff46d8e8873b3cbcd06606212989e248f0bd615a5bf365070074079`).
Use Visual Studio 17 2022 x64 on Windows and the upstream GCC 14 native
container family on Linux (`cpu_x86_64_almalinux8_gcc14:20251017.1` or
`cpu_aarch64_almalinux8_gcc14:20251017.1`); record the resolved image digest.
On macOS, pin the Xcode image and record the exact `xcodebuild -version`, SDK,
and architecture. The helper carries these as required attestation fields;
missing pins are a failed gate, not an implicit default.

Fetch source and submodules while network access is permitted. Create the
source archive/checksums, then disconnect network access before running the
build, wheel install, tests, or attestation. The helper does not fetch or build
anything by itself.

## Telemetry switch and commands

ONNX Runtime 1.29.0's official source build exposes the upstream
`--no_telemetry` switch and the CMake option
`onnxruntime_USE_TELEMETRY`. The commands below pass both: the upstream switch
expresses build intent, while the explicit CMake definition protects against a
wrapper/default silently turning telemetry on. Each command also supplies the
required, target-specific `--build_dir`. `scripts/build_ort_no_telemetry.py
plan` emits the same commands as JSON.

Run each command natively on the target (do not cross-compile and label it as
the target), with an exact CMake/Ninja/compiler version recorded in the
attestation:

```text
# Windows x86_64, Visual Studio 17 2022 x64 developer shell
py -3.11 tools/ci_build/build.py --build_dir build/ort-no-telemetry/windows-x86_64 --config Release --build_wheel --skip_tests --skip_submodule_sync --skip_pip_install --use_vcpkg --no_telemetry --parallel 1 --enable_pybind --cmake_generator "Visual Studio 17 2022" --cmake_extra_defines onnxruntime_USE_TELEMETRY=OFF

# Linux x86_64 (native GCC/Clang)
python3.11 tools/ci_build/build.py --build_dir build/ort-no-telemetry/linux-x86_64 --config Release --build_wheel --skip_tests --skip_submodule_sync --skip_pip_install --use_vcpkg --no_telemetry --parallel 1 --enable_pybind --cmake_generator Ninja --cmake_extra_defines onnxruntime_USE_TELEMETRY=OFF

# Linux aarch64 (native aarch64 host)
python3.11 tools/ci_build/build.py --build_dir build/ort-no-telemetry/linux-aarch64 --config Release --build_wheel --skip_tests --skip_submodule_sync --skip_pip_install --use_vcpkg --no_telemetry --parallel 1 --enable_pybind --cmake_generator Ninja --cmake_extra_defines onnxruntime_USE_TELEMETRY=OFF

# macOS Intel
python3.11 tools/ci_build/build.py --build_dir build/ort-no-telemetry/macos-x86_64 --config Release --build_wheel --skip_tests --skip_submodule_sync --skip_pip_install --use_vcpkg --no_telemetry --parallel 1 --enable_pybind --cmake_generator Xcode --cmake_extra_defines onnxruntime_USE_TELEMETRY=OFF CMAKE_OSX_ARCHITECTURES=x86_64

# macOS Apple Silicon
python3.11 tools/ci_build/build.py --build_dir build/ort-no-telemetry/macos-arm64 --config Release --build_wheel --skip_tests --skip_submodule_sync --skip_pip_install --use_vcpkg --no_telemetry --parallel 1 --enable_pybind --cmake_generator Xcode --cmake_extra_defines onnxruntime_USE_TELEMETRY=OFF CMAKE_OSX_ARCHITECTURES=arm64
```

The helper's `--parallel 1` is for reproducible build pressure on the 1-CPU
development target; it does not make a host's cgroup quota claim for a native
compiler. Keep the runtime's ORT intra/inter-op threads at one as required by
the Xeon runbook.

## Attestation and checks

Create an attestation only after the wheel exists and the checkout is at the
pinned commit:

```text
python3 scripts/build_ort_no_telemetry.py attest \
  --target linux-x86_64 \
  --artifact build/onnxruntime-1.29.0-cp311-cp311-*.whl \
  --source-root /src/onnxruntime \
  --output ort-attestation.json
python3 scripts/build_ort_no_telemetry.py verify-attestation ort-attestation.json
```

The JSON schema requires the source URL/ref/commit, recursive submodule
records, target architecture, artifact SHA-256, recorded tool versions, the
exact `onnxruntime_USE_TELEMETRY=OFF` definition, and `production_approved:
false`. It also requires `telemetry.status: not_verified`; this project does
not claim telemetry-free status based only on a build flag. A separate review
must inspect the binary, source configuration, linked dependency closure, and
closed-network logs before changing that status or approving production.

The verifier performs no network access. It rejects a floating/wrong source
commit, hash mismatch, missing artifact, telemetry flag other than `OFF`, or a
telemetry status other than `not_verified`.

Authoritative upstream references: [ONNX Runtime privacy and telemetry
policy](https://onnxruntime.ai/docs/reference/privacy.html) and [inference
build instructions](https://onnxruntime.ai/docs/build/inferencing.html).
