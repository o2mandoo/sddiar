# Development bundle third-party notices

Status: `DEVELOPMENT_ONLY`  
Target: macOS arm64 / CPython 3.11  
Prepared: 2026-08-26

This inventory records the development artifacts currently present in this directory. It is not a legal approval, a signed release notice, or permission to redistribute model weights. Exact file hashes are in `manifest.json` and `sbom.cdx.json`.

| Component | Version/artifact | Declared license or status | Release note |
|---|---|---|---|
| ONNX Runtime | 1.29.0 | MIT | Official wheel is development-only here. Production requires an approved `--no_telemetry` build. |
| NumPy | 2.4.6 | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | Wheel contains its license files. |
| protobuf | 7.36.0 | BSD-3-Clause | Wheel metadata declaration. |
| FlatBuffers | 25.12.19 | Apache-2.0 | Wheel metadata declaration. |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | Wheel contains both license texts. |
| kaldi-native-fbank | 1.22.3 | Apache-2.0 | Direct TorchAudio parity evidence is recorded separately. |
| Silero VAD repository/artifact | v6.2.1, SHA-256 `1a153a...d8788e3` | Repository declares MIT; model-weight terms require internal confirmation | Do not redistribute in a production pack until weight approval is recorded. |
| WeSpeaker code | ResNet34 runtime path | Apache-2.0 | Code license does not replace model-weight terms. |
| WeSpeaker VoxCeleb pretrained weight | SHA-256 `9fea65...056a1` | Upstream states CC-BY-4.0 for VoxCeleb pretrained models | Attribution and internal use/redistribution approval are required before release. |
| WeSpeaker VoxCeleb CAM++ challenger | SHA-256 `b50810...1efad` | Upstream states CC-BY-4.0 for VoxCeleb pretrained models | Standalone fairness gate failed; retained only as a development challenger. |
| Local ResNet34 Q1/Q1b INT8 derivatives | SHA-256 `396e15...823f`, `4632ef...e09b` | Derivatives inherit source-weight conditions | Both parity gates failed and are runtime-ineligible. |
| pyannote segmentation-3.0 converted ONNX bundle | FP32 `220ad6...e1079`, dynamic QUInt8 `d582f4...27b5d` | Bundled model license declares MIT; converted by k2-fsa/sherpa-onnx | Dynamic INT8 parity was rejected. FP32 is retained only as a development candidate for later fresh annotated SCD/OSD calibration, subject to model-access terms and internal provenance/redistribution approval; it is not default-runtime, split, or overlap approved. |

Primary upstream references:

- ONNX Runtime privacy/build policy: https://raw.githubusercontent.com/microsoft/onnxruntime/v1.29.0/docs/Privacy.md
- Silero repository license: https://github.com/snakers4/silero-vad/blob/master/LICENSE
- WeSpeaker pretrained model/license statement: https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md
- WeSpeaker code license: https://github.com/wenet-e2e/wespeaker/blob/master/LICENSE
- kaldi-native-fbank license: https://github.com/csukuangfj/kaldi-native-fbank/blob/master/LICENSE
- pyannote segmentation-3.0 model card: https://huggingface.co/pyannote/segmentation-3.0
- sherpa-onnx segmentation conversion: https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/pyannote/segmentation

For a production bundle, copy the complete applicable license texts into the target notice directory, add organization-specific approvals/attribution, and sign the final manifest/SBOM together.
