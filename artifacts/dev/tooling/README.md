# Development-only quantization tooling

This directory contains build-time wheels and a hash manifest used to derive
development ONNX quantization candidates. These packages are not runtime
dependencies and are not part of a production-approved model pack.

The runtime candidate remains loadable with the existing offline
`onnxruntime` CPU wheel; build tooling must never be imported by the serving
path.
