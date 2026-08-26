# SCD/OSD 전체 파일 shadow 실행

상태: `MODEL_SHADOW_READY / SPAN_PARITY / RELEASE_AUTHORITY_NONE`

pyannote segmentation-3.0 FP32 ONNX를 3,171.732초 전체 파일에 1 thread로
실행했다. 이 단계는 candidate를 관찰만 하며 기존 화자 span을 변경하지 않는다.

- wall: `56.468초`
- model inference: `54.523초`
- peak RSS: `200.59MiB` (host process, cgroup 증거 아님)
- 10초 window / 1초 shift: `3,163개`
- source-time frame: `187,955개`
- diagnostic SCD local maximum: `11개`
- overlap probability `>=0.5`: `1,263 frame`
- overlap probability `>=0.75`: `322 frame`
- 화자 span 변경: `0`

11개 SCD 후보는 아직 WeSpeaker 좌우 probe와 함께 dual gate를 통과하지 않았다.
OSD frame도 calibrated interval이 아니므로 `OVERLAP` veto로 사용하지 않았다.
현재 결과는 staged shadow 실행 가능성과 비용만 증명한다.
