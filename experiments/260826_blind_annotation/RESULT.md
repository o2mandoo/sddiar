# Blind micro-annotation pack 생성 결과

상태: `PACK_VERIFIED / PRIVATE / HUMAN_LABELS_PENDING`

실제 3,171.732초 음성, Clova time-only turn, baseline UNKNOWN/conflict span으로
48개의 10초 blind clip을 만들었다.

- uniform non-overlap remainder: `24개`
- reference boundary: `12개`
- system UNKNOWN/disagreement stress: `12개`
- metric union: `480초`
- clip overlap: `0`
- duplicate audio hash: `0`
- second annotator slot: `12개`
- directory/file permission: `0700/0600`
- strict manifest/clip/template hash verification: `PASS`

annotator bundle에는 category, 원본 source-time, audit slot, 화자명, 전사문이
없다. evaluator manifest와 annotation audio/labels는 `.private` 아래에만 있으며
Git에서 제외된다.

현재 label template은 비어 있으므로 이것을 정답셋이라고 부르지 않는다.
사람이 `HUMAN_SPK_0/1`, `SILENCE`, `OVERLAP`, `UNCLEAR`, change boundary를
입력한 뒤에만 micro-DER/SCD/OSD 평가에 사용할 수 있다.
