# Independent Korean annotation intake

같은 녹음의 추가 threshold 조정은 중단했다. 다음 품질 개발은 파일·세션 단위로 독립된 speech mask/RTTM이 있어야 한다.

## 최초 유효 tranche

- 실제 1인 파일 2개 이상
- 실제 2인 파일 8개 이상
- MM/FF/MF 조합, 조용한 화자/원거리/소음 포함
- 16 kHz와 원본 8 kHz 각각 포함
- 짧은 turn(1초 이하), 연속 speech speaker change, overlap 포함
- 동일 화자·세션·원본 recording·augmentation은 `CALIBRATION`, `DEVELOPMENT_HOLDOUT`, `RELEASE_HOLDOUT` 사이를 넘지 않음

이 수량은 개발 착수 최소치이며 release 통계 증거는 아니다. 최종 release에는 더 큰 독립 file 분모와 subgroup confidence interval이 필요하다.

## 파일 구조

```text
dataset/
  manifest.jsonl
  audio/<audio_id>.wav
  rttm/<audio_id>.rttm
  uem/<audio_id>.uem
  words/<audio_id>.jsonl          # optional
```

`manifest.jsonl` 예시:

```json
{"audio_id":"opaque-001","audio_sha256":"<64 hex>","session_id":"session-001","split":"CALIBRATION","sample_rate_hz":16000,"speaker_count":2,"gender_pair":"MF","conditions":["near","quiet-secondary"],"audio":"audio/opaque-001.wav","rttm":"rttm/opaque-001.rttm","uem":"uem/opaque-001.uem","words":"words/opaque-001.jsonl","words_sha256":"<64 hex>","words_timebase":"microseconds"}
```

원본 이름·고객명·실명은 manifest에 넣지 않는다. `audio_id`, `session_id`, speaker label은 opaque ID만 사용한다.

`words`, `words_sha256`, `words_timebase`는 선택 artifact를 사용할 때 항상 함께 선언한다. 세 필드 중 일부만 선언하거나, 선언하지 않은 words 파일을 별도로 읽지 않는다. 모든 artifact path는 dataset root 아래의 상대경로이고 `..`, URL, symlink는 금지한다. `audio_sha256`와 `words_sha256`는 manifest가 가리키는 원본 bytes의 SHA-256이다.

words JSONL의 한 줄은 다음 계약을 따른다. `recording_id`는 반드시 해당 manifest의 `audio_id`와 같아야 하며, `timebase`는 `words_timebase=microseconds`로 고정한다.

```json
{"recording_id":"opaque-001","word_id":"word-001","start_us":320000,"end_us":500000,"text":"<annotation text>","ref_speaker_id":"REF_00","attributable":true,"overlap_flag":false,"boundary_crossing_flag":false,"micro_flag":false}
```

- `word_id`, `recording_id`, `ref_speaker_id`는 opaque ID다. `REF_OTHER`는 명시적인 비귀속 reference label로만 허용한다.
- `start_us`/`end_us`는 정수 source microseconds이고, word 전체가 audio와 UEM 범위 안에 있어야 한다.
- `text`는 문자열이어야 하며, `attributable`, `overlap_flag`, `boundary_crossing_flag`, `micro_flag`는 boolean이다. `attributable=true`이고 overlap/boundary가 아니면 `ref_speaker_id`가 필수다.
- `micro_flag`는 UEM scorer가 MICRO 후보를 표시하기 위한 보조 flag이며, 자동 화자 할당을 의미하지 않는다.
- 검증기의 aggregate JSON에는 word ID, timestamp, text, reference label, raw path를 절대 출력하지 않는다. UEM scorer가 필요할 때만 typed loader가 검증된 `WordAnnotation`을 반환한다.

## RTTM/UEM

RTTM 한 줄:

```text
SPEAKER opaque-001 1 0.320 2.150 <NA> <NA> REF_00 <NA> <NA>
```

- 시작·길이는 source audio seconds
- 실제 동시발화는 두 speaker row가 시간상 겹치도록 각각 기록
- silence는 RTTM에 기록하지 않음
- speaker change boundary를 사람이 source waveform 기준으로 확인

UEM:

```text
opaque-001 1 0.000 3171.732
```

평가 제외 구간이 있으면 UEM을 여러 줄로 나눈다.

## split 규칙

- manifest의 split은 정확히 `CALIBRATION`, `DEVELOPMENT_HOLDOUT`, `RELEASE_HOLDOUT` 중 하나다. 세 split이 모두 존재해야 한다.
- `CALIBRATION`/`DEVELOPMENT_HOLDOUT`/`RELEASE_HOLDOUT`을 파일·세션 단위로 분리한다.
- threshold/model 선택 후 `RELEASE_HOLDOUT`은 한 번만 평가한다.
- 8/16 kHz와 MM/FF/MF, near/far/noisy 결과를 각각 보고
- known 2-speaker complete merge는 파일별 hard failure
- known 1-speaker false second-speaker duration을 별도 hard metric으로 기록

이 형식이 준비되면 temporal VAD, per-cluster distance, CAM++ 보조 evidence, pyannote FP32 SCD/OSD를 동일한 release gate로 다시 평가할 수 있다.
