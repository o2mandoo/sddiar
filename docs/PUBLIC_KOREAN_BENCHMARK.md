# 한국어 공개 데이터 평가 체계 v1

상태: `IMPLEMENTED / SYNTHETIC_CONTRACT_VERIFIED / PUBLIC_AUDIO_NOT_IMPORTED`

이 체계는 한국어 자료만 사용하며 AI Hub 자료를 전제로 하지 않는다. 원음,
전사, RTTM/UEM, 이용 약정은 `.private/korean-evaluation/` 아래에만 두고 Git에는
변환 코드, 합성 fixture, 집계 지표와 SHA-256만 기록한다.

## 데이터 역할

| 역할 | 용도 | metric gate 권한 |
|---|---|---|
| `GOLD` | 사람이 만든 연속 speech/speaker/overlap 시간 정답 | 검증된 release holdout의 수치 판정만 가능 |
| `SILVER` | 자동 라벨, 발화 클립, 시간축·overlap가 불완전한 자료 | 불가, 개발 참고만 |
| `CHALLENGE` | 결정론적 합성 1·2인/잡음/overlap 회귀 | 실패는 veto, 성공은 Gold나 출시 승인 대체 불가 |

현재 후보와 공식 진입점은 다음처럼 고정한다.

- [국립국어원 `지역어 말뭉치 2021`](https://kli.korean.go.kr/corpus/main/requestMain.do?lang=ko):
  연속 WAV+JSON 290쌍의 `GOLD` 후보.
  시간 단위, 전체 UEM, overlap 완전성, 이용 권한을 감사하기 전에는
  `PROVISIONAL / REVIEW_REQUIRED`다.
- 국립국어원 `일상 대화 음성 말뭉치 2020~2025`: 발화 클립과 불완전한
  overlap 때문에 `SILVER`다.
- [Zeroth-Korean](https://openslr.org/40/): CC BY 4.0 단일화자 발화를 결정론적으로 혼합하는
  `CHALLENGE`다. 실제 자연대화 Gold를 대체하지 않는다.
- [TalkBank Ko Corpus](https://talkbank.org/childes/access/EastAsian/Korean/Ko.html) 등
  비상업 자료: 별도 `RESEARCH_ONLY` lane이다.
- [ClovaCall](https://github.com/clovaai/ClovaCall)/CoreaSpeech: 연속 human
  diarization 정답이 아니므로 제외한다.

## 구현 경계

```text
NIKL JSON + continuous WAV
  -> sddiar.nikl_adapter (speaker/time schema, explicit unit, redaction)
  -> normalized WAV/RTTM/UEM/manifest.jsonl (private)
  -> validate_annotation_dataset (hash/path/WAV/privacy/split gate)
  -> corpus.lock.json (GOLD/SILVER/CHALLENGE, license, audit, split lock)
  -> prediction manifest + RTTM
  -> evaluate_korean_benchmark
  -> DER/JER/SCD/OSD/coverage/accuracy/subgroup/bootstrap + redacted receipt
```

기존 scorer는 한 파일당 1~2화자만 지원한다. 참조 overlap은 `OVERLAP` 한 줄이
아니라 서로 다른 두 `REF_*` RTTM 행의 시간 중첩으로 표현한다.

## NIKL 지역어 어댑터

`parse_nikl_reference`는 원본 텍스트와 화자 ID를 출력하지 않고 다음만 반환한다.

- canonical payload SHA-256에서 파생한 opaque recording ID
- 세션 로컬 `REF_00`, `REF_01`
- 정수 source microsecond `RTTMRecord`
- 서로 다른 화자의 실제 시간 중첩
- annotation 최소 시작~최대 종료의 provisional UEM
- time unit, overlap duration, count/hash만 있는 public evidence

지역어 설명서의 `start/end` 단위 표기가 실제 예시와 충돌하므로 호출자는
`milliseconds` 또는 `seconds`를 명시해야 한다. 값이 WAV duration을 1µs라도
벗어나면 clamp하지 않고 실패한다. provisional UEM은 전체 발화 누락·비식별
구간을 사람이 감사한 뒤에만 full-audio 또는 audited-exclusion UEM으로 승격한다.

## 정규화 reference manifest

기존 annotation manifest에 다음 binding을 추가했다. 일반 intake에서는 기존
manifest 호환을 위해 일부가 선택적이지만, 한국어 benchmark runner에서는
`speaker_group_ids`, reference/UEM 상태와 변환 evidence hash까지 필수다.

```json
{
  "audio_id": "opaque-001",
  "audio_sha256": "<sha256>",
  "session_id": "session-001",
  "speaker_group_ids": ["person-hmac-001", "person-hmac-002"],
  "split": "RELEASE_HOLDOUT",
  "sample_rate_hz": 16000,
  "speaker_count": 2,
  "gender_pair": "UNKNOWN",
  "conditions": ["regional-interview", "overlap"],
  "audio": "audio/opaque-001.wav",
  "rttm": "rttm/opaque-001.rttm",
  "rttm_sha256": "<sha256>",
  "uem": "uem/opaque-001.uem",
  "uem_sha256": "<sha256>"
}
```

동일 session/source/augmentation/speaker HMAC가 둘 이상의 split에 나타나면
전체 평가를 거부한다. 원본 화자 ID와 HMAC key는 공개 artifact에 넣지 않는다.
`speaker_group_ids`의 개수는 `speaker_count`와 정확히 같아야 한다.

## corpus lock

`sddiar-korean-corpus-lock/v1`은 다음을 hash로 묶는다.

- corpus/version/language=`ko`
- `GOLD`, `SILVER`, `CHALLENGE`
- human/publisher/machine/synthetic annotation origin
- 이용 권한 상태와 license text hash
- continuous timeline, independent speaker split, audit 상태
- diarization/overlap/SCD/word reference capability
- source archive, annotation manifest, split lock, audit SHA-256

`SILVER`, `CHALLENGE`, 연구 전용 license, 미감사 timeline/capability는 점수를
계산해도 Gold metric gate 자격을 만들지 못한다. v1은 호출자가 verifier나
임계값을 주입할 수 있는 일반 라이브러리이므로 출시 PASS를 발급하지 않는다.
모든 지표가 기준을 만족해도 최상위 상태는
`METRIC_GATES_PASS_REVIEW_REQUIRED`, `release_authority=none`이며 항상
`EXTERNAL_RELEASE_AUTHORITY_REQUIRED`를 남긴다. 고정된 외부 trust root가
corpus/split/policy/challenge registry/결과/코드 버전을 함께 승인하는 계층은
이 라이브러리 밖의 후속 작업이다.

Gold metric gate가 통과하더라도 사전 등록된 합성 overlap/noise/저음량
challenge가 누락되거나 하나라도 실패하면 suite metric gate가 실패하거나
review 상태로 남는다. 모두 통과해도 `SUITE_METRICS_PASS_REVIEW_REQUIRED`이며,
Challenge 성공만으로 Gold 실패나 누락을 대신할 수 없다.

## prediction manifest와 실행

```json
{"audio_id":"opaque-001","rttm":"pred/opaque-001.rttm","rttm_sha256":"<sha256>","quality_status":"REVIEW_REQUIRED"}
```

```sh
PYTHONPATH=src python3.11 scripts/evaluate_korean_benchmark.py \
  .private/korean-evaluation/manifest.jsonl \
  .private/korean-evaluation/predictions.jsonl \
  --corpus-lock .private/korean-evaluation/corpus.lock.json \
  --split DEVELOPMENT_HOLDOUT \
  --dataset-root .private/korean-evaluation \
  --prediction-root .private/korean-evaluation
```

각 실행은 한 split만 평가한다. prediction ID는 전체 reference ID와 정확히
일치해야 하며, reference/prediction RTTM과 UEM hash, 상대경로, symlink,
크기 제한을 검증한다. 선택된 WAV는 PCM을 디코딩하지 않고 동일 file descriptor의
header/duration/hash snapshot으로 RTTM/UEM 범위와 다시 결합한다. 공개 JSON에는
개별 recording ID, path, transcript, speaker label이 없다.

## 지표와 고정 gate

- DER: duration micro와 recording macro
- DER non-overlap: duration micro와 recording macro
- JER recording macro
- attributed non-overlap speech coverage: duration micro와 recording macro
- assigned accuracy: duration micro와 recording macro
- miss/false alarm/confusion duration
- complete merge, false-H2, speaker-count accuracy
- capability가 있는 경우만 SCD/OSD
- gender/rate/condition/split subgroup와 recording bootstrap

기본 metric gate는 250ms collar DER `<=15%`, JER `<=25%`, non-overlap coverage `>=85%`,
assigned accuracy `>=95%`, complete merge `0`, false-H2 secondary duration
`<=1%`, SCD F1 `>=0.75`, OSD precision/recall `>=0.75/0.60`, 주요 subgroup
DER 격차 `<=5%p`다. 최소 20개 recording을 요구하지만 실제 출시 holdout은
더 큰 독립 세트를 사용해야 한다. 경계 관용이 없는 0ms DER도 항상 별도
diagnostic으로 함께 출력한다.

## 아직 외부에서 필요한 것

1. 국립국어원 지역어 말뭉치 2021 이용 신청·약정·반입
2. 공식 290쌍 inventory와 SHA-256 고정
3. 최소 WAV 10개 시간 단위 smoke, 전체 30개 층화 수동 감사
4. full-audio/UEM와 overlap completeness 확정
5. speaker HMAC 연결요소 기반 split lock 생성
6. release holdout 결과를 본 뒤 threshold를 변경하지 않는 운영 분리
7. 실제 출시 판정이 필요하면 고정된 외부 신뢰 루트와 독립 승인 절차 구축

실제 데이터가 반입되기 전 현재 증거는 합성 fixture를 이용한 계약·scorer
정합성에 한정된다.
