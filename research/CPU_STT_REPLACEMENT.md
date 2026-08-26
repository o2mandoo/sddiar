# 1 CPU STT 대체 검토

기준일: 2026-08-26  
상태: `FUNCTIONAL_INTEGRATION_READY / CPU_ONLY_DROP_IN_REPLACEMENT_REJECTED`

## 결론

기능 대체는 가능하다. `ProductionOrchestrator`가 로컬 STT의 단어 시간축을
받아 화자분리 결과와 결합하고, 화자분리 실패 시에도 화자 중립 대본을
보존한다.

다만 현재 GPU `large-v3` 수준의 STT 정확도와 짧은 응답시간을 Xeon 1 CPU에서
동시에 대체하는 후보는 찾지 못했다. 지금 가장 안전한 전환은 **기존 STT는
유지하고 sddiar의 화자분리·단어 귀속을 추가**하는 것이다. STT 엔진 자체도
CPU로 바꿔야 한다면 정확도 또는 처리시간 중 하나를 양보해야 한다.

## 5분 동일 구간 개발 실측

조건:

- Apple M5 Max, macOS arm64
- inference thread 1, GPU/Metal/BLAS/Accelerate/OpenMP off
- 16 kHz mono, 300초
- Clova Note 결과를 정답 proxy로 가정
- primary metric: NFC, 문장부호·공백 제거 CER
- 이 결과는 Xeon cgroup 증거가 아님

| 후보 | 5분 wall | RTF | peak RSS | CER | 판정 |
|---|---:|---:|---:|---:|---|
| whisper.cpp base Q5 greedy | 34.15초 | 0.114 | 723MiB | 48.25% | 정확도 탈락 |
| whisper.cpp base Q5 beam5 | 43.76초 | 0.146 | 772MiB | 39.53% | 정확도 탈락 |
| whisper.cpp small Q5 beam5 | 138.91초 | 0.463 | 1,036MiB | 26.50% | 정확도 탈락 |
| whisper.cpp turbo Q5 beam5 | 629.91초 | 2.100 | 1,427MiB | 15.90% | 품질 기준선, 속도 탈락 |
| Korean Zipformer INT8, 25초 chunk | 3.18초 | 0.011 | 835MiB | 60.11% | 정확도 탈락 |
| SenseVoice INT8, 25초 chunk | 8.13초 | 0.027 | 1,055MiB | 24.35% | 빠른 fallback 후보, 정확도 탈락 |

5분에서 10분으로 단순 선형 환산하면 turbo Q5는 약 21분이다. Xeon 6230R
1 CPU에서는 M5보다 빨라질 근거가 없으므로 운영 기본값으로 두지 않는다.
반대로 SenseVoice는 속도는 충분하지만 Clova 대비 CER 비열화가 크다.

Korean Zipformer에 5분을 통째로 넣으면 decode는 13.60초였지만 offline
activation 때문에 peak RSS가 약 21.5GiB까지 증가했다. 25초 chunk로 바꾸면
메모리는 약 835MiB로 줄었지만 CER가 60.11%라 탈락했다. 짧은 공식 샘플의
RTF를 장시간 파일로 외삽하면 안 된다는 반증이다.

상세 수치와 artifact hash는
[`experiments/260826_stt_cpu_proxy/evidence.json`](../experiments/260826_stt_cpu_proxy/evidence.json)에
있다. 음성·대본·로컬 경로는 포함하지 않는다.

## 엔진 판단

### whisper.cpp

개발 기준은 `v1.9.3/b4938`, commit
`371b5a7561823ab2bb32142d2751e35e7534727b`이다. CPU-only와 정수 양자화,
thread/process 수 제한, full JSON token timestamp를 제공한다. 최신 tag와
실행 옵션은 [공식 README](https://github.com/ggml-org/whisper.cpp/blob/371b5a7561823ab2bb32142d2751e35e7534727b/README.md)와
[b4938 release](https://github.com/ggml-org/whisper.cpp/releases/tag/b4938)를 기준으로
고정했다.

OpenAI는 turbo를 large-v3의 최적화 계열로 설명한다. 그러나 그 설명은
한국어 Q5 양자화 parity 증거가 아니다. 실제 동일 구간에서 CER 15.90%와
RTF 2.10이므로 이 환경에서는 품질 비교 기준선으로만 남긴다.
[OpenAI Whisper](https://github.com/openai/whisper/tree/31243bad24cc746f07d4c8bfdd2d974872cb1803)

### sherpa-onnx Korean Zipformer

`sherpa-onnx 1.13.6`과 Korean Zipformer INT8을 사용했다. 공식 문서는 CPU
thread 수와 token timestamp를 제공하지만 `words`와 word duration은 비어
있다. 다음 token 시작점을 word 끝으로 간주하는 방식은 별도 time-alignment
검증 전에는 speaker attribution 권한을 가질 수 없다.
[공식 Korean model 문서](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#sherpa-onnx-zipformer-korean-2024-06-24)

자체 KsponSpeech model card 수치는 이 ONNX INT8 artifact와 현재 회의 음성의
검증값이 아니다. 실제 동일 구간 CER 60.11%를 우선한다.

### SenseVoice INT8

공식 sherpa-onnx 변환본은 한국어를 포함한 5개 언어와 1-thread CPU 실행을
지원한다. 현재 가장 빠르면서 base/small Whisper보다 나은 CER를 보였지만,
24.35%는 대체 기준을 넘지 못한다. native word duration 부재와 model
redistribution/license 승인도 남아 있다.
[공식 SenseVoice model 문서](https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html)

## 운영 권고

1. 기존 STT가 word timestamp를 반환한다면 그 엔진을 그대로 둔다.
2. sddiar는 별도 1-CPU worker로 화자분리를 수행한다.
3. `ProductionOrchestrator`가 whole-word 기준으로 화자를 귀속한다.
4. 화자분리 실패 시 STT 대본은 보존하고 모든 화자를 `UNKNOWN`으로 내린다.
5. 독립 signed calibration 전에는 speaker-aware summary를 열지 않는다.
6. STT와 diarization 모델은 동시에 resident로 두지 않는 순차 프로세스를
   기본으로 한다.

CPU-only STT를 반드시 써야 할 때는 두 profile을 분리한다.

- `QUALITY`: turbo Q5, 느린 background batch, 2GiB 이상 별도 memory gate
- `FAST_DRAFT`: SenseVoice INT8, 사람 검토 또는 후속 교정 필수

어느 profile도 현재 production 승인 상태가 아니다.

## 다음 승인 조건

- 동일 입력의 현재 GPU large-v3 실제 출력 확보
- GPU large-v3 대비 CER `+1%p`, SA-WER `+2%p` 이내 비열화
- 실제 Xeon cgroup-v1 `100000/100000`, thread 1, full-file 측정
- 5분/10분 prefix 이후 전체 3,171.732초 순차 gate
- word start/end annotation 또는 검증된 timestamp alignment
- 독립 Korean recording/session holdout
- engine/model/license/SBOM/no-network build 승인

## 0.4.0 이후 시스템 최적화 경로

단독 모델 재시험보다 다음 순서가 우선이다.

1. Linux x86_64 Xeon에서 CTranslate2 turbo INT8을 1 thread로 직접 비교한다.
   AVX-512/INT8 지원은 후보 선정 근거일 뿐 속도 승인 근거가 아니다.
2. 동일 turbo Q5의 `baseline / VAD / OpenVINO encoder / VAD+OpenVINO`
   네 arm을 60초에서 비교하고, 승자만 300초로 확장한다.
3. fast draft와 full refiner의 구간별 additive edit-error count로 10/20/30/40%
   duration-budget oracle을 계산한다.
4. 20% oracle이 full refiner 대비 허용 비열화를 만족할 때만 실제 router를
   구현한다. oracle도 실패하면 cascade 개발을 중단한다.

`sddiar.stt_cascade_experimental`은 3번을 위한 default-off 하네스다. 임의의
구간 CER/WER rate나 문자열 길이 차는 합산하지 않는다. 동일 종류의 character
또는 word edit count와 전체 reference unit count만 허용하며, 결과는 항상
`REVIEW_REQUIRED / release_authority=none`이다.

VAD/OpenVINO와 CTranslate2 artifact는 전략별 다중 파일 pack으로 해시를 묶는다.
pack은 명시된 local artifact root 밖을 읽지 않고 파일 수·tree entry·총 byte를
제한한다. 이 계약은 runtime 승인 기능이 아니라 폐쇄망 A/B 재현성 경계다.

현재 저장소에는 OpenVINO/CT2 실제 모델 실행 결과가 없다. 필요한 native
runtime/model pack과 실제 Xeon cgroup-v1 환경이 들어오기 전에는 5분·10분
처리시간 또는 기존 GPU large-v3 비열화를 승인하지 않는다.

CT2 executable prototype은 검증한 wheel과 child가 실제 import한 설치 코드가
동일하다는 증거, immutable audio/model snapshot, descendant process termination을
동시에 만족하지 못해 이번 library source에는 포함하지 않았다. 버전 문자열이나
wheel hash 문자열만 맞는 backend를 공개하지 않는다. 다음 구현은 verified
isolated interpreter/site-packages/worker tree와 model snapshot을 하나의 실행
artifact로 반입한 뒤 시작한다.
