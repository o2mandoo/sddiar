# whisper.cpp turbo Q5 + VAD 60초 A/B

상태: `VAD_REJECTED / OPENVINO_ARTIFACTS_MISSING / RELEASE_AUTHORITY_NONE`

정확히 60초인 PCM16 mono clip을 만들고, 동일한 turbo Q5·beam5·DTW·1 thread
조건에서 baseline과 Silero v6.2.0 VAD를 각각 두 번 실행했다. 측정 순서는
`VAD → baseline → baseline → VAD`로 고정해 실행 위치 평균을 같게 했다.

| 항목 | baseline | VAD |
|---|---:|---:|
| median wall | 103.26초 | 71.07초 |
| median RTF | 1.721 | 1.1845 |
| peak RSS | 870.6MiB | 850.3MiB |
| Clova proxy CER | 15.42% | 23.83% |
| turn-aligned CER | 20.09% | 87.38% |
| inferred timestamp ratio | 1.54% | 5.63% |

VAD는 24개 speech segment, 총 41.87초를 선택해 입력 sample을 26.4% 줄였고
wall도 31.17% 단축했다. 하지만 RTF 1.0을 넘었고 CER가 8.41%p,
turn-aligned CER가 67.29%p 악화됐다. 두 반복의 JSON/timeline hash는 arm별로
동일했으므로 우연한 한 번의 실패가 아니다. 따라서 VAD는 기본값으로 올리지
않고 `REJECT_VAD_QUALITY_AND_RTF_KEEP_DEFAULT_OFF`로 동결한다.

OpenVINO 두 arm은 실행하지 않았다. 현재 binary가 `WHISPER_OPENVINO=OFF`이고,
검증된 Linux x86_64 binary·encoder XML/BIN·OpenVINO runtime pack이 없다.
또한 upstream external encoder는 `-t 1`만으로 inference thread 1을 보장하지
않으므로 실제 Xeon의 cgroup-v1 `100000/100000` 안에서만 속도를 승인한다.

이 결과는 한 파일의 Clova proxy 개발 A/B다. 사람 정답, word-boundary 정답,
실제 Xeon 결과 또는 production 승인으로 해석하지 않는다. 음성·전사·가설·로컬
경로는 Git에 포함하지 않았다.
