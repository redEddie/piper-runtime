# 최종 SHOW-108 MLP+LSTM24배 연결

사용자 확정 기준은 [최종 브랜치](https://github.com/redEddie/piper-sim-lab/tree/codex/show108-mlp-lstm-24u-final), snapshot `939bdb623a6bd641152388429396b8ad74e1e70a`입니다. 학습 원본은 `d377cae0afdc87a389b09071eae6bcde72465e86`입니다.

| 저장소 | 이번 최종 연결 |
|---|---|
| showonce-studio | 최종 추론 패키지 해시 검증·등록, 선택 UI, 관측·LSTM 재생·정책/QP 모의 실행 |
| piper-learning | 최종 환경·보상·관측·배치 bank·학습/평가/내보내기 소스, 공통 제어 import |
| piper-runtime | 기존0.5.0 추론·계약·QP 수식 보존, 선택 모델의 JAX 기준 벡터 회귀 검사 |

최종 통합 당시에는 세 저장소를 비공개로 유지했습니다. 이후 2026-11-10 소유자 요청으로 GitHub 공개를 진행했습니다. [현재 공개 범위](PUBLICATION.md)를 따르며, 자체 코드 라이선스는 미정입니다. 이번 이력 정리로 새 실기 명령이나 추가 학습은 수행하지 않습니다.

## 모델과 검증 범위

- Actor49 → SiLU256 → SiLU64 → LayerNorm → LSTM128 → 선형14 → 행동7.
- 누적201,326,592전이(24배), 마지막 실행의 checkpoint33554432는 추가4배입니다.
- learner SHA256: `84e064bf50d34c30b346f08c109a387b0d4b1ca58fdeb30b93db638380d44306`.
- weights.npz SHA256: `d70b4081d5bf2599ec97b2c329127fff2236fdfcf75c959294b643abd9cd345c`.
- `show108-mlp-lstm-v1`, TCP138.5mm, 정책10Hz·제어200Hz, 관절 속도1/가속도5/jerk100을 유지합니다. 배포 TCP 하한+10mm와 그리퍼force0.5도 유지합니다.
- 선택24배 시뮬레이션 접근→개방·안착351/384(91.4%)와 이전14+2배 단일 실기 성공은 서로 다른 기록입니다. 2026-11-10 사용자가 최종24배로 실기10회 시도·10회 성공을 확인했습니다. [수동 실기 결과](https://github.com/redEddie/showonce-studio/blob/main/docs/FINAL24-REAL-ROBOT-20261110.md)를 따르며 자동 집계 결과와 구분합니다.
- 자체 normalizer와 LSTM cell/hidden을 함께 사용합니다. 새 에피소드에서만 reset합니다. 14+2배의 통계와 섞지 않습니다.

## Studio에 설치

Runtime과 Studio를 설치한 Python 환경에서 실행합니다. 기존 모델은 그대로 보존합니다. JAX·학습 checkout은 필요하지 않습니다.

```bash
gh release download show108-mlp-lstm-24u-final-20261109 \
  --repo redEddie/piper-sim-lab \
  --pattern show108-mlp-lstm-24u-final-inference.tar.gz --dir downloads/final24
python -P -m showonce_studio.policy_release \
  --archive downloads/final24/show108-mlp-lstm-24u-final-inference.tar.gz
```

Studio의 versioned release 명세로 아카이브와 내부 파일 SHA를 검사합니다. 추론 코드/모델별 파일을 덮어쓰지 않고 정책 데이터만 가져옵니다. 원본 policy.json·weights·normalizer·golden 벡터를 바꾸지 않습니다.

현재 Thor 등록ID는 `1a361f3b4126da160d07`입니다. **04 정책 실행 → 최종 배포 기준 → SHOW-108 · 최종24배**를 선택합니다. 이전 정책이 URL/브라우저에 저장되어 있으면 **최종24배 모델 선택** 버튼으로 전환합니다. 명시적으로 고른 이전 모델을 자동 변경하지 않습니다. 이 화면의 자동 실기 검증 상태는 별도이며, 현장10/10 수동 확인은 위 결과 문서에 기록합니다.

1. **추론 검증**: 합성 입력80×4와 저장 시뮬레이션 관측200×20에서 행동·logits·LSTM cell/hidden을 비교합니다.
2. **정책+QP 모의 실행**: 가짜 로봇으로 제어 계약·정지 경로를 확인합니다. 물리 시뮬레이션이나 실제 작업 성공 평가가 아닙니다.
3. 실제 시험은 실시간 SAM·현재 보정·로봇 피드백을 준비한 후 shadow, Enable, 시작 자세A, 짧은run 순서입니다. 이 문서의 설치/검사는 실기 동작을 시작하지 않습니다.

## 종료 의미

학습 가중치에 A 복귀 동작이 남아 있습니다. 선정 지표에서 A를 제외한 것이 정책 출력을 바꾼 것은 아닙니다. 현재 실행기는 설정한 시간(최대20초)이나 사용자 정책 중단에서 기존 checked-stop 경로로 참조를 감속합니다.

실제 접촉·바닥 지지를 확인한④ 완료 자동 판정은 아직 연결하지 않았습니다. 상태 추정의 released_rest/그릇 내부 좌표만으로 물리적 성공을 선언하지 않습니다. 보고서의 success는 실행 정상 종료이며 task_success_evaluated는false입니다. 실제 안착 확인 후 중단하려면 정책 중단(Ctrl+C)을 사용합니다. SDK Stop/Disable은 별도 명령입니다.

## 두 Python 환경과 Runtime 잠금

Runtime 구현 SHA는 `e197cf618137f8327312ae06b303f288e5c7e0617c4a121d6cdf7022edcdf743`입니다. 수식·실행 파일을 수정하지 않아 이전 정책의 strict runtime_identity도 유효합니다. 문서/회귀 테스트만 바뀌며 버전은0.5.0을 유지합니다. 두 소비자의 lock은 같은 내용 SHA를 고정합니다.

SAM은 기존 NumPy1.26.4 환경, Studio/분석/제어는 NumPy2 환경을 유지합니다. uv에 analysis와 perception의 상호 배제를 명시하고 SAM 원본 리비전7f473cfd를 source로 고정했습니다. PyPI의 동명/동버전 메타데이터로 원본 SAM을 대체하지 않습니다. 실제 SAM 설치는 기존 scripts/setup_sam_worker.py와 SAM-WORKER.md의 전용 환경 절차를 따릅니다.

## Learning 소스 범위

최종 upstream의src/configs/scripts/tests/docs를 동기화했습니다. 공통4개 제어 모듈과 FK만 설치한 Runtime으로 import하며 수식이 원본과 같음을 비교했습니다. 내보내기/캠페인의 존재하지 않는 runtime/src 경로 삽입을 제거하고 공통 Runtime identity를 기록합니다.

Thor에서는 원본 최종 브랜치와 분리본 모두 초기 자세 bank의 compiled geometry SHA가다릅니다(expected c9bfb04f…, actual440b5ee4…). 새 소스 이식 때문에 생긴 차이가 아님을 원본에서도 재현했습니다. 은행 파일/기하 검사를 완화하거나 재작성하지 않았습니다. 이 플랫폼의 bank 기반 물리 검사·신규 학습은 검증 완료로 표시하지 않습니다. CPU 모델 추론·export·보상/관측 수식·실기 연결 계약 검사와 구분합니다.

기존 작업 폴더의 showonce_studio.py가 설치 패키지를 가리지 않도록 관리 worker와 복사 명령에 Python safe-path를 적용합니다. 기존 웹 프로세스가 켜져 있으면 코드 갱신 후 웹 프로세스만 다시 시작합니다.

## 2026-11-10 마감 결정

사용자 현장 확인 기준으로 최종24배 큐브→그릇 작업10회 시도·10회 성공을 기록하고 현행 파이프라인을 마감합니다. 성공률 자동 집계·실제 안착 자동 완료 판정은 추가 구현하지 않습니다. [실기 결과 기록](https://github.com/redEddie/showonce-studio/blob/main/docs/FINAL24-REAL-ROBOT-20261110.md)을 공통 근거로 사용합니다.
