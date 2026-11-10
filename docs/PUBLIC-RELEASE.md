# 공개 준비 상태 — 2026-11-05

2026-11-10 소유자 요청으로 GitHub 공개 전환을 진행했습니다. 현재 공개 범위는 [PUBLICATION.md](PUBLICATION.md)를 따릅니다. 아래는 2026-11-05 준비 당시의 검사 범위와 미확정 항목이며, 새 검증 결과를 뜻하지 않습니다.

## 준비한 범위

- 새 Python 환경의 CPU wheel 설치 절차와 합성 예제/검사 명령.
- 자체 코드 라이선스 후보와 외부 라이선스 적용 범위. 추천 Apache-2.0, 아직 채택 전.
- wheel에 포함되는 외부 고지와 원본 소스/자산 출처.
- 코드·모델·배포 계약과 버전/내용 해시 고정.
- Git 이력과 기존 비공개 운영 소스 첨부의 알려진 토큰/개인키 패턴 검사.

검증 결과는 [public-preparation-validation.json](public-preparation-validation.json)에 있습니다.
SAM worker 후속 검증은 [Studio SAM 설치 안내](https://github.com/redEddie/showonce-studio/blob/main/docs/SAM-WORKER.md)에 있습니다.
이 검사를 나머지 GPU worker·실제 카메라·로봇 또는 외부 팀 소스의 설치 검증으로 확대하지 않습니다.

## 공개 전 남은 결정/검증

| 항목 | 상태 | 공개할 때의 표시 |
|---|---|---|
| 자체 코드 라이선스/소유자 | 사용자 결정 필요 | 선택 후 루트 LICENSE/SPDX 표기 |
| CPU 설치·예제 | 새 격리 환경 검증 | Python 3.12/Linux aarch64 검증; x86 새 설치는 별도 |
| SDK/제어 커널 | 기존 검증 보존 | 실제 모터 동작은 이번 준비에서 실행하지 않음 |
| SAM worker | Thor 새 환경 설치·pip check·녹화 RGB 추론 검증 | 원본 SAM/NumPy 1.x, Studio/NumPy 2.x 및 기존 HTTP/파일 연결 검증 |
| 나머지 GPU/실시간 통합 | 별도 새 환경 재현 필요 | SAM 검증을 전체 카메라·FoundationPose·실기 검증으로 확대하지 않음 |
| FoundationPose | 원본·파생 부분 비상업 연구/평가 조건 | 전체 파이프라인 상업 사용 가능으로 표시하지 않음 |
| real2sim | 확인 리비전 LICENSE 없음 | 작성자 조건 확인 전 공개 기본 설치에 넣지 않음 |
| 실시간 인식 | Studio 내부 이식 및 외부 소스 차단 검사 완료 | 모델 가중치·현재 보정·capture 필요; 현장 두 물체 완전 관측은 미확인 |
| 보정 생성 | Studio 내부 이식, 원본 로봇 모델/설정 포함 | [보정 안내](https://github.com/redEddie/showonce-studio/blob/main/docs/CAMERA-CALIBRATION.md), native 환경 별도 준비 |
| 시작 자세 A 복귀 | Studio 내부, 외부 실행기 불필요 | 기존 QP/Ruckig·모델 경계 복귀 보존; 새 실기 이동 미실행 |
| 스크립트 파지 | 독립 터미널 튜토리얼 | 웹앱·외부 실행기·정책 가중치 없이 사용; 기본 정책 흐름에서 제외 |
| 비공개 실기 Release/영상 | 소스 첨부 간이 검사 완료 | 공개할 첨부·미디어 범위는 소유자 결정 |

Studio와 Runtime은 학습 프로젝트와 별도로 코드/CPU 예제 범위를 공개할 수 있도록 준비합니다.
전체 RGB-D→GPU 인식→자동 보정→실기 실행이 새 장비에서 즉시 동작하는 배포물이라고 발표하지 않습니다.
학습 인계본 `piper-learning`과 현재 3090 원본 `piper-sim-lab`은 이 준비의 변경 대상이 아닙니다.

## 공개 실행 시 별도로 할 일

GitHub 공개는 2026-11-10 요청에 따라 진행했습니다. 자체 코드 라이선스 결정과 새 환경 통합 검증은 별도 항목으로 남습니다. 원본 가중치·실측 보정·토큰·촬영 원본은 자동 포함하지 않습니다.
