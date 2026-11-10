# ShowOnce: 세 저장소 작업본

현재 서비스와 원본 프로젝트를 유지하면서 `/home/pi_lab/projects` 아래에 만든 별도 작업본입니다.
이 디렉터리는 작업 공간이며 Git 저장소는 아래 세 하위 디렉터리입니다.

| 저장소 | 역할 | 공통 의존성 |
|---|---|---|
| showonce-studio | 수집·스테이지·객체 관계·정책 실행 준비 | piper-runtime==0.4.0 |
| piper-learning | 환경·보상·학습·평가·정책 export | piper-runtime[training]==0.4.0 |
| piper-runtime | 관측·행동·운동학·CPU/JAX 제어·명시적 SDK 어댑터 | NumPy 기본, 학습/실기 extra 선택 |

두 소비자는 같은 버전뿐 아니라 `piper-runtime.lock.json`의 내용 해시를 고정합니다.
세 저장소를 `redEddie` 계정의 GitHub 비공개 저장소로 생성하고 원격 `origin`을 연결해 업로드했습니다.

Studio: [redEddie/showonce-studio](https://github.com/redEddie/showonce-studio) · 학습: [redEddie/piper-learning](https://github.com/redEddie/piper-learning) · 공통 실행: [redEddie/piper-runtime](https://github.com/redEddie/piper-runtime)

2026-11-10 세 저장소의 이력을 각각 단일 `upload` 커밋으로 정리하고 공개했습니다. 자체 코드 라이선스는 미정입니다. [현재 공개 범위](PUBLICATION.md)를 따르며, 아래 최초 분리 기록은 당시 상황입니다.
 학습 코드는 기존 `show-103`
인계 commit의 snapshot이며 원본 Git 이력/원본 선택 정책은 기존 위치에 그대로 있습니다.

## 유지한 운영 경로

- 기존 웹앱: `/home/pi_lab/piper_ws`, localhost:8765 (원격 PC의 18765 터널)
- 기존 카메라/SAM/씬 노드 및 가상환경: 변경·재시작하지 않음
- 기존 학습/인계 checkout: 변경하지 않음
- 실제 로봇 SDK/CAN/모터 동작: 이번 분리 검증에서 실행하지 않음

새 코드의 설치/검증은 `.verification`, wheel 설치는 `.wheel-verification`에서만 수행합니다.
운영 환경의 pip/conda 설정을 수정하지 않습니다. 검증용 새 웹은 18766/offline 모드이며 검증 후 종료합니다.

## 로컬 사용

각 저장소 README를 따릅니다. 가장 작은 예제는 Studio의 합성 예제입니다.
`showonce-studio/configs/local.toml`은 이 머신 검증용이고 Git에 포함하지 않습니다.
새 머신은 `example.toml`을 복사하고 자신의 경로/인터프리터/보정을 설정합니다.

공통 코드를 변경하면 두 소비자의 잠금을 함께 갱신하고 회귀 검사를 진행합니다.
현재 Studio 순차 제어 프로필과 최신 SHOW-103의 순차 조정기 없는 프로필은 다릅니다.
새 SHOW-103 정책을 이전 실행기로 자동 대체하지 않습니다. 새 MJX task/물리 검증은 원본 인계의 미완료 항목입니다.

## 최초 분리 당시 공개와 운영 전환 전 남은 결정

1. 프로젝트 코드 라이선스를 소유자가 결정하고 외부 코드·모델·자산 고지를 유지합니다.
2. 비공개 저장소에서 설치·의존성 및 공개할 파일 범위를 검토한 뒤, 라이선스가 결정되면 공개 전환합니다.
3. GPU 모델 환경은 새 머신에서 설치/회귀 검증합니다. 이번 검증은 CPU·합성 예제·패키징입니다.
4. 실제 녹화/실기 흐름을 검증한 뒤 운영 서비스를 새 코드로 명시적으로 전환합니다.

현재 제공한 로컬 설정은 데이터 경로를 분리합니다. 원본 영상/에셋/체크포인트를 자동으로 이동하거나
새 앱의 가짜 예제 데이터로 덮어쓰지 않습니다.
