# Runtime 설치

Python 3.12 / Linux를 기준으로 검증합니다. 현재 저장소는 GitHub에 공개되어 있으며 PyPI에는 게시하지 않았습니다. [공개 범위](PUBLICATION.md)를 확인하세요.

```bash
git clone https://github.com/redEddie/piper-runtime.git
cd piper-runtime
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
python -m pip check
piper-runtime-info
python -m pytest -q
```

기본 의존성은 NumPy입니다. 공통 유틸리티는 NumPy 1.26.4와 2.5.4 양쪽에서 검증했습니다.
Studio/control 설치는 NumPy 2.5.4, 원본 SAM worker는 별도 환경에서 NumPy 1.26.4를 사용합니다. 기본 import/테스트는 GPU·JAX·카메라·SDK/CAN 연결을 요구하지 않습니다.
CPU planner의 첫 사용은 C++17 컴파일러(`g++` 또는 `clang++`)와 공유 라이브러리 빌드 가능한 임시 경로가 필요합니다.
검사에서는 저장된 Flax/JAX golden 벡터와 가짜 피드백을 사용합니다.

## Studio에서 사용

형제 checkout에 Studio를 두고 같은 환경에서 설치합니다.

```bash
python -m pip install '../showonce-studio[analysis,test]'
python -m piper_runtime.identity --lock ../showonce-studio/piper-runtime.lock.json
```

현재 버전은 0.5.0이고 정책 패키지는 **버전과 내용 해시**를 고정합니다.
코드·모델·수식을 수정하면 새 버전과 소비자 lock/export 검증이 필요합니다.
문서·외부 고지 변경은 기존 제어 커널을 변경하지 않습니다.

## 사용 역할에 따른 필수 의존성

해당 역할을 사용할 때는 그 extra의 의존성이 필수입니다. 예를 들어 실기 SDK 어댑터를 쓰려면 `robot`을 설치해야 합니다.

| extra | 설치되는 내용 | 검증 범위 |
|---|---|---|
| `test` | pytest | CPU 단위·golden 비교 |
| `training` | JAX 0.10.2, SciPy, OSQP 1.1.3, Ruckig 0.19.4 | 학습 쪽 수학 커널; CUDA 환경 설치를 대신하지 않음 |
| `robot` | pinned pyAgxArm, python-can | SDK 어댑터; 설치/import만으로 모터 명령을 보내지 않음 |

```bash
# 별도 장비 제어 환경에서만 선택
python -m pip install '.[robot]'
# 학습용 CPU 수학 환경에서 선택
python -m pip install '.[training,test]'
```

실제 정책 실행에는 별도 Studio worker, 현재 보정·실시간 인식, 선택 정책의 체크포인트/정규화 통계가 필요합니다.
이 저장소만으로 영상 인식·자동 보정·새 학습 전체가 설치되지는 않습니다.
SDK는 LGPL 조건을 적용하며 수정·재배포 시 원본 고지를 확인합니다.
[외부 의존성](../THIRD_PARTY.md)과 [공개 준비 상태](PUBLIC-RELEASE.md)를 참고하세요.
