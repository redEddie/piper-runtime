# Piper Runtime

Studio: [redEddie/showonce-studio](https://github.com/redEddie/showonce-studio) · 학습: [redEddie/piper-learning](https://github.com/redEddie/piper-learning) · 공통 실행: [redEddie/piper-runtime](https://github.com/redEddie/piper-runtime)

세 저장소는 GitHub에 공개되어 있습니다. 자체 코드 라이선스는 아직 미정이며, [공개 범위와 조건](docs/PUBLICATION.md)을 확인하세요.

Studio와 강화학습 프로젝트가 함께 설치하는 Python 배포 패키지 `piper-runtime==0.5.0`입니다.

```bash
python -m pip install .
piper-runtime-info
```

기본 설치는 NumPy 추론·운동학·계약을 제공합니다. 공통 유틸리티는 NumPy 1.26.4/2.5.4를 검증했고,
Studio/control은 2.x, 원본 SAM worker는 별도 1.x 환경을 사용합니다. JAX 학습 커널은 `.[training]`,
실기 SDK 어댑터는 `.[robot]`로 분리합니다. import만으로 SDK/CAN/카메라를 열지 않습니다.
CPU native planner의 첫 사용에는 C++17 컴파일러가 필요합니다.

[새 환경 설치](docs/INSTALLATION.md) · [외부 코드·자산](THIRD_PARTY.md) · [공개 준비 상태](docs/PUBLIC-RELEASE.md) · [라이선스 제안](docs/LICENSE-PROPOSAL.md)

## 포함 범위

- `piper_runtime.contracts`: 관측 필드·행동 의미·TCP·주기와 실행 프로필
- `piper_policy_runtime`: 기존 NumPy policy/운동학/CPU 제어 경로. 기존 import 이름을 보존합니다.
- `piper_runtime.control`: 학습에서 사용하는 JAX QP/jerk 계획과 CPU 수학 도구
- `piper_runtime.transports`: 명시적으로 구성하는 SDK/CAN 및 가짜 피드백
- `piper_runtime.resources`: 검증된 Piper XML 및 테이블 간격 검사 형상
- `piper_runtime.identity`: 소스·자산 해시를 통한 소비자 버전 고정

학습 환경·보상·체크포인트·인식 모델은 포함하지 않습니다. XML은 운동학용으로 제공하며,
MuJoCo 렌더링용 전체 메시 자산은 학습 저장소의 출처/고지와 함께 관리합니다.

## 최종24배 기준

사용자 확정한 SHOW-108 관측DR·확장스폰 MLP+LSTM24배는 기존0.5.0의 추론·제어 구현과 호환됩니다. 구현을 바꾸거나 버전을 올리지 않고 최종 learner의 원본 JAX 벡터 회귀 검사를 추가했습니다. [3저장소 연결·검증 범위](docs/FINAL24-INTEGRATION.md), [고정 모델 명세](compatibility/show108-24u-final.json)를 따릅니다.

## 서로 다른 프로필

`studio-legacy-serialized-v1`은 현재 검증용 Studio의 순차 제어 동작을 보존합니다.
`show103-stage-reward-v1`은 최신 인계 설계의 actor49/critic66/action7, 속도1/가속도5/jerk100,
TCP138.5mm, 별도 MOVE/BRAKE/GRIP 조정기 없는 학습 계약입니다.
이는 과거 SHOW-103 인계 당시 상태입니다. 현재24배 기준은 위 최종 연결 문서를 따릅니다. 이 정책을 이전 순차 실행기로 자동 대체하지 않습니다.
CPU와 JAX 구현이 같은 패키지에 있다는 사실만으로 두 프로필의 폐루프 동등성을 주장하지 않습니다.

`show108-mlp-lstm-v1`은 선정한 SHOW-108 모델의 별도 실기 어댑터입니다.
MLP256→64→LayerNorm→마지막 LSTM128→선형14를 NumPy FP32로 실행하며,
원본 Flax 출력·메모리와 JAX checked-plan 검증 벡터를 비교합니다. 그리퍼는 팔과 독립적으로
최종 목표를 유지하고 매 100ms마다 전체 폭 참조를 최대20mm 변경합니다.
실기 TCP 하한 테이블+10mm, SDK force0.5는 학습과 다른 배포 설정으로 명시합니다.
이전 정책에는 기존 순차 제어기를 유지합니다. 검증은 실기 집기 성능 평가를 대신하지 않습니다.

```bash
python -m pip install '.[training,test]'
python -m pytest -q
piper-runtime-info --lock ../showonce-studio/piper-runtime.lock.json
```

소비자는 버전과 내용 해시를 함께 고정합니다. 공통 커널을 바꾸면 회귀 검증 뒤 버전을 올리고,
두 소비자의 lock 및 정책 export를 함께 갱신합니다. 초기 추출 검사는 기존 정책 추론과
동일 가짜 피드백 400 제어 주기 비교입니다. 실기 작업 성공률 검증은 아닙니다.

프로젝트 코드의 공개 라이선스는 소유자 결정 전입니다. `third-party-notices/`를 보존합니다.
