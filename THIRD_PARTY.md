# Runtime 외부 코드·자산

프로젝트 자체 코드 라이선스는 미확정입니다. 추천안은 [라이선스 제안](docs/LICENSE-PROPOSAL.md)에 있습니다.

| 구성 | 포함/설치 형태 | 조건과 출처 |
|---|---|---|
| Piper 운동학 XML·안전 형상 | `src/piper_runtime/data/` | 기존 변환 자산. MIT 고지 `data/notices/LICENSE` 보존. 아래 범위 설명 참고 |
| NumPy | 기본 외부 의존성 | NumPy의 BSD 계열·개별 외부 고지 |
| SciPy | training extra | BSD 계열·외부 고지 |
| JAX | training extra | Apache-2.0 |
| OSQP | training extra | Apache-2.0, QP 비교/CPU 계산 |
| Ruckig | training extra | MIT, 궤적 생성 비교 |
| pyAgxArm | robot extra, pinned Git 소스 | LGPL v3, https://github.com/agilexrobotics/pyAgxArm/tree/841a625f5f4920e776f20b934eb13048b747e6d0 |
| python-can | robot extra | LGPL-3.0 계열, https://github.com/hardbyte/python-can |

외부 패키지의 설치는 해당 코드를 우리 라이선스로 재배포한다는 뜻이 아닙니다.
완성 환경·바이너리를 재배포할 경우 그 환경의 개별 라이선스와 소스 제공 조건도 확인합니다.
SDK·NumPy 등의 본체를 이 소스 저장소에 복제하지 않았습니다.

## 포함 모델의 범위

이 Runtime은 Piper 운동학용 XML과 테이블 간격 검사 형상만 포함합니다.
전체 MuJoCo 시각 mesh는 포함하지 않으므로 이 XML을 독립 렌더링 씬으로 해석하지 않습니다.
`piper_runtime.kinematics`가 아니라 `piper_policy_runtime.kinematics.PiperKinematics`에서
공통 자산을 읽으며, 전체 물리 학습 모델은 별도 학습 프로젝트에서 관리합니다.

제조사 모델 출처는 `agilexrobotics/piper_ros` noetic
`ac41fcbcdda598f01b51cf6175ed9a24d0dacadc`이며 기존 고지의 저작권자는 RosenYin입니다.
안전 형상 JSON은 기존 제조사 URDF와 팀 MJCF에서 변환한 점/볼록 형상에 대한 provenance를 포함합니다.
그 파일에 기록된 로컬 경로는 역사적 출처이며 설치 시 읽는 입력 경로가 아닙니다.
과거 v0/v1 전체 학습 자산 설명은 `third-party-notices/ASSETS.md`에 보존되어 있습니다.
원본 고지와 변환 형상의 공개 권한을 공개 전 범위에 포함해 검토합니다.

학습 체크포인트·실측 보정·녹화 원본·SDK 바이너리는 이 Runtime에 포함하지 않습니다.
