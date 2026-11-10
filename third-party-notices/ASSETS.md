# Piper 모델 출처와 사용 범위

이 패키지는 v0의 MuJoCo Menagerie 모델과 v1의 제조사 URDF 기반 모델을 포함한다.
소스 코드의 라이선스와 외부 모델 자산의 라이선스를 구별하며 자산의 원본 고지를 함께 배포한다.

## v1: 제조사 URDF 기반 100 mm 그리퍼

SHOW-76은 `src/piper_sim_lab/assets/agilex_piper_v100/`를 사용한다.
출처는 `agilexrobotics/piper_ros`, `noetic` revision
`ac41fcbcdda598f01b51cf6175ed9a24d0dacadc`의 `piper_description_v100.urdf`다.
[SOURCE.json](src/piper_sim_lab/assets/agilex_piper_v100/SOURCE.json)에 원본·변환 자산 해시가 있으며
[LICENSE.piper_ros](src/piper_sim_lab/assets/agilex_piper_v100/LICENSE.piper_ros)의 MIT 고지를 보존한다.
변환기는 `scripts/build_piper_v100.py`다. 기본 학습에는 포함된 MJCF·mesh를 사용하며 재변환하지 않는다.

관절·시각 형상은 이 URDF에서 변환하고 충돌 primitive·position actuator·손가락 결합·
관절 마찰·중력 보상·제한은 시뮬레이션 설정으로 추가했다.
학습 환경 TCP는 link6 로컬 z = 135.8 mm, 전체 명목 개구는 100 mm다.
제조사 모델을 사용해도 실제 제어기·관성·접촉이 식별된 동역학 모델을 뜻하지 않는다.
`piper-assets` CLI의 manifest 검사는 아래 v0 자산 대상이며 v1 해시는 별도 `SOURCE.json`에 있다.

## v0: MuJoCo Menagerie

| 항목 | 고정한 값 |
| --- | --- |
| 원본 저장소 | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) |
| 원본 경로 | [`agilex_piper`](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/agilex_piper) |
| 원본 revision | `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35` |
| 패키지 내 경로 | `src/piper_sim_lab/assets/agilex_piper/` |
| 자산 라이선스 | 디렉터리의 [MIT License](src/piper_sim_lab/assets/agilex_piper/LICENSE) |
| 저작권 고지 | `Copyright (c) 2024 RosenYin` |
| 파일별 SHA-256 / 크기 | [manifest.json](src/piper_sim_lab/assets/manifest.json) |

MIT 고지와 허가 문구를 자산 복사본 및 중요한 파생 부분에 유지한다. Menagerie 저장소 최상위의 라이선스만 보고 이 디렉터리의 개별 MIT 고지를 덮어쓰지 않는다. 자산의 라이선스는 모델 코드·메시에 관한 것으로, 로봇 하드웨어나 제조사 서비스의 이용 조건을 대신하지 않는다.

## 원본과 포함 범위

원본 [README](src/piper_sim_lab/assets/agilex_piper/README.md), `LICENSE`, `CHANGELOG.md`, `piper.xml`, `scene.xml` 및 메시 파일을 보존한다. 포함한 89개 파일의 크기와 SHA-256은 manifest에 기록되어 있다. README의 소개용 `piper.png`는 이 실행 자산 묶음에 포함하지 않았으므로 해당 이미지의 원본 표시는 [고정된 upstream README](https://github.com/google-deepmind/mujoco_menagerie/blob/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/agilex_piper/README.md)를 참고한다.

upstream README는 공개된 [AgileX Piper ROS URDF](https://github.com/agilexrobotics/Piper_ros/tree/ros-noetic-no-aloha/src/piper_description/urdf)에서 파생한 단순화 MJCF라고 설명한다. 또한 모델 기여자를 Omar Rayyan으로 표기한다. README의 기여자 표기와 LICENSE의 저작권자 표기는 서로 다른 고지이므로 둘 다 원문 그대로 유지한다. 이 패키지가 제조사 모델을 직접 변환했다거나 최신 하드웨어를 그대로 반영한다고 주장하지 않는다.

자산 업데이트 때는 revision과 manifest를 함께 갱신하고 MJCF 로딩·관절/액추에이터 계약·충돌·정책 호환성을 다시 검증한다. upstream 파일에 실험 변경을 덮어쓰기보다는 환경 코드에서 적용하는 변경을 구별해 추적한다.

## 모델의 물리·제어 해석

| 구성 | 이 모델에서의 의미 |
| --- | --- |
| 팔 | `joint1`–`joint6`의 6개 회전 관절과 position actuator |
| 그리퍼 | 두 slide joint `joint7`, `joint8`; actuator는 `joint7`을 구동하는 1개 |
| 손가락 결합 | equality의 `polycoef="0 -1 0 0 0"`에 따라 `joint8 = -joint7`을 제약 |
| 차원 | 8개 관절 좌표, 7개 액추에이터. equality가 있으므로 8개 독립 구동축이 아니다 |
| 그리퍼 범위 | `joint7`은 `[0, 0.035] m`, `joint8`은 `[-0.035, 0] m`; 두 손가락의 명목 최대 개구는 70 mm |
| reach 실험 설정 | 그리퍼 position target `0.025 m`, 명목 개구 50 mm; 정책 action은 팔 6개만 사용 |
| 시각 형상 | mesh 기반이며 visual geom은 충돌 비활성 |
| 충돌 형상 | 팔은 capsule 근사, 손가락은 box 근사. 원본은 `base_link`와 `link1` 접촉을 제외 |
| 중력 보상 | 원본의 이동 body에 `gravcomp="1"` 설정이 포함됨 |

이 환경은 원본의 팔·손가락 충돌 형상을 사용한다. “전체 충돌 형상 유지”는 해당 capsule/box들을 삭제하지 않는다는 뜻이며, 실제 메시 표면이나 모든 body pair의 접촉을 정밀하게 재현한다는 뜻은 아니다. 기본 MuJoCo 접촉 필터와 원본의 명시적 제외 조건도 적용된다.

MJCF의 관성, 마찰, position gain, 힘 제한, 중력 보상과 충돌 형상은 시뮬레이션 모델의 설정이다. 제조사 실물의 식별·보정된 동역학이나 제어기와 같다고 보장하지 않는다. 손가락 equality와 position control에도 수치·접촉 오차가 있을 수 있다. 이 reach 예제의 결과만으로 실물 토크·안전성·집기 성능이나 sim-to-real 전이를 검증했다고 해석하지 않는다.

실험 환경에서 추가하는 목표 marker, 말단 site, 바닥, 카메라, 시간 간격이나 solver 설정 등은 패키지 코드와 실행 설정의 변경 사항이다. 원본 asset revision과 함께 이 설정도 기록해야 같은 실험을 재현할 수 있다.

## 외부 소프트웨어와의 구별

MuJoCo, MJX/JAX, MuJoCo Playground, Brax는 설치하는 외부 소프트웨어다. 이 자산 묶음의 MIT 고지가 그 의존성의 라이선스를 대체하지 않는다. 각 의존성은 고정한 버전의 원본 라이선스를 따른다. PPO 코어는 Brax에서 사용하며 이 자산 디렉터리에 복사하지 않는다.

이 문서는 standalone Piper 학습 예제의 자산 기록이다. Dex-One2Many 원문의 로봇·시뮬레이터·모델을 대체하기로 확정한 문서가 아니다.

기본 실행 설정은 물리 간격 `0.005 s`, solver 반복 상한 `8`, line-search 반복 상한 `4`입니다. 원본 파일을 수정하지 않고 MjSpec에서 적용하며 YAML의 `sim_dt`, `solver_iterations`, `solver_ls_iterations`로 변경합니다. 충돌 형상과 equality는 유지합니다.
