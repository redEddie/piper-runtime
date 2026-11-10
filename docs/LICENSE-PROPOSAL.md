# 프로젝트 라이선스 제안 — 아직 적용 전

**추천: Studio와 Runtime의 자체 코드에 Apache-2.0.** 현재 소유자 선택은 미확정이며
루트 LICENSE나 패키지의 SPDX license 필드를 추가하지 않았습니다. 이 문서는 권리 허여가 아닙니다.

| 후보 | 선택 이유/차이 |
|---|---|
| Apache-2.0 (추천) | 상업 사용·수정·배포를 허용하는 permissive license. 기여자의 특허 허여와 특허 소송 시 종료 조항, 변경·고지 유지 규정이 명시되어 로봇/기업 협업에 적합 |
| MIT | 짧고 단순한 permissive license. 사용·수정·배포 및 저작권/허가문 유지 조건. Apache와 같은 명시적인 특허 허여 조항은 없음 |

공식 문서: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0),
[MIT](https://opensource.org/license/mit).
어느 후보도 타인의 특허·모델·데이터 전체에 대한 권리를 보장하지 않습니다.

## 적용 범위

- 자체 작성한 Studio UI/API/작업 관리/어댑터와 Runtime 계약·제어·추론 코드가 대상입니다.
- Three.js/ECharts, Kaolin 호환 함수, Piper 모델, 외부 패치와 모델 가중치는 원래 고지를 유지합니다.
- FoundationPose 관련 원본·파생 패치의 비상업 조건과 SAM/모델 조건을 Apache로 바꾸지 않습니다.
- real2sim 및 외부 팀 통합 코드의 권한은 별도 확인 대상입니다.
- 영상·실측 데이터·생성 에셋·학습 체크포인트를 코드 라이선스로 자동 허가하지 않습니다.

## 선택 후 적용할 내용

1. 코드 소유자/기여자와 공개 가능한 범위를 확인합니다.
2. 선택한 정식 LICENSE를 두 저장소에 넣고 외부 파일 예외·고지를 유지합니다.
3. `pyproject.toml`의 SPDX 표기 및 sdist/wheel에 포함되는 고지를 확인합니다.
4. 기존 비공개 Release의 운영 소스·영상 공개 범위를 결정합니다.

GitHub 공개 전환과 PyPI 배포는 이 준비 작업에서 수행하지 않습니다.
