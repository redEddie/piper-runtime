# GitHub 공개 상태 — 2026-11-10

소유자 요청으로 아래 세 저장소를 공개합니다. 각 저장소의 `main`은 현재 파일을 보존한 단일 `upload` 커밋입니다. 이전 이력은 로컬에 백업했으며 공개 브랜치로 보관하지 않습니다.

| 저장소 | 역할 |
|---|---|
| [showonce-studio](https://github.com/redEddie/showonce-studio) | RGB-D 수집, 스테이지 분할, 객체 관계, 정책 실행 UI |
| [piper-learning](https://github.com/redEddie/piper-learning) | 최종 SHOW-108 24배 학습·평가·내보내기 통합본 |
| [piper-runtime](https://github.com/redEddie/piper-runtime) | 공통 관측 계약, 추론, 제어, 실기 연결 |

Studio와 Learning은 `piper-runtime.lock.json`으로 동일한 Runtime 구현을 고정합니다. 이력 정리는 정책 가중치·정규화 통계·제어 수식을 변경하지 않습니다. 원본 학습 개발 저장소 `piper-sim-lab`의 이력과 공개 범위는 변경하지 않습니다.

## 라이선스와 배포 범위

GitHub에서 소스를 열람할 수 있도록 공개한 것이며, 자체 코드에 적용할 오픈소스 라이선스는 아직 선택하지 않았습니다. GitHub 공개 자체를 상업적 사용·수정·재배포 권리 허여로 설명하지 않습니다. 외부 코드·모델·자산은 각자의 라이선스와 고지를 따릅니다. `docs/LICENSE-STATUS.md`와 각 저장소의 외부 의존성 고지를 확인하세요.

기존 Studio Release의 실기 영상·실행 기록·운영 소스 첨부는 원본 바이트를 보존합니다. 태그는 새 `upload` 커밋을 가리키므로 당시 소스는 운영 소스 첨부에서 확인합니다. 새 Release나 PyPI 배포는 만들지 않습니다.

새 장비에서는 CUDA·RealSense·각 GPU worker 환경, 모델 접근 권한과 로컬 보정이 별도로 필요합니다. 기존 설치 검증 기록은 각 기록의 날짜와 범위에 해당하며 이번 공개로 재검증된 것으로 확대하지 않습니다. 실측 보정·모델 캐시·인증정보·촬영 원본 데이터는 추가로 업로드하지 않습니다.

개인 Plane 서버 주소는 포함하지 않습니다. 이슈는 번호와 설명으로만 참조합니다.
