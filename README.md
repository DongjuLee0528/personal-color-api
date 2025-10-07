이 저장소에 대해

이 프로젝트는 [starbucksdolcelatte/ShowMeTheColor]를 기반으로 기능 수정·추가·삭제를 진행한 커스텀 포크입니다. 원본 라이선스와 크레딧을 유지합니다. 원본은 퍼스널 컬러 진단 로직 중심이며 웹 프레임워크 코드는 포함하지 않습니다. 
GitHub

변경 사항(요약)

FastAPI 기반 HTTP API 추가: /health, /analyze, /analyze/file, /face-shape, /face-shape/file

입력 검증 및 가드레일: INVALID_IMAGE / NO_FACE / LOW_QUALITY

MediaPipe 기반 랜드마크로 분석 파이프라인 재작성

피부 ROI 추출·지표(ITA/Lab) 기반 톤·시즌 판정 + 팔레트/디버그 이미지

Dockerfile/requirements, 추적 ID(X-Trace-Id) 지원, 단위 테스트 추가

업스트림(원본)과의 차이

dlib → MediaPipe 전환, HTTP API 제공, 얼굴형 분석 유틸 추가, 운영·품질 도구(도커/테스트) 보강