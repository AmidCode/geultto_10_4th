# 글또 10기 - 파이썬 3부작

이 repository는 파이썬 3부작 게시글의 상세 코드입니다.

# 디렉터리 구성

```
.
└── third
    ├── containers
    │   ├── alpine
    │   │   ├── 3.12
    │   │   └── 3.13
    │   └── debian
    │       ├── 3.12
    │       └── 3.13
    ├── cpu
    ├── io
    │   └── query
    │       └── tests
    └── tests

```

- `third`: 3부 관련 코드
    - `containers`: Alpine, Debian 기반 Dockerfile 모음집
    - `cpu`: CPU bound task 코드
    - `io`: IO bound task 코드
    - `tests`: 기타 학습 테스트 및 작업중인 코드 모음집

# Workarounds

## 의존성 추출방법

- cpu
    - `poetry export -f requirements.txt --only cpu --output cpu-requirements.txt`
    - 이후 파일 옮기기

- io
    - `poetry export -f requirements.txt --only io --output io-requirements.txt`
    - 이후 파일 옮기기
