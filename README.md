# CVaR 포트폴리오 최적화 웹 애플리케이션

NVIDIA [cufolio](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization) 라이브러리 기반의 Mean-CVaR 포트폴리오 최적화 웹 애플리케이션입니다.

## 주요 기능

- **CVaR 최적화** - S&P 500 데이터 기반 Mean-CVaR 포트폴리오 최적화 및 백테스트
- **효율적 프론티어** - 위험 회피 계수 스윕을 통한 효율적 프론티어 생성
- **CVaR 이론 가이드** - 한국어로 작성된 CVaR 이론 및 사용법 설명
- **GPU 가속** - NVIDIA cuOpt 솔버를 통한 GPU 가속 최적화

## 사전 요구사항

- NVIDIA GPU (cuOpt 지원)
- [NVIDIA quantitative-portfolio-optimization](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization) 환경이 `/workspace/quantitative-portfolio-optimization/`에 설치되어 있어야 합니다 (NVIDIA Brev 컨테이너 기준).

## 서버 실행

### 방법 1: 실행 스크립트 사용

```bash
./run.sh
```

### 방법 2: 직접 실행

```bash
cd webapp
/workspace/quantitative-portfolio-optimization/.venv/bin/python app.py
```

서버가 시작되면 다음과 같은 메시지가 출력됩니다:

```
 * Running on http://0.0.0.0:5000
```

## 접속 방법

브라우저에서 아래 주소로 접속합니다:

```
http://localhost:5000
```

외부에서 접속하는 경우 서버의 IP 주소를 사용합니다:

```
http://<서버IP>:5000
```

## 테스트 실행

NVIDIA 노트북(cvar_basic.ipynb, efficient_frontier.ipynb)과 동일한 결과가 나오는지 검증하는 테스트입니다:

```bash
cd webapp
/workspace/quantitative-portfolio-optimization/.venv/bin/python tests/test_comparison.py
```

## 프로젝트 구조

```
hedgeFund/
├── run.sh                          # 서버 실행 스크립트
├── nvidia-cufolio/                 # git submodule (NVIDIA cufolio)
└── webapp/
    ├── app.py                      # Flask 백엔드
    ├── requirements.txt            # 웹앱 의존성
    ├── templates/index.html        # 한국어 웹 UI
    └── tests/test_comparison.py    # 노트북 결과 비교 테스트
```
