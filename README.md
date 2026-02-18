# CVaR 포트폴리오 최적화 웹 애플리케이션

NVIDIA [cufolio](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization) 라이브러리 기반의 Mean-CVaR 포트폴리오 최적화 웹 애플리케이션입니다.

## 주요 기능

- **CVaR 최적화** - S&P 500 데이터 기반 Mean-CVaR 포트폴리오 최적화 및 백테스트
- **효율적 프론티어** - 위험 회피 계수 스윕을 통한 효율적 프론티어 생성
- **CVaR 이론 가이드** - 한국어로 작성된 CVaR 이론 및 사용법 설명
- **GPU 가속** - NVIDIA cuOpt 솔버를 통한 GPU 가속 최적화
- **모바일 반응형** - 모바일 환경에서도 차트와 테이블이 정상 표시
- **데이터 자동 업데이트** - 서버 시작 시 S&P 500 데이터를 최신으로 자동 업데이트

## 사전 요구사항

- NVIDIA GPU
- Docker (GPU 지원)
- NVIDIA 컨테이너 이미지 (nvidia_torch:portfolio2, ID: 117230367a1c)

## 환경 설정 (처음 한 번만)

### 1. Docker 컨테이너 생성

```bash
# WSL Ubuntu 또는 Linux에서 실행
docker run --gpus all -it \
  -v ./:/workspace/host \
  --ipc=host \
  -p 5000:5000 \
  --name hedgefund_dev \
  117230367a1c
```

### 2. 컨테이너 내부에서 레포 클론 및 설정

```bash
cd /workspace
git clone --recurse-submodules https://github.com/ssjl06/hedgeFund.git
cd hedgeFund
git checkout dev/v2
git submodule update --init --recursive

# 필요한 패키지 설치
/workspace/quantitative-portfolio-optimization/.venv/bin/python -m pip install flask plotly gunicorn
```

## 서버 실행

### 개발 모드 (디버그)

```bash
cd /workspace/hedgeFund/webapp
/workspace/quantitative-portfolio-optimization/.venv/bin/python app.py
```

### 프로덕션 모드 (gunicorn)

```bash
cd /workspace/hedgeFund/webapp
/workspace/quantitative-portfolio-optimization/.venv/bin/gunicorn \
  --bind 0.0.0.0:5000 \
  --workers 1 \
  --timeout 300 \
  app:app
```

> workers는 GPU 메모리 제한으로 1로 설정합니다.
> timeout은 최적화 계산 시간을 고려하여 300초로 설정합니다.

### 컨테이너 재시작 후

```bash
# 컨테이너가 멈춰있으면 시작
docker start hedgefund_dev

# 컨테이너에 접속
docker exec -it hedgefund_dev bash

# gunicorn 실행
cd /workspace/hedgeFund/webapp
/workspace/quantitative-portfolio-optimization/.venv/bin/gunicorn \
  --bind 0.0.0.0:5000 --workers 1 --timeout 300 app:app
```

## 접속 방법

### 로컬 접속

```
http://localhost:5000
```

### 외부 공개 (ngrok 사용)

외부 인터넷에서 접속할 수 있도록 ngrok 터널을 설정합니다.

#### 1. ngrok 설치

```bash
# Windows (winget)
winget install ngrok.ngrok

# 또는 https://ngrok.com/download 에서 직접 다운로드
```

#### 2. 인증 토큰 설정

[ngrok 대시보드](https://dashboard.ngrok.com/get-started/your-authtoken)에서 토큰을 확인합니다.

```bash
ngrok config add-authtoken <YOUR_TOKEN>
```

#### 3. 터널 실행

```bash
ngrok http 5000
```

터미널에 표시되는 `https://xxxx.ngrok-free.dev` URL로 외부에서 접속할 수 있습니다.

> 무료 플랜에서는 첫 접속 시 ngrok 경고 페이지가 나타납니다 (Visit Site 클릭).

## 테스트

```bash
cd /workspace/hedgeFund/webapp
/workspace/quantitative-portfolio-optimization/.venv/bin/python tests/test_comparison.py
```

## 프로젝트 구조

```
hedgeFund/
├── README.md
├── run.sh                          # 서버 실행 스크립트
├── nvidia-cufolio/                 # git submodule (NVIDIA cufolio)
└── webapp/
    ├── app.py                      # Flask 백엔드 (최적화, 백테스트, API)
    ├── requirements.txt            # 웹앱 의존성
    ├── templates/index.html        # 한국어 웹 UI (모바일 반응형)
    └── tests/test_comparison.py    # 노트북 결과 비교 테스트
```

## 기술 스택

- **Backend**: Flask + gunicorn
- **Optimization**: NVIDIA cufolio (CVaR, cvxpy, cuOpt)
- **Charts**: Plotly.js
- **Frontend**: Bootstrap 5 + Jinja2
- **Data**: yfinance (S&P 500)
