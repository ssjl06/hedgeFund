# CVaR Portfolio Optimization Web Server

NVIDIA의 [cvar_basic.ipynb](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/blob/main/notebooks/cvar_basic.ipynb)와
[efficient_frontier.ipynb](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/blob/main/notebooks/efficient_frontier.ipynb)를
기반으로 한 Mean-CVaR 포트폴리오 최적화 웹 애플리케이션입니다.

## 아키텍처

```
Browser (localhost:5000)
    │
    ▼
Docker Container (-p 5000:5000)
    │
    ▼
Gunicorn (:5000)  ←  WSGI 서버
    │
    ▼
Flask App (app.py)
    ├─ optimizer.py       →  Mean-CVaR LP/MILP 최적화 (scipy)
    ├─ scenarios.py       →  시나리오 생성 (Historical / KDE / Gaussian)
    ├─ frontier.py        →  효율적 프론티어 분석
    ├─ charts.py          →  기본 Plotly 차트
    ├─ charts_advanced.py →  고급 최적화 차트
    └─ charts_frontier.py →  프론티어 차트
```

## 빠른 시작 (Docker 컨테이너 내)

### 1. 컨테이너 시작 (WSL 터미널에서)

```bash
# 포트 매핑과 함께 컨테이너 실행
docker run -it -p 5000:5000 -v /path/to/hedgeFund:/workspace/hedgeFund <이미지명>
```

### 2. 의존성 설치

```bash
cd /workspace/hedgeFund/webapp
pip install -r requirements.txt
```

### 3. 서버 실행

```bash
# 개발 모드
python app.py

# 프로덕션 모드 (Gunicorn, 워커 2개)
gunicorn -b 0.0.0.0:5000 -w 2 app:app
```

### 4. 접속

```
http://localhost:5000
```

## 기능

### 기능 1: CVaR 포트폴리오 최적화

NVIDIA cvar_basic.ipynb 기반의 고급 포트폴리오 최적화.

| 매개변수 | 설명 | 기본값 |
|----------|------|--------|
| 종목 코드 | Yahoo Finance 티커 (최소 2개) | — |
| 시작/종료일 | 과거 데이터 기간 | 최근 2년 |
| CVaR 신뢰수준 (β) | 꼬리 위험 백분위 | 0.95 |
| 위험 회피도 (λ) | 0=안전, 1=공격적 | 0.50 |
| 수익률 유형 | 단순(Simple) / 로그(Log) | Simple |
| 시나리오 방법 | Historical / KDE / Gaussian | Historical |
| 시나리오 수 | KDE/Gaussian 생성 시나리오 수 | 자동 |
| KDE 대역폭 | 커널 밀도 추정 대역폭 | 0.5 |
| 비중 제약 | 전체 및 종목별 최소/최대 비중 | 0.0 / 1.0 |
| 현금 배분 | 현금 비중 최소/최대 | 0 / 0 |
| 레버리지 목표 | 총 투자 비중 목표 | 1.0 |
| CVaR 한도 | 일별 CVaR 상한 (%) | 없음 |
| 카디널리티 | 최대 종목 수 (MILP) | 없음 |
| 턴오버 제한 | 리밸런싱 허용량 | 없음 |
| Train/Test 분할 | 백테스트 분할일 | 없음 |
| 벤치마크 | 비교 포트폴리오 (JSON) | 없음 |

**출력 지표**: 기대 수익률, CVaR, 변동성, Sharpe, Sortino, 최대 낙폭(MaxDD)

**출력 차트**: 누적 수익률 (Train/Test 분할), 포트폴리오 비중, 시나리오 분포, 벤치마크 비교

### 기능 2: 효율적 프론티어

NVIDIA efficient_frontier.ipynb 기반의 프론티어 분석.

| 매개변수 | 설명 | 기본값 |
|----------|------|--------|
| 종목 코드 | Yahoo Finance 티커 (최소 2개) | — |
| 시작/종료일 | 과거 데이터 기간 | 최근 2년 |
| CVaR 신뢰수준 (β) | 꼬리 위험 백분위 | 0.95 |
| 최소/최대 비중 | 종목별 비중 범위 | 0.0 / 1.0 |
| 최소 지수 | λ 스윕 시작 (10^x) | -3 |
| 최대 지수 | λ 스윕 종료 (10^x) | 1 |
| 단계 수 | 프론티어 포인트 수 | 50 |
| 비교 포트폴리오 | 프론티어 위에 표시할 포트폴리오 | 없음 |

**출력**: 효율적 프론티어 차트, 접선 포트폴리오 (최대 Sharpe), 비중 히트맵, λ 프로파일, CSV 내보내기

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 메인 UI |
| `GET` | `/api/presets` | 프리셋 종목 그룹 |
| `POST` | `/api/optimize` | 기본 최적화 (하위 호환) |
| `POST` | `/api/optimize-advanced` | 고급 CVaR 최적화 |
| `POST` | `/api/frontier` | 효율적 프론티어 분석 |
| `POST` | `/api/frontier/export` | 프론티어 CSV 내보내기 |

## 파일 구조

```
webapp/
├── app.py                  # Flask 라우트 및 입력 검증
├── optimizer.py            # Mean-CVaR LP/MILP 최적화 엔진
├── scenarios.py            # 시나리오 생성 (Historical/KDE/Gaussian)
├── frontier.py             # 효율적 프론티어 분석
├── charts.py               # 기본 Plotly 차트
├── charts_advanced.py      # 고급 최적화 차트
├── charts_frontier.py      # 프론티어 차트
├── templates/
│   └── index.html          # 프론트엔드 (Bootstrap 5 + Plotly.js, 한국어)
├── requirements.txt        # Python 의존성
├── deploy/                 # 레거시 배포 스크립트 (Nginx/systemd/PowerShell)
│   ├── setup.sh
│   ├── nginx.conf
│   ├── gunicorn.conf.py
│   ├── cvar-optimizer.service
│   └── windows-port-forward.ps1
└── README.md
```

## 이론적 배경

### Mean-CVaR LP (Rockafellar-Uryasev, 2000)

```
min  -λ·μ'w  +  (1-λ)·[α + 1/((1-β)S) · Σ z_s]

s.t.  z_s ≥ -r_s'w - α    ∀s
      z_s ≥ 0              ∀s
      Σ w_i + cash = L     (레버리지 목표)
      lo_i ≤ w_i ≤ hi_i    (종목별 비중 제약)
```

### 고급 제약 (확장 LP)

- **현금 배분**: `c_min ≤ cash ≤ c_max`
- **CVaR 한도**: `α + 1/((1-β)S) · Σ z_s ≤ CVaR_limit`
- **턴오버**: `Σ |w_i - w_i_current| ≤ T_tar` (보조 변수로 선형화)
- **카디널리티**: `Σ b_i ≤ K` (MILP, 이진 변수 `b_i`)

### 효율적 프론티어

λ를 로그 스케일(`10^min_exp` ~ `10^max_exp`)로 스윕하여 위험-수익 최적 조합 곡선을 계산합니다.
λ가 작을수록 수익 추구, 클수록 위험 회피.
