# Korean Linux Colab 개발 계획

> **목표**: Google Colab에서 한국어 자연어로 리눅스 명령어를 실행하는 학습 환경 구축

---

## 1. 프로젝트 개요

### 1.1 왜 Colab인가?

| Docker 방식 | Colab 방식 |
|-------------|------------|
| Docker 설치 필요 | ❌ 설치 불필요 |
| 로컬 GPU 필요 | ✅ 무료 GPU 제공 |
| 복잡한 환경 설정 | ✅ 웹 브라우저만 있으면 OK |
| CPU 모드 불안정 | ✅ CUDA 환경 안정 |

### 1.2 사용 모델

- **모델**: [HybriKo-117M-LinuxFC-SFT-v2](https://huggingface.co/Yaongi/HybriKo-117M-LinuxFC-SFT-v2)
- **아키텍처**: Griffin-style Hybrid (RNN + Attention, 2:1 비율)
- **파라미터**: 117.8M
- **정확도**: Action Name 100% (21개 명령어)

### 1.3 지원 명령어 (21개)

```
ls, cd, mkdir, rm, cp, mv, find, cat, grep, head, 
tail, wc, ps, df, du, top, ping, curl, chmod, tar, Finish
```

---

## 2. Colab 명령어 호환성

| 명령어 | Colab | 조정 사항 |
|--------|:-----:|-----------|
| `ls`, `mkdir`, `rm`, `cp`, `mv` | ✅ | - |
| `find`, `cat`, `grep` | ✅ | - |
| `head`, `tail`, `wc` | ✅ | - |
| `ps`, `df`, `du` | ✅ | - |
| `ping` | ✅ | `-c` 옵션 자동 추가 |
| `curl`, `chmod`, `tar` | ✅ | - |
| `cd` | ⚠️ | `os.chdir()` 사용 |
| `top` | ⚠️ | `top -b -n 1` 사용 |
| `Finish` | ✅ | Python 레벨 처리 |

**결론**: Python 레벨에서 조정하면 **모두 사용 가능**

---

## 3. 사용 방식

### 간단한 함수 호출
```python
# 모델 로딩 후
한글("현재 폴더의 파일 목록을 보여줘")
# 출력: 
# 🤖 명령어: ls -la
# 📁 결과:
# total 8
# drwxr-xr-x 2 root root 4096 ...
```

### 실행 흐름
```
한국어 입력 → HybriKo 모델 → 파라미터 보정 → 명령어 실행 → 결과 출력
```

---

## 4. 파일 구조

```
korean-linux-colab/
├── README.md                 # 프로젝트 소개
├── development_plan.md       # 이 파일
├── 한글_linux.ipynb          # 메인 Colab 노트북
├── src/
│   ├── korean_linux.py       # 핵심 패키지
│   └── command_executor.py   # 명령어 실행 + 보정
└── examples/
    └── sample_files/         # 테스트용 파일
```

---

## 5. 파라미터 보정 로직

### 5.1 cd 명령어 처리
```python
if action == "cd_command":
    os.chdir(params["path"])  # !cd는 작동 안 함
```

### 5.2 top 명령어 처리
```python
if action == "top_command":
    return "top -b -n 1"  # 1회만 실행
```

### 5.3 ping 명령어 처리
```python
if action == "ping_command":
    count = params.get("count", 4)
    return f"ping -c {count} {params['host']}"
```

### 5.4 위험 명령어 확인
```python
if action == "rm_command" and params.get("recursive"):
    print("⚠️ 경고: 재귀 삭제 명령입니다. 계속하시겠습니까?")
```

---

## 6. 개발 단계

| 단계 | 작업 | 상태 |
|------|------|:----:|
| 1 | 폴더 구조 생성 | 🔄 |
| 2 | korean_linux.py 구현 | ⬜ |
| 3 | command_executor.py 구현 | ⬜ |
| 4 | 한글_linux.ipynb 생성 | ⬜ |
| 5 | 예제 파일 생성 | ⬜ |
| 6 | 테스트 및 문서화 | ⬜ |

---

## 7. Quick Start (예정)

```python
# Colab에서 실행
!pip install -q huggingface_hub sentencepiece
!git clone https://github.com/gyunggyung/korean-linux-colab.git
%cd korean-linux-colab

# 사용
from src.korean_linux import 한글
한글("현재 폴더에 있는 파일들을 보여줘")
```
