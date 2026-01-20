# 🐧 Korean Linux Colab

> **한국어로 리눅스 명령어를 실행하세요!**  
> Google Colab에서 자연어로 리눅스를 배우는 가장 쉬운 방법

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gyunggyung/korean-linux-colab/blob/main/한글_linux.ipynb)

---

## ✨ 특징

- 🇰🇷 **한국어 자연어 입력** → 리눅스 명령어 자동 변환
- 🚀 **설치 불필요** - Colab에서 바로 실행
- 🛡️ **안전한 학습 환경** - Colab 가상 환경에서 실행
- 🤖 **AI 기반** - HybriKo-117M 모델 사용

---

## 📋 지원 명령어 (21개)

| 파일 관리 | 시스템 | 네트워크 |
|-----------|--------|----------|
| `ls`, `cd`, `mkdir` | `ps`, `df`, `du` | `ping`, `curl` |
| `rm`, `cp`, `mv` | `top`, `wc` | |
| `find`, `cat`, `grep` | `chmod`, `tar` | |
| `head`, `tail` | | |

---

## 🚀 Quick Start

### Colab에서 바로 실행
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gyunggyung/korean-linux-colab/blob/main/한글_linux.ipynb)

### 또는 수동 설정
```python
# 1. 저장소 클론
!git clone https://github.com/gyunggyung/korean-linux-colab.git
%cd korean-linux-colab

# 2. 사용
from src.korean_linux import 한글

한글("현재 폴더의 파일 목록을 보여줘")
# 🤖 명령어: ls -la
# 📁 결과: ...

한글("디스크 사용량을 확인해줘")
# 🤖 명령어: df -h
# 📁 결과: ...
```

---

## 📖 사용 예제

```python
# 파일 관리
한글("파일 목록 보여줘")           # ls
한글("data 폴더 만들어줘")         # mkdir data
한글("test.txt 내용 보여줘")       # cat test.txt

# 검색
한글("error가 포함된 줄 찾아줘")    # grep error
한글("txt 파일 찾아줘")            # find . -name "*.txt"

# 시스템
한글("디스크 사용량 알려줘")        # df -h
한글("실행 중인 프로세스 보여줘")   # ps aux
```

---

## 🔧 기술 스택

- **모델**: [HybriKo-117M-LinuxFC-SFT-v2](https://huggingface.co/Yaongi/HybriKo-117M-LinuxFC-SFT-v2)
- **아키텍처**: Griffin-style Hybrid (RNN + Attention)
- **정확도**: Action Name 100%

---

## ⚠️ 알려진 제한사항

| 항목 | 정확도 |
|------|--------|
| 명령어 선택 | ✅ 100% |
| 파라미터 | ⚠️ 가끔 오류 (자동 보정됨) |

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- HybriKo 모델 개발: [@gyunggyung](https://github.com/gyunggyung)
