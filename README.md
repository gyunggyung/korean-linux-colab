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

##  Quick Start

### Colab에서 바로 실행
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gyunggyung/korean-linux-colab/blob/main/한글_linux.ipynb)

### 사용 예시
```python
한글("현재 폴더에 뭐 있어?")      # ls -la
한글("test.txt 내용 보여줘")      # cat test.txt
한글("main.py 몇 줄이야?")        # wc -l main.py
한글("구글에 핑 날려봐")          # ping -c 4 google.com
```

---

## ✅ 잘 되는 명령어

| 작업 | 한국어 명령 | 결과 |
|------|------------|------|
| 파일 목록 | `현재 폴더에 뭐 있어?` | ✅ `ls -la .` |
| 파일 내용 | `test.txt 내용 보여줘` | ✅ `cat test.txt` |
| 줄 수 | `main.py 몇 줄이야?` | ✅ `wc -l main.py` |
| 폴더 용량 | `현재 폴더 용량 얼마야?` | ✅ `du -sh .` |
| 네트워크 | `구글에 핑 날려봐` | ✅ `ping -c 4 google.com` |
| 폴더 만들기 | `backup 폴더 만들어줘` | ✅ `mkdir -p backup` |
| 폴더 이동 | `examples 폴더로 이동해` | ✅ `cd examples` |

---

## ⚠️ 아직 불안정한 명령어

> **Note**: 117M 소형 모델의 한계로 일부 명령어는 파라미터 생성이 불안정합니다.

| 작업 | 한국어 명령 | 현재 상태 |
|------|------------|----------|
| 파일 끝부분 | `app.log 마지막 3줄 보여줘` | ⚠️ `tail` 대신 `cat` 사용 |
| 파일 검색 | `app.log에서 'ERROR' 찾아줘` | ⚠️ 패턴 추출 불안정 |
| 파일 찾기 | `현재 폴더에서 txt 파일 찾아줘` | ⚠️ `find` 대신 `grep` 사용 |
| 디스크 용량 | `디스크 남은 용량 보여줘` | ⚠️ `df` 인식 불안정 |
| 프로세스 | `실행 중인 프로세스 보여줘` | ⚠️ `ps` 인식 불안정 |
| 파일 복사 | `test.txt를 backup/으로 복사해` | ⚠️ 파라미터 파싱 실패 |

**개선 예정**: 더 큰 모델로 SFT를 진행하면 정확도가 향상될 예정입니다.

---

## 📋 지원 명령어 (21개)

| 파일 관리 | 시스템 | 네트워크 |
|-----------|--------|----------|
| `ls`, `cd`, `mkdir` | `ps`, `df`, `du` | `ping`, `curl` |
| `rm`, `cp`, `mv` | `top`, `wc` | |
| `find`, `cat`, `grep` | `chmod`, `tar` | |
| `head`, `tail` | | |

---

## 🔧 기술 스택

- **모델**: [HybriKo-117M-LinuxFC-SFT-v2](https://huggingface.co/Yaongi/HybriKo-117M-LinuxFC-SFT-v2)
- **아키텍처**: Griffin-style Hybrid (RNN + Attention)
- **파라미터**: 117.8M
- **Action Name 정확도**: 100% (21개 명령어)
- **Parameters 정확도**: 약 60% (일부 불안정)

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- HybriKo 모델 개발: [@gyunggyung](https://github.com/gyunggyung)
