#!/usr/bin/env python3
"""
Korean Linux - 한국어로 리눅스 명령어 실행하기
Google Colab 전용 패키지
"""

import os
import sys
import subprocess
import re
import json

# 전역 변수
_model = None
_tokenizer = None
_device = None

# ============================================================
# 시스템 프롬프트 (학습 시 사용한 것과 동일)
# ============================================================
SYSTEM_PROMPT = """You are a Linux command assistant. You can use many tools (functions) to help users with their Linux tasks.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually execute your step. Your output should follow this format:
Thought:
Action
Action Input:

After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your final answer.

Remember:
1. The state change is irreversible, you can't go back to one of the former state.
2. All the thought is short, at most in 5 sentences.
3. ALWAYS call "Finish" function at the end of the task.
4. If you cannot handle the task with the available tools, say you don't know and call Finish with give_answer.

You have access of the following tools:
[
  {"name": "ls_command", "description": "List directory contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "options": {"type": "string"}}, "required": ["path"]}},
  {"name": "cd_command", "description": "Change the current working directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
  {"name": "mkdir_command", "description": "Create a new directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
  {"name": "rm_command", "description": "Remove files or directories.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"]}},
  {"name": "cp_command", "description": "Copy files or directories.", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}},
  {"name": "mv_command", "description": "Move or rename files.", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}},
  {"name": "find_command", "description": "Find files by name pattern.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "name": {"type": "string"}}, "required": ["path", "name"]}},
  {"name": "cat_command", "description": "Display file contents.", "parameters": {"type": "object", "properties": {"file": {"type": "string"}}, "required": ["file"]}},
  {"name": "grep_command", "description": "Search for patterns in files.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "file": {"type": "string"}}, "required": ["pattern", "file"]}},
  {"name": "head_command", "description": "Display first lines of a file.", "parameters": {"type": "object", "properties": {"file": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["file"]}},
  {"name": "tail_command", "description": "Display last lines of a file.", "parameters": {"type": "object", "properties": {"file": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["file"]}},
  {"name": "wc_command", "description": "Count lines, words, and bytes.", "parameters": {"type": "object", "properties": {"file": {"type": "string"}}, "required": ["file"]}},
  {"name": "ps_command", "description": "Display running processes.", "parameters": {"type": "object", "properties": {"options": {"type": "string"}}, "required": []}},
  {"name": "df_command", "description": "Display disk space usage.", "parameters": {"type": "object", "properties": {"options": {"type": "string"}}, "required": []}},
  {"name": "du_command", "description": "Display directory space usage.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "options": {"type": "string"}}, "required": ["path"]}},
  {"name": "top_command", "description": "Display system processes in real-time.", "parameters": {"type": "object", "properties": {}, "required": []}},
  {"name": "ping_command", "description": "Test network connectivity.", "parameters": {"type": "object", "properties": {"host": {"type": "string"}, "count": {"type": "integer"}}, "required": ["host"]}},
  {"name": "curl_command", "description": "Transfer data from URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "options": {"type": "string"}}, "required": ["url"]}},
  {"name": "chmod_command", "description": "Change file permissions.", "parameters": {"type": "object", "properties": {"mode": {"type": "string"}, "file": {"type": "string"}}, "required": ["mode", "file"]}},
  {"name": "tar_command", "description": "Archive or extract files.", "parameters": {"type": "object", "properties": {"options": {"type": "string"}, "archive": {"type": "string"}, "files": {"type": "string"}}, "required": ["options", "archive"]}},
  {"name": "Finish", "description": "Complete the task.", "parameters": {"type": "object", "properties": {"give_answer": {"type": "string"}}, "required": ["give_answer"]}}
]"""


def setup():
    """모델 및 토크나이저 로딩"""
    global _model, _tokenizer, _device
    
    if _model is not None:
        return  # 이미 로딩됨
    
    print("🔧 Korean Linux 초기화 중...")
    
    # 필요한 패키지 설치
    try:
        import torch
        import sentencepiece as spm
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("📦 필요한 패키지 설치 중...")
        os.system("pip install -q torch sentencepiece huggingface_hub")
        import torch
        import sentencepiece as spm
        from huggingface_hub import hf_hub_download
    
    # 파일 다운로드
    repo_id = "Yaongi/HybriKo-117M-LinuxFC-SFT-v2"
    files = ["configuration_hybridko.py", "modeling_hybridko.py", 
             "pytorch_model.pt", "HybriKo_tok.model"]
    
    download_dir = "/content/korean_linux_model"
    os.makedirs(download_dir, exist_ok=True)
    
    for f in files:
        if not os.path.exists(os.path.join(download_dir, f)):
            print(f"  📥 {f} 다운로드 중...")
            hf_hub_download(repo_id, f, local_dir=download_dir)
    
    # 모델 로딩
    sys.path.insert(0, download_dir)
    from configuration_hybridko import HybriKoConfig
    from modeling_hybridko import HybriKoModel
    
    print("🤖 모델 로딩 중...")
    _tokenizer = spm.SentencePieceProcessor()
    _tokenizer.Load(os.path.join(download_dir, "HybriKo_tok.model"))
    
    config = HybriKoConfig(
        d_model=768, n_layers=12, vocab_size=32000,
        n_heads=12, n_kv_heads=3, ff_mult=3, max_seq_len=6144
    )
    _model = HybriKoModel(config)
    checkpoint = torch.load(
        os.path.join(download_dir, "pytorch_model.pt"),
        map_location="cpu", weights_only=False
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device).eval()
    
    print(f"✅ 초기화 완료! (Device: {_device})")


def _generate(prompt: str, max_new_tokens: int = 150) -> str:
    """모델로 텍스트 생성"""
    import torch
    
    input_ids = _tokenizer.EncodeAsIds(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=_device)
    prompt_len = len(input_ids)
    
    with torch.no_grad():
        generated = input_tensor
        for _ in range(max_new_tokens):
            outputs = _model(generated)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # 종료 조건 확인
            new_tokens = generated[0, prompt_len:].tolist()
            new_text = _tokenizer.DecodeIds(new_tokens)
            
            if "<|im_end|>" in new_text:
                break
            
            # Action Input JSON 완료 확인
            if "Action Input:" in new_text:
                ai_idx = new_text.find("Action Input:")
                after_ai = new_text[ai_idx + 13:].strip()
                if after_ai.startswith("{"):
                    brace_count = 0
                    for c in after_ai:
                        if c == "{": brace_count += 1
                        elif c == "}": brace_count -= 1
                        if brace_count == 0:
                            return new_text
    
    new_tokens = generated[0, prompt_len:].tolist()
    return _tokenizer.DecodeIds(new_tokens)


def _parse_response(response: str) -> dict:
    """모델 응답 파싱"""
    if "<|im_end|>" in response:
        response = response.split("<|im_end|}")[0]
    
    result = {"thought": None, "action": None, "params": None}
    
    # Thought 추출
    thought_match = re.search(r"Thought:\s*(.+?)(?=\s*Action:|$)", response, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()
    
    # Action 추출
    action_match = re.search(r"Action:\s*(\w+)", response)
    if action_match:
        result["action"] = action_match.group(1)
    
    # Action Input 추출
    input_match = re.search(r"Action Input:\s*(\{[^}]+\})", response, re.DOTALL)
    if input_match:
        try:
            result["params"] = json.loads(input_match.group(1))
        except:
            result["params"] = {}
    
    return result


def _build_command(action: str, params: dict) -> str:
    """액션과 파라미터로 실제 명령어 생성 (보정 포함)"""
    
    # Colab 특수 처리
    if action == "cd_command":
        return f"__CD__:{params.get('path', '.')}"  # 특수 마커
    
    if action == "top_command":
        return "top -b -n 1"  # interactive 모드 불가
    
    if action == "ping_command":
        count = params.get("count", 4)
        host = params.get("host", "")
        return f"ping -c {count} {host}"
    
    if action == "Finish":
        return f"__FINISH__:{params.get('give_answer', '')}"
    
    # 일반 명령어
    cmd_map = {
        "ls_command": lambda p: f"ls {p.get('options', '-la')} {p.get('path', '.')}",
        "mkdir_command": lambda p: f"mkdir -p {p.get('path', '')}",
        "rm_command": lambda p: f"rm {'-rf' if p.get('recursive') else ''} {p.get('path', '')}",
        "cp_command": lambda p: f"cp {p.get('source', '')} {p.get('destination', '')}",
        "mv_command": lambda p: f"mv {p.get('source', '')} {p.get('destination', '')}",
        "find_command": lambda p: f"find {p.get('path', '.')} -name '{p.get('name', '*')}'",
        "cat_command": lambda p: f"cat {p.get('file', '')}",
        "grep_command": lambda p: f"grep '{p.get('pattern', '')}' {p.get('file', '')}",
        "head_command": lambda p: f"head -n {p.get('lines', 10)} {p.get('file', '')}",
        "tail_command": lambda p: f"tail -n {p.get('lines', 10)} {p.get('file', '')}",
        "wc_command": lambda p: f"wc {p.get('file', '')}",
        "ps_command": lambda p: f"ps {p.get('options', 'aux')}",
        "df_command": lambda p: f"df {p.get('options', '-h')}",
        "du_command": lambda p: f"du {p.get('options', '-sh')} {p.get('path', '.')}",
        "curl_command": lambda p: f"curl {p.get('options', '')} {p.get('url', '')}",
        "chmod_command": lambda p: f"chmod {p.get('mode', '')} {p.get('file', '')}",
        "tar_command": lambda p: f"tar {p.get('options', '')} {p.get('archive', '')} {p.get('files', '')}",
    }
    
    if action in cmd_map:
        return cmd_map[action](params or {})
    
    return None


def _execute_command(cmd: str) -> str:
    """명령어 실행"""
    
    # cd 특수 처리
    if cmd.startswith("__CD__:"):
        path = cmd[7:]
        try:
            os.chdir(path)
            return f"현재 디렉토리: {os.getcwd()}"
        except Exception as e:
            return f"오류: {e}"
    
    # Finish 처리
    if cmd.startswith("__FINISH__:"):
        return cmd[11:]
    
    # 일반 명령어 실행
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout or result.stderr or "(출력 없음)"
        return output.strip()
    except subprocess.TimeoutExpired:
        return "⏰ 시간 초과 (30초)"
    except Exception as e:
        return f"오류: {e}"


def 한글(query: str, execute: bool = True, confirm_dangerous: bool = True) -> dict:
    """
    한국어로 리눅스 명령어 실행
    
    Args:
        query: 한국어 명령 (예: "현재 폴더의 파일 목록을 보여줘")
        execute: True면 명령어 실행, False면 변환만
        confirm_dangerous: True면 위험 명령어 확인 요청
    
    Returns:
        dict: {"command": str, "result": str, "action": str, "thought": str}
    """
    # 초기화 확인
    if _model is None:
        setup()
    
    # 프롬프트 생성
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    
    # 생성 및 파싱
    response = _generate(prompt)
    parsed = _parse_response(response)
    
    # 명령어 생성
    cmd = _build_command(parsed["action"], parsed["params"])
    
    result_dict = {
        "command": cmd,
        "result": None,
        "action": parsed["action"],
        "thought": parsed["thought"],
        "params": parsed["params"]
    }
    
    # 출력
    print(f"\n🗣️ 입력: {query}")
    if parsed["thought"]:
        print(f"💭 생각: {parsed['thought']}")
    print(f"🔧 액션: {parsed['action']}")
    if cmd and not cmd.startswith("__"):
        print(f"🤖 명령어: {cmd}")
    
    # 위험 명령어 확인
    if confirm_dangerous and parsed["action"] == "rm_command":
        if parsed["params"] and parsed["params"].get("recursive"):
            print("⚠️  경고: 재귀 삭제 명령입니다!")
            confirm = input("계속하시겠습니까? (y/N): ")
            if confirm.lower() != 'y':
                result_dict["result"] = "사용자가 취소했습니다."
                print(f"📁 결과: {result_dict['result']}")
                return result_dict
    
    # 실행
    if execute and cmd:
        result_dict["result"] = _execute_command(cmd)
        print(f"📁 결과:\n{result_dict['result']}")
    
    print()
    return result_dict


# 별칭
linux = 한글
ㅎㄱ = 한글


if __name__ == "__main__":
    setup()
    print("\n" + "="*50)
    print("Korean Linux 준비 완료!")
    print("사용법: 한글('파일 목록 보여줘')")
    print("="*50 + "\n")
