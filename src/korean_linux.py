#!/usr/bin/env python3
"""
Korean Linux - 한국어로 리눅스 명령어 실행하기
Google Colab 전용 패키지 (v2 - 파라미터 보정 강화)
"""

import os
import sys
import subprocess
import re
import json
from difflib import SequenceMatcher

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
  {"name": "ls_command", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "options": {"type": "string"}}, "required": ["path"]}},
  {"name": "cd_command", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
  {"name": "mkdir_command", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
  {"name": "rm_command", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"]}},
  {"name": "cp_command", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}},
  {"name": "mv_command", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}},
  {"name": "find_command", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "name": {"type": "string"}}, "required": ["path", "name"]}},
  {"name": "cat_command", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
  {"name": "grep_command", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern", "path"]}},
  {"name": "head_command", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["path"]}},
  {"name": "tail_command", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["path"]}},
  {"name": "wc_command", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "options": {"type": "string"}}, "required": ["path"]}},
  {"name": "ps_command", "parameters": {"type": "object", "properties": {"options": {"type": "string"}}, "required": []}},
  {"name": "df_command", "parameters": {"type": "object", "properties": {"options": {"type": "string"}}, "required": []}},
  {"name": "du_command", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "options": {"type": "string"}}, "required": ["path"]}},
  {"name": "top_command", "parameters": {"type": "object", "properties": {"options": {"type": "string"}}, "required": []}},
  {"name": "ping_command", "parameters": {"type": "object", "properties": {"host": {"type": "string"}, "count": {"type": "integer"}}, "required": ["host"]}},
  {"name": "curl_command", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "options": {"type": "string"}}, "required": ["url"]}},
  {"name": "chmod_command", "parameters": {"type": "object", "properties": {"mode": {"type": "string"}, "path": {"type": "string"}}, "required": ["mode", "path"]}},
  {"name": "tar_command", "parameters": {"type": "object", "properties": {"operation": {"type": "string"}, "archive": {"type": "string"}, "files": {"type": "string"}}, "required": ["operation", "archive"]}},
  {"name": "Finish", "parameters": {"type": "object", "properties": {"return_type": {"type": "string"}, "final_answer": {"type": "string"}}, "required": ["return_type"]}}
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
    """모델 응답 파싱 - 강화된 버전"""
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    if "<|im" in response:
        response = response.split("<|im")[0]
    
    result = {"thought": None, "action": None, "params": None, "raw": response}
    
    # Thought 추출 - hallucination 제거
    thought_match = re.search(r"Thought:\s*(.+?)(?=\s*Action:|$)", response, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
        # hallucination 필터링: <|im 또는 이상한 텍스트 제거
        if "<|im" in thought:
            thought = thought.split("<|im")[0].strip()
        if len(thought) > 100:
            thought = thought[:100] + "..."
        result["thought"] = thought
    
    # Action 추출 - 정규 액션만 허용
    valid_actions = [
        "ls_command", "cd_command", "mkdir_command", "rm_command",
        "cp_command", "mv_command", "find_command", "cat_command",
        "grep_command", "head_command", "tail_command", "wc_command",
        "ps_command", "df_command", "du_command", "top_command",
        "ping_command", "curl_command", "chmod_command", "tar_command", "Finish"
    ]
    
    action_match = re.search(r"Action:\s*(\w+)", response)
    if action_match:
        action = action_match.group(1)
        if action in valid_actions:
            result["action"] = action
    
    # Action Input 추출
    input_match = re.search(r"Action Input:\s*(\{[^}]+\})", response, re.DOTALL)
    if input_match:
        try:
            result["params"] = json.loads(input_match.group(1))
        except:
            result["params"] = {}
    
    return result


def _extract_params_from_query(query: str, action: str) -> dict:
    """사용자 쿼리에서 파라미터 추출 (fallback)"""
    params = {}
    
    # 파일/폴더 패턴
    file_pattern = r'([^\s]+\.(txt|log|py|sh|csv|json|md|tar\.gz|tar|gz|zip))'
    folder_pattern = r'([a-zA-Z0-9_\-./]+(?:폴더|디렉토리)?)'
    
    file_match = re.search(file_pattern, query)
    folder_match = re.search(r'([a-zA-Z0-9_\-./]+)\s*(폴더|디렉토리)', query)
    
    # 액션별 파라미터 추출
    if action == "ls_command":
        params["path"] = "."
        if folder_match:
            params["path"] = folder_match.group(1)
    
    elif action == "cd_command":
        if folder_match:
            params["path"] = folder_match.group(1)
        elif "홈" in query:
            params["path"] = "~"
        elif ".." in query or "상위" in query:
            params["path"] = ".."
        else:
            # 가장 긴 경로 같은 문자열 추출
            path_match = re.search(r'([a-zA-Z0-9_\-./]+)', query)
            if path_match:
                params["path"] = path_match.group(1)
    
    elif action in ["cat_command", "head_command", "tail_command", "wc_command"]:
        if file_match:
            params["path"] = file_match.group(1)
    
    elif action == "grep_command":
        # 패턴 추출 (따옴표 안이나 영문 단어)
        pattern_match = re.search(r"['\"]([^'\"]+)['\"]|(\b[a-zA-Z]+\b)", query)
        if pattern_match:
            params["pattern"] = pattern_match.group(1) or pattern_match.group(2)
        if file_match:
            params["path"] = file_match.group(1)
    
    elif action == "find_command":
        params["path"] = "."
        if "txt" in query:
            params["name"] = "*.txt"
        elif "log" in query:
            params["name"] = "*.log"
        elif "py" in query:
            params["name"] = "*.py"
        else:
            params["name"] = "*"
    
    elif action == "mkdir_command":
        if folder_match:
            params["path"] = folder_match.group(1)
        else:
            name_match = re.search(r'([a-zA-Z0-9_\-]+)', query)
            if name_match:
                params["path"] = name_match.group(1)
    
    elif action == "rm_command":
        if file_match:
            params["path"] = file_match.group(1)
        elif folder_match:
            params["path"] = folder_match.group(1)
            params["recursive"] = True
    
    elif action == "ping_command":
        if "구글" in query or "google" in query.lower():
            params["host"] = "google.com"
        elif "네이버" in query or "naver" in query.lower():
            params["host"] = "naver.com"
        else:
            host_match = re.search(r'([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})', query)
            if host_match:
                params["host"] = host_match.group(1)
        params["count"] = 4
    
    elif action == "df_command":
        params["options"] = "-h"
    
    elif action == "du_command":
        params["path"] = "."
        params["options"] = "-sh"
    
    elif action == "ps_command":
        params["options"] = "aux"
    
    elif action == "top_command":
        params["options"] = "-b -n 1"
    
    return params


def _find_similar_file(filename: str) -> str:
    """현재 디렉토리에서 유사한 파일 찾기"""
    try:
        files = os.listdir(".")
        best_match = None
        best_ratio = 0.0
        
        for f in files:
            ratio = SequenceMatcher(None, filename.lower(), f.lower()).ratio()
            if ratio > best_ratio and ratio > 0.5:
                best_ratio = ratio
                best_match = f
        
        return best_match
    except:
        return None


def _correct_params(action: str, params: dict, query: str) -> dict:
    """파라미터 보정 - 모델 출력이 불완전할 때"""
    if params is None:
        params = {}
    
    # 쿼리에서 추출한 파라미터로 보완
    fallback_params = _extract_params_from_query(query, action)
    
    # 필요한 파라미터가 없으면 fallback 사용
    if action in ["cat_command", "head_command", "tail_command", "wc_command"]:
        if not params.get("path"):
            params["path"] = fallback_params.get("path", "")
        # 파일 존재 확인
        if params.get("path") and not os.path.exists(params["path"]):
            similar = _find_similar_file(params["path"])
            if similar:
                params["path"] = similar
    
    elif action == "cd_command":
        if not params.get("path"):
            params["path"] = fallback_params.get("path", ".")
    
    elif action == "ls_command":
        if not params.get("path"):
            params["path"] = fallback_params.get("path", ".")
    
    elif action == "grep_command":
        if not params.get("pattern"):
            params["pattern"] = fallback_params.get("pattern", "")
        if not params.get("path"):
            params["path"] = fallback_params.get("path", "")
    
    elif action == "find_command":
        if not params.get("path"):
            params["path"] = fallback_params.get("path", ".")
        if not params.get("name"):
            params["name"] = fallback_params.get("name", "*")
    
    elif action == "mkdir_command":
        if not params.get("path"):
            params["path"] = fallback_params.get("path", "")
    
    elif action == "ping_command":
        if not params.get("host"):
            params["host"] = fallback_params.get("host", "google.com")
        if not params.get("count"):
            params["count"] = 4
    
    elif action in ["df_command", "ps_command", "top_command", "du_command"]:
        params = {**fallback_params, **params}
    
    return params


def _build_command(action: str, params: dict) -> str:
    """액션과 파라미터로 실제 명령어 생성"""
    
    # Colab 특수 처리
    if action == "cd_command":
        return f"__CD__:{params.get('path', '.')}"
    
    if action == "top_command":
        return "top -b -n 1"
    
    if action == "ping_command":
        count = params.get("count", 4)
        host = params.get("host", "google.com")
        return f"ping -c {count} {host}"
    
    if action == "Finish":
        return f"__FINISH__:{params.get('final_answer', params.get('give_answer', ''))}"
    
    # 일반 명령어
    cmd_map = {
        "ls_command": lambda p: f"ls {p.get('options', '-la')} {p.get('path', '.')}".strip(),
        "mkdir_command": lambda p: f"mkdir -p {p.get('path', '')}",
        "rm_command": lambda p: f"rm {'-rf' if p.get('recursive') else ''} {p.get('path', '')}".strip(),
        "cp_command": lambda p: f"cp -r {p.get('source', '')} {p.get('destination', '')}",
        "mv_command": lambda p: f"mv {p.get('source', '')} {p.get('destination', '')}",
        "find_command": lambda p: f"find {p.get('path', '.')} -name '{p.get('name', '*')}'",
        "cat_command": lambda p: f"cat {p.get('options', '')} {p.get('path', '')}".strip(),
        "grep_command": lambda p: f"grep {p.get('options', '')} '{p.get('pattern', '')}' {p.get('path', '')}".strip(),
        "head_command": lambda p: f"head -n {p.get('lines', 10)} {p.get('path', '')}",
        "tail_command": lambda p: f"tail -n {p.get('lines', 10)} {p.get('path', '')}",
        "wc_command": lambda p: f"wc {p.get('options', '-l')} {p.get('path', '')}",
        "ps_command": lambda p: f"ps {p.get('options', 'aux')}",
        "df_command": lambda p: f"df {p.get('options', '-h')}",
        "du_command": lambda p: f"du {p.get('options', '-sh')} {p.get('path', '.')}",
        "curl_command": lambda p: f"curl {p.get('options', '')} {p.get('url', '')}".strip(),
        "chmod_command": lambda p: f"chmod {p.get('mode', '')} {p.get('path', '')}",
        "tar_command": lambda p: f"tar -czf {p.get('archive', '')} {p.get('files', '')}".strip() if p.get('operation') == 'create' else f"tar -xzf {p.get('archive', '')}",
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


def _infer_action_from_query(query: str) -> str:
    """쿼리에서 액션 추론 (모델이 실패했을 때)"""
    query_lower = query.lower()
    
    keywords = {
        "ls_command": ["파일 목록", "뭐 있", "ls", "폴더 내용", "디렉토리 내용", "파일 보여", "목록 보여"],
        "cd_command": ["이동", "폴더로", "디렉토리로", "가줘", "cd"],
        "cat_command": ["내용 보여", "내용 출력", "읽어", "cat", "보여줘"],
        "grep_command": ["찾아", "검색", "grep", "포함된"],
        "find_command": ["find", "파일 찾", "검색"],
        "mkdir_command": ["폴더 만들", "디렉토리 만들", "mkdir", "생성"],
        "rm_command": ["삭제", "지워", "rm", "제거"],
        "df_command": ["디스크", "용량", "df", "남은 공간"],
        "du_command": ["폴더 크기", "폴더 용량", "du"],
        "ps_command": ["프로세스", "실행 중", "ps"],
        "ping_command": ["핑", "ping", "네트워크"],
        "head_command": ["앞부분", "처음", "head"],
        "tail_command": ["뒷부분", "마지막", "끝", "tail"],
        "wc_command": ["줄 수", "라인 수", "몇 줄", "wc"],
        "top_command": ["시스템 상태", "top", "리소스"],
    }
    
    for action, kws in keywords.items():
        for kw in kws:
            if kw in query_lower:
                return action
    
    return None


def 한글(query: str, execute: bool = True, confirm_dangerous: bool = True) -> dict:
    """
    한국어로 리눅스 명령어 실행
    
    Args:
        query: 한국어 명령 (예: "현재 폴더에 뭐 있어?", "test.txt 내용 보여줘")
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
    
    # 액션이 없으면 쿼리에서 추론
    if not parsed["action"]:
        parsed["action"] = _infer_action_from_query(query)
    
    # 파라미터 보정
    parsed["params"] = _correct_params(parsed["action"], parsed["params"], query)
    
    # 명령어 생성
    cmd = _build_command(parsed["action"], parsed["params"]) if parsed["action"] else None
    
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
    if parsed["action"]:
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
    print("사용법: 한글('현재 폴더에 뭐 있어?')")
    print("="*50 + "\n")
