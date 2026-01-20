#!/usr/bin/env python3
"""
Korean Linux - 한국어로 리눅스 명령어 실행하기
Google Colab 전용 패키지 (v3 - 강력한 보정 로직)
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
        return
    
    print("🔧 Korean Linux 초기화 중...")
    
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
    
    repo_id = "Yaongi/HybriKo-117M-LinuxFC-SFT-v2"
    files = ["configuration_hybridko.py", "modeling_hybridko.py", 
             "pytorch_model.pt", "HybriKo_tok.model"]
    
    download_dir = "/content/korean_linux_model"
    os.makedirs(download_dir, exist_ok=True)
    
    for f in files:
        if not os.path.exists(os.path.join(download_dir, f)):
            print(f"  📥 {f} 다운로드 중...")
            hf_hub_download(repo_id, f, local_dir=download_dir)
    
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
            
            new_tokens = generated[0, prompt_len:].tolist()
            new_text = _tokenizer.DecodeIds(new_tokens)
            
            if "<|im_end|>" in new_text:
                break
            
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
        response = response.split("<|im_end|>")[0]
    if "<|im" in response:
        response = response.split("<|im")[0]
    
    result = {"thought": None, "action": None, "params": None, "raw": response}
    
    thought_match = re.search(r"Thought:\s*(.+?)(?=\s*Action:|$)", response, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
        if "<|im" in thought:
            thought = thought.split("<|im")[0].strip()
        if len(thought) > 100:
            thought = thought[:100] + "..."
        result["thought"] = thought
    
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
    
    input_match = re.search(r"Action Input:\s*(\{[^}]+\})", response, re.DOTALL)
    if input_match:
        try:
            result["params"] = json.loads(input_match.group(1))
        except:
            result["params"] = {}
    
    return result


def _infer_action_from_query(query: str) -> str:
    """쿼리에서 액션 추론 - 강화된 키워드 매칭"""
    q = query.lower()
    
    # 우선순위 높은 키워드 (먼저 확인)
    # tail (마지막, 끝, 뒷부분)
    if any(kw in q for kw in ["마지막", "끝", "뒷부분", "tail"]) and any(kw in q for kw in ["줄", "보여", "봐"]):
        return "tail_command"
    
    # head (처음, 앞부분, 첫)
    if any(kw in q for kw in ["처음", "앞부분", "첫", "head"]) and any(kw in q for kw in ["줄", "보여", "봐"]):
        return "head_command"
    
    # grep (찾아, 검색, 에서 ... 찾)
    if "에서" in q and any(kw in q for kw in ["찾아", "검색"]):
        return "grep_command"
    
    # find (파일 찾, 폴더에서 찾)
    if any(kw in q for kw in ["파일 찾", "폴더에서"]) and "찾" in q:
        return "find_command"
    
    # df (디스크, 남은 용량, df)
    if any(kw in q for kw in ["디스크", "남은 용량", "남은 공간"]):
        return "df_command"
    
    # du (폴더 용량, 폴더 크기, 현재 폴더 용량)
    if any(kw in q for kw in ["폴더 용량", "폴더 크기", "현재 폴더 용량"]):
        return "du_command"
    
    # ps (프로세스, 실행 중)
    if any(kw in q for kw in ["프로세스", "실행 중"]):
        return "ps_command"
    
    # ping
    if any(kw in q for kw in ["핑", "ping"]):
        return "ping_command"
    
    # cp (복사)
    if any(kw in q for kw in ["복사", "copy", "cp "]):
        return "cp_command"
    
    # mv (이동, 옮겨, 이름 바꿔)
    if any(kw in q for kw in ["이름 바꿔", "rename"]) or ("옮겨" in q and "폴더" not in q):
        return "mv_command"
    
    # cd (이동, 폴더로, 가줘)
    if any(kw in q for kw in ["폴더로 이동", "디렉토리로 이동", "폴더로 가", "가줘"]):
        return "cd_command"
    
    # mkdir (폴더 만들, 디렉토리 만들)
    if any(kw in q for kw in ["폴더 만들", "디렉토리 만들", "mkdir"]):
        return "mkdir_command"
    
    # rm (삭제, 지워)
    if any(kw in q for kw in ["삭제", "지워", "rm "]):
        return "rm_command"
    
    # wc (몇 줄, 줄 수, 라인 수)
    if any(kw in q for kw in ["몇 줄", "줄 수", "라인 수"]):
        return "wc_command"
    
    # cat (내용 보여, 읽어)
    if any(kw in q for kw in ["내용 보여", "내용 출력", "읽어"]):
        return "cat_command"
    
    # ls (파일 목록, 뭐 있, 폴더 내용, 상세 정보)
    if any(kw in q for kw in ["파일 목록", "뭐 있", "폴더 내용", "상세 정보", "ls"]):
        return "ls_command"
    
    # top
    if any(kw in q for kw in ["시스템 상태", "리소스", "top"]):
        return "top_command"
    
    return None


def _extract_file_from_query(query: str) -> str:
    """쿼리에서 파일명 추출"""
    # 파일 확장자 패턴
    file_match = re.search(r'([a-zA-Z0-9_\-./]+\.(txt|log|py|sh|csv|json|md|tar\.gz|tar|gz|zip|yaml|yml))', query)
    if file_match:
        return file_match.group(1)
    return None


def _extract_folder_from_query(query: str) -> str:
    """쿼리에서 폴더명 추출"""
    # "XXX 폴더" 패턴
    folder_match = re.search(r'([a-zA-Z0-9_\-./]+)\s*(폴더|디렉토리)', query)
    if folder_match:
        return folder_match.group(1)
    return None


def _extract_pattern_from_query(query: str) -> str:
    """쿼리에서 검색 패턴 추출 (따옴표 안 우선)"""
    # 작은따옴표 안의 내용
    sq_match = re.search(r"'([^']+)'", query)
    if sq_match:
        return sq_match.group(1)
    
    # 큰따옴표 안의 내용
    dq_match = re.search(r'"([^"]+)"', query)
    if dq_match:
        return dq_match.group(1)
    
    # "XXX가 포함된" 패턴
    include_match = re.search(r'(\w+)가?\s*(포함된|있는|들어간)', query)
    if include_match:
        return include_match.group(1)
    
    return None


def _extract_lines_from_query(query: str) -> int:
    """쿼리에서 줄 수 추출"""
    # "N줄" 패턴
    line_match = re.search(r'(\d+)\s*줄', query)
    if line_match:
        return int(line_match.group(1))
    return 10  # 기본값


def _extract_cp_params_from_query(query: str) -> dict:
    """cp 명령어용 소스/목적지 추출"""
    # "XXX를 YYY로 복사" 패턴
    cp_match = re.search(r'([a-zA-Z0-9_\-./]+)\s*를?\s*(backup/|[a-zA-Z0-9_\-./]+/?)\s*(로|으로)?\s*복사', query)
    if cp_match:
        return {"source": cp_match.group(1), "destination": cp_match.group(2)}
    
    # 파일명만 추출
    file = _extract_file_from_query(query)
    if file:
        # 목적지 폴더 찾기
        dest_match = re.search(r'(backup|[a-zA-Z0-9_\-]+)/?', query)
        if dest_match and dest_match.group(1) != file.split('.')[0]:
            return {"source": file, "destination": dest_match.group(1) + "/"}
    
    return {}


def _correct_action(action: str, query: str) -> str:
    """모델이 반환한 액션이 잘못됐을 때 보정"""
    inferred = _infer_action_from_query(query)
    
    # 모델이 cat을 반환했지만 실제로는 다른 명령어여야 하는 경우
    if action == "cat_command":
        if inferred in ["df_command", "ps_command", "tail_command", "head_command", "ls_command"]:
            return inferred
    
    # 모델이 grep을 반환했지만 실제로는 find여야 하는 경우
    if action == "grep_command":
        if inferred == "find_command":
            return inferred
    
    # 액션이 None이면 추론한 것 사용
    if action is None:
        return inferred
    
    return action


def _correct_params(action: str, params: dict, query: str) -> dict:
    """파라미터 보정 - 강화된 버전"""
    if params is None:
        params = {}
    
    # 액션별 파라미터 보정
    if action == "cat_command":
        if not params.get("path"):
            params["path"] = _extract_file_from_query(query) or ""
    
    elif action == "head_command":
        if not params.get("path"):
            params["path"] = _extract_file_from_query(query) or ""
        if not params.get("lines"):
            params["lines"] = _extract_lines_from_query(query)
    
    elif action == "tail_command":
        if not params.get("path"):
            params["path"] = _extract_file_from_query(query) or ""
        if not params.get("lines"):
            params["lines"] = _extract_lines_from_query(query)
    
    elif action == "grep_command":
        # 패턴 추출 (따옴표 안 우선)
        if not params.get("pattern") or params.get("pattern") == "app":
            extracted = _extract_pattern_from_query(query)
            if extracted:
                params["pattern"] = extracted
        # 파일 추출
        if not params.get("path"):
            params["path"] = _extract_file_from_query(query) or ""
    
    elif action == "find_command":
        params["path"] = "."
        # 확장자 추출
        if "txt" in query:
            params["name"] = "*.txt"
        elif "log" in query:
            params["name"] = "*.log"
        elif "py" in query:
            params["name"] = "*.py"
        else:
            params["name"] = "*"
    
    elif action == "cd_command":
        if not params.get("path"):
            folder = _extract_folder_from_query(query)
            if folder:
                params["path"] = folder
            elif "홈" in query:
                params["path"] = "~"
            elif ".." in query or "상위" in query:
                params["path"] = ".."
    
    elif action == "ls_command":
        if not params.get("path"):
            folder = _extract_folder_from_query(query)
            params["path"] = folder or "."
    
    elif action == "mkdir_command":
        if not params.get("path"):
            folder = _extract_folder_from_query(query)
            if folder:
                params["path"] = folder
    
    elif action == "rm_command":
        if not params.get("path"):
            file = _extract_file_from_query(query)
            folder = _extract_folder_from_query(query)
            params["path"] = file or folder or ""
            if folder and not file:
                params["recursive"] = True
    
    elif action == "cp_command":
        cp_params = _extract_cp_params_from_query(query)
        if cp_params:
            params.update(cp_params)
    
    elif action == "wc_command":
        if not params.get("path"):
            params["path"] = _extract_file_from_query(query) or ""
    
    elif action == "ping_command":
        if not params.get("host"):
            if "구글" in query or "google" in query.lower():
                params["host"] = "google.com"
            elif "네이버" in query or "naver" in query.lower():
                params["host"] = "naver.com"
            else:
                host_match = re.search(r'([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})', query)
                if host_match:
                    params["host"] = host_match.group(1)
                else:
                    params["host"] = "google.com"
        if not params.get("count"):
            params["count"] = 4
    
    elif action == "df_command":
        params["options"] = "-h"
    
    elif action == "du_command":
        params["path"] = params.get("path", ".")
        params["options"] = "-sh"
    
    elif action == "ps_command":
        params["options"] = "aux"
    
    elif action == "top_command":
        params["options"] = "-b -n 1"
    
    return params


def _build_command(action: str, params: dict) -> str:
    """액션과 파라미터로 실제 명령어 생성"""
    
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
    
    cmd_map = {
        "ls_command": lambda p: f"ls {p.get('options', '-la')} {p.get('path', '.')}".strip(),
        "mkdir_command": lambda p: f"mkdir -p {p.get('path', '')}",
        "rm_command": lambda p: f"rm {'-rf' if p.get('recursive') else ''} {p.get('path', '')}".strip(),
        "cp_command": lambda p: f"cp -r {p.get('source', '')} {p.get('destination', '')}",
        "mv_command": lambda p: f"mv {p.get('source', '')} {p.get('destination', '')}",
        "find_command": lambda p: f"find {p.get('path', '.')} -name '{p.get('name', '*')}'",
        "cat_command": lambda p: f"cat {p.get('path', '')}".strip(),
        "grep_command": lambda p: f"grep '{p.get('pattern', '')}' {p.get('path', '')}".strip(),
        "head_command": lambda p: f"head -n {p.get('lines', 10)} {p.get('path', '')}",
        "tail_command": lambda p: f"tail -n {p.get('lines', 10)} {p.get('path', '')}",
        "wc_command": lambda p: f"wc -l {p.get('path', '')}",
        "ps_command": lambda p: f"ps {p.get('options', 'aux')}",
        "df_command": lambda p: f"df {p.get('options', '-h')}",
        "du_command": lambda p: f"du {p.get('options', '-sh')} {p.get('path', '.')}",
        "curl_command": lambda p: f"curl {p.get('options', '')} {p.get('url', '')}".strip(),
        "chmod_command": lambda p: f"chmod {p.get('mode', '')} {p.get('path', '')}",
        "tar_command": lambda p: f"tar -xzf {p.get('archive', '')}" if "풀" in str(p) else f"tar -czf {p.get('archive', '')} {p.get('files', '')}".strip(),
    }
    
    if action in cmd_map:
        return cmd_map[action](params or {})
    
    return None


def _execute_command(cmd: str) -> str:
    """명령어 실행"""
    
    if cmd.startswith("__CD__:"):
        path = cmd[7:]
        try:
            os.chdir(path)
            return f"현재 디렉토리: {os.getcwd()}"
        except Exception as e:
            return f"오류: {e}"
    
    if cmd.startswith("__FINISH__:"):
        return cmd[11:]
    
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
        query: 한국어 명령
        execute: True면 명령어 실행
        confirm_dangerous: True면 위험 명령어 확인
    
    Returns:
        dict: {"command": str, "result": str, "action": str}
    """
    if _model is None:
        setup()
    
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    
    response = _generate(prompt)
    parsed = _parse_response(response)
    
    # 액션 보정 (모델이 잘못 반환했을 때)
    parsed["action"] = _correct_action(parsed["action"], query)
    
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
    
    print(f"\n🗣️ 입력: {query}")
    if parsed["thought"]:
        print(f"💭 생각: {parsed['thought']}")
    if parsed["action"]:
        print(f"🔧 액션: {parsed['action']}")
    if cmd and not cmd.startswith("__"):
        print(f"🤖 명령어: {cmd}")
    
    if confirm_dangerous and parsed["action"] == "rm_command":
        if parsed["params"] and parsed["params"].get("recursive"):
            print("⚠️  경고: 재귀 삭제 명령입니다!")
            confirm = input("계속하시겠습니까? (y/N): ")
            if confirm.lower() != 'y':
                result_dict["result"] = "사용자가 취소했습니다."
                print(f"📁 결과: {result_dict['result']}")
                return result_dict
    
    if execute and cmd:
        result_dict["result"] = _execute_command(cmd)
        print(f"📁 결과:\n{result_dict['result']}")
    
    print()
    return result_dict


linux = 한글
ㅎㄱ = 한글


if __name__ == "__main__":
    setup()
    print("\n" + "="*50)
    print("Korean Linux v3 준비 완료!")
    print("사용법: 한글('현재 폴더에 뭐 있어?')")
    print("="*50 + "\n")
