"""
Tiny MoA 오케스트레이터
======================
Brain과 Specialist를 조율하여 사용자 요청 처리
Tool Calling 지원 추가
"""

import re
import sys
import threading
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging

from tiny_moa.brain import Brain
from tiny_moa.reasoner import Reasoner

# 번역 모듈 import
try:
    from translation.detector import detect_language
    from translation.pipeline import TranslationPipeline
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

console = Console(force_terminal=True, color_system="auto")


class TinyMoA:
    """Tiny MoA (Mixture of Agents) 오케스트레이터"""
    
    def __init__(
        self,
        brain_path: Optional[str] = None,
        reasoner_path: Optional[str] = None,
        tool_caller_path: Optional[str] = None,
        n_ctx: int = 4096,
        use_thinking: bool = False,
        show_thinking: bool = False,
        lazy_load: bool = True,
        enable_tools: bool = True,
        enable_translation: bool = True,
    ):
        """
        Args:
            brain_path: Brain 모델 경로
            reasoner_path: Reasoner 모델 경로
            tool_caller_path: Tool Caller (Falcon-90M) 경로
            n_ctx: 컨텍스트 길이
            use_thinking: LFM Thinking 모델 사용 여부 (실험 중)
            lazy_load: Reasoner/ToolCaller를 첫 사용 시 로드할지 여부
            enable_tools: Tool Calling 기능 활성화 여부
        """
        self.brain_path = brain_path
        self.reasoner_path = reasoner_path
        self.tool_caller_path = tool_caller_path
        self.n_ctx = n_ctx
        self.use_thinking = use_thinking
        self.show_thinking = show_thinking
        self.lazy_load = lazy_load
        self.enable_tools = enable_tools
        self.enable_translation = enable_translation and TRANSLATION_AVAILABLE
        
        # 번역 파이프라인 초기화
        self._translation_pipeline = None
        if self.enable_translation:
            try:
                self._translation_pipeline = TranslationPipeline(use_simple_translator=True)
                console.print("[dim]🌐 Translation Pipeline 활성화[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠️ 번역 비활성화: {e}[/yellow]")
                self.enable_translation = False
        
        self._brain: Optional[Brain] = None
        self._reasoner: Optional[Reasoner] = None
        self._tool_caller = None
        self._tool_executor = None
        self.dashboard = None
        self._model_lock = threading.Lock()
        
        console.print("[bold blue]🤖 Tiny MoA 초기화 중...[/bold blue]")
        
        # Brain은 항상 로드 (라우터 역할)
        self._load_brain()
        
        # Reasoner/ToolCaller는 lazy_load 설정에 따라
        if not lazy_load:
            self._load_reasoner()
            if enable_tools:
                self._load_tool_caller()
        
        console.print("[bold green]✅ Tiny MoA 준비 완료![/bold green]")
    
    def _load_brain(self):
        """Brain 모델 로드"""
        if self._brain is None:
            console.print("[dim]Loading Brain (LFM2.5-1.2B)...[/dim]")
            self._brain = Brain(
                model_path=self.brain_path,
                n_ctx=self.n_ctx,
                use_thinking=self.use_thinking,
            )
    
    def _load_reasoner(self):
        """Reasoner 모델 로드 (Lazy)"""
        if self._reasoner is None:
            console.print("[dim]Loading Reasoner (Falcon-R-0.6B)...[/dim]")
            self._reasoner = Reasoner(
                model_path=self.reasoner_path,
                n_ctx=self.n_ctx,
            )
    
    def _load_tool_caller(self):
        """Tool Caller 로드 (Lazy)"""
        if self._tool_caller is None and self.enable_tools:
            try:
                from tools.caller import ToolCaller
                from tools.executor import ToolExecutor
                
                console.print("[dim]Loading Tool Caller (Falcon-90M)...[/dim]")
                self._tool_caller = ToolCaller(
                    falcon_path=self.tool_caller_path,
                    brain_model=self._brain,  # Brain으로 JSON 보정
                )
                self._tool_executor = ToolExecutor()
                console.print("[dim]✅ Tool Caller 준비 완료[/dim]")
            except ImportError as e:
                console.print(f"[yellow]⚠️ Tool Calling 비활성화: {e}[/yellow]")
                self.enable_tools = False
    
    @property
    def brain(self) -> Brain:
        if self._brain is None:
            self._load_brain()
        return self._brain
    
    @property
    def reasoner(self) -> Reasoner:
        if self._reasoner is None:
            self._load_reasoner()
        return self._reasoner
    
    @property
    def tool_caller(self):
        if self._tool_caller is None:
            self._load_tool_caller()
        return self._tool_caller
    
    @property
    def tool_executor(self):
        if self._tool_executor is None:
            self._load_tool_caller()
        return self._tool_executor
    
    def _handle_tool_call(self, user_input: str, tool_hint: str = "", arg_hint: str = "", verbose: bool = True, return_raw: bool = False) -> str:
        """
        Tool 호출 처리
        
        Args:
            return_raw: True일 경우 Brain 통합 없이 raw result(dict or string) 반환
        """
        if not self.enable_tools or self.tool_executor is None:
            return self.brain.direct_respond(
                user_input,
                system_prompt="The user is asking about real-time information but tools are not available. Apologize and explain."
            )
        
        tool_call = {}
        
        # [NEW] Explicit Tool Handoff from Planner
        # If input is like "execute_command: python --version", parse it directly.
        explicit_match = re.match(r"^([a-zA-Z_]+):\s*(.+)$", user_input.strip())
        if explicit_match:
            ex_tool = explicit_match.group(1).lower()
            ex_arg = explicit_match.group(2).strip()
            
            # Allow known tools only
            known_tools = ["execute_command", "get_weather", "search_web", "search_news", "read_url"]
            if ex_tool in known_tools:
                if verbose: console.print(f"[dim]⚡ Explicit Tool Handoff: {ex_tool}({ex_arg})[/dim]")
                
                if ex_tool == "execute_command":
                    tool_call = {"name": "execute_command", "arguments": {"command": ex_arg}}
                elif ex_tool == "get_weather":
                    tool_call = {"name": "get_weather", "arguments": {"location": ex_arg}}
                elif ex_tool in ["search_web", "search_news"]:
                    tool_call = {"name": "search_web", "arguments": {"query": ex_arg}}
                elif ex_tool == "read_url":
                    tool_call = {"name": "read_url", "arguments": {"url": ex_arg}}
                
                # If explicit, we skip Brain hint logic below
                return self._execute_tool_logic(tool_call, verbose, return_raw, user_input)

        # 1. Brain이 제공한 최적화 인자 사용 (우선순위 1)
        if arg_hint and tool_hint:
            if verbose:
                console.print(f"[dim]🧠 Brain 최적화 인자 사용: {tool_hint}({arg_hint})[/dim]")
            
            arguments = {}
            if tool_hint in ["search_web", "search_news", "search_wikipedia"]:
                arguments = {"query": arg_hint}
            elif tool_hint == "execute_command":
                # 방어 로직: 명령어가 자연어 문장으로 보이면 무시하고 키워드 폴백 사용
                # LFM 1.2B가 가끔 "Check if..." 같은 지시문을 생성함
                is_valid_cmd = True
                bad_starters = ["Check", "Verify", "Confirm", "Please", "Ensure", "See", "Test", "Determine"]
                
                # 1. 자연어 시작 패턴 체크
                if any(arg_hint.strip().startswith(s) for s in bad_starters) and len(arg_hint.split()) > 2:
                    is_valid_cmd = False
                
                # 2. 한글 포함 여부 체크 (명령어에 한글이 있으면 자연어 설명일 확률 높음)
                if re.search(r'[가-힣]', arg_hint):
                    is_valid_cmd = False
                
                if is_valid_cmd:
                    # [Defense] Command validation: simple keyword check
                    # "main idea of paper" treated as command -> FAIL
                    # known safe prefixes
                    safe_prefixes = ["python", "uv", "pip", "git", "docker", "ls", "dir", "cat", "echo", "where", "which", "node", "npm"]
                    
                    cmd_clean = arg_hint.strip().lower()
                    is_safe = any(cmd_clean.startswith(p) for p in safe_prefixes)
                    
                    # If not starts with safe prefix, check if it has spaces (natural language?)
                    # Single word command is usually fine (e.g. "dir"), but "main idea" is bad.
                    has_spaces = " " in cmd_clean
                    
                    if not is_safe and has_spaces and len(cmd_clean.split()) > 2:
                         if verbose: console.print(f"[yellow]⚠️ Invalid command detected ('{arg_hint}'). Fallback to search_web.[/yellow]")
                         tool_hint = "search_web"
                         arguments = {"query": arg_hint}
                    else:
                         arguments = {"command": arg_hint}
                else:
                    if verbose:
                         console.print(f"[yellow]⚠️ Brain 생성 명령어('{arg_hint}')가 자연어 설명으로 감지되어 무시합니다. 키워드 추론을 사용합니다.[/yellow]")
                    # arguments를 비워두면 아래쪽 tool_call 생성 조건(if arguments:)을 만족하지 못해
                    # 자연스럽게 2. Falcon/키워드 폴백 로직으로 넘어감
                    arguments = {}
            elif tool_hint == "get_weather":
                arguments = {"location": arg_hint}
            elif tool_hint == "get_current_time":
                arguments = {"timezone": arg_hint}
            elif tool_hint == "calculate":
                arguments = {"expression": arg_hint}
            elif tool_hint == "read_url":
                arguments = {"url": arg_hint}
            
            if arguments:
                tool_call = {"name": tool_hint, "arguments": arguments}
        
        # 2. Tool Call이 아직 없으면 Falcon/키워드 사용
        if not tool_call:
            if self.tool_caller and self.tool_caller._falcon:
                # Falcon-90M 사용
                if verbose:
                    console.print("[dim]🔧 Tool Caller (Falcon-90M) 호출 중...[/dim]")
                with self._model_lock:
                    tool_call = self.tool_caller.generate_tool_call(user_input)
            else:
                # 키워드 기반 폴백 (모델 없이)
                if verbose:
                    console.print("[dim]🔧 키워드 기반 Tool 추론 중...[/dim]")
                tool_call = self._infer_tool_from_keywords(user_input, tool_hint)
        
        return self._execute_tool_logic(tool_call, verbose, return_raw, user_input)

    def _execute_tool_logic(self, tool_call: dict, verbose: bool, return_raw: bool, user_input: str) -> str:
        """
        Helper to execute the constructed tool call
        """
        # [Critical Fix] Validate arguments against tool definition to prevent "unexpected keyword" errors
        if tool_call and "name" in tool_call and "arguments" in tool_call:
            t_name = tool_call["name"]
            t_args = tool_call["arguments"]
            
            # search_web does not accept 'location', only 'query'
            if t_name in ["search_web", "search_news"] and "location" in t_args and "query" not in t_args:
                if verbose: console.print(f"[yellow]⚠️ Fixing invalid argument for {t_name}: location -> query[/yellow]")
                t_args["query"] = t_args.pop("location")
            
            # get_weather does not accept 'query', only 'location'
            if t_name == "get_weather" and "query" in t_args and "location" not in t_args:
                if verbose: console.print(f"[yellow]⚠️ Fixing invalid argument for {t_name}: query -> location[/yellow]")
                t_args["location"] = t_args.pop("query")

        if "error" in tool_call:
            if verbose:
                console.print(f"[yellow]⚠️ Tool 파싱 실패: {tool_call['error']}[/yellow]")
            return self.brain.direct_respond(user_input)
        
        # 2. Tool 실행 (Retry Logic)
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        
        # [Sanitization] execute_command prefix cleaning
        if tool_name == "execute_command" and "command" in arguments:
            raw_cmd = arguments["command"]
            # Remove common prefixes hallucinated by Brain (e.g. "tool: ls", "command: ls", "도구: ls")
            clean_cmd = re.sub(r'^(tool|command|cmd|도구|명령|실행)\s*[:：]\s*', '', raw_cmd, flags=re.IGNORECASE).strip()
            if clean_cmd != raw_cmd:
                if verbose: console.print(f"[dim]🧹 Command Sanitized: '{raw_cmd}' -> '{clean_cmd}'[/dim]")
                arguments["command"] = clean_cmd
        
        if verbose:
            console.print(f"[dim]🔨 Tool 실행: {tool_name}({arguments})[/dim]")
        
        if self.dashboard:
            self.dashboard.add_log(f"API Call: {tool_name}({arguments})", "Tool")
        
        result = self.tool_executor.execute(tool_name, arguments)
        
        if verbose:
            console.print(Panel(
                JSON.from_data(result),
                title=f"[bold cyan]🔧 {tool_name} { '성공' if result.get('success') else '실패' }[/bold cyan]",
                border_style="cyan" if result.get("success") else "red",
            ))
        
        # [Semantic Error Detection] Soft Error 감지
        if result.get("success", False):
            raw_result = str(result.get("result", "")).lower()
            error_keywords = ["timeout", "timed out", "rate limit", "api error", "access denied", "404 not found", "500 internal server error", "traceback"]
            
            for keyword in error_keywords:
                if keyword in raw_result:
                    if verbose:
                        console.print(f"[yellow]⚠️ Semantic Error 감지: '{keyword}' - 재시도 트리거[/yellow]")
                    result["success"] = False
                    result["error"] = f"Tool returned success but contained error keyword: {keyword}"
                    break
        
        # 3. Brain으로 결과 포맷팅 or 재시도
        if result.get("success", False):
            if return_raw:
                 return result # Return full result dict (with tool, arguments, result keys)
            tool_result = result.get("result", {})
            # Brain의 integrate_response를 사용하여 환각 방지 및 포맷팅 적용
            with self._model_lock:
                return self.brain.integrate_response(user_input, str(tool_result))
        else:
            # Tool 실패 -> 재시도 (Retry)
            # 여기서는 재시도 로직 생략하고 에러 반환 (복잡도 감소)
            return self.brain.direct_respond(f"Tool execution failed: {result.get('error')}")
            error = result.get("error", "Unknown error")
            
            # 모든 Tool 실패 시 1회 재시도 (Brain에게 수정 요청)
            if "retry" not in arguments: # 무한 루프 방지
                if verbose:
                    console.print(f"[bold red]⚠️ 실행 실패: {error}. Brain에게 수정을 요청합니다...[/bold red]")
                
                # Brain에게 수정을 요청하는 프롬프트
                retry_prompt = f"""The tool '{tool_name}' failed with arguments '{arguments}'
Error: "{error}".
The user wants to: "{user_input}".
Please provide CORRECTED arguments for the tool '{tool_name}' to fix this error.
Return ONLY the JSON arguments (e.g. {{"location": "Seoul"}} or {{"command": "python --version"}}). Do NOT explain."""

                with self._model_lock:
                    corrected_args_str = self.brain.direct_respond(
                        retry_prompt, 
                        system_prompt="You are a tool expert. Provide only the corrected JSON arguments."
                    ).strip()
                
                # 마크다운/JSON 파싱 시도
                corrected_args_str = corrected_args_str.replace("```json", "").replace("```", "").strip()
                
                try:
                    import json
                    # 단순 문자열인 경우(예: command string) 처리
                    if not corrected_args_str.startswith("{"):
                         # execute_command라면 문자열을 command로 간주
                         if tool_name == "execute_command":
                             retry_args = {"command": corrected_args_str}
                         else:
                             # 다른 툴은 location 등 키를 알기 어려우므로 JSON 파싱 재시도하거나 포기
                             # 여기서는 간단히 location이나 query로 가정하는 휴리스틱 추가 가능하나,
                             # Brain이 JSON을 주도록 프롬프트했으므로 일단 JSON 로드 시도
                             pass
                    
                    if corrected_args_str.startswith("{"):
                        retry_args = json.loads(corrected_args_str)
                        retry_args["retry"] = True # 재귀 방지 플래그
                        
                        if verbose:
                            console.print(f"[dim]🧠 Brain 수정 제안: {retry_args}[/dim]")

                        retry_result = self.tool_executor.execute(tool_name, retry_args)
                        
                        if verbose:
                            console.print(Panel(
                                JSON.from_data(retry_result),
                                title="[bold cyan]🔧 재시도 결과[/bold cyan]",
                                border_style="cyan" if retry_result.get("success") else "red",
                            ))
                            
                        if retry_result.get("success"):
                            # 성공 시 포맷팅 후 반환
                            tool_result = retry_result.get("result", {})
                            if return_raw:
                                 return retry_result
                            # Brain의 integrate_response를 사용하여 환각 방지 및 포맷팅 적용
                            with self._model_lock:
                                return self.brain.integrate_response(user_input, str(tool_result))
                        else:
                            error = retry_result.get("error", error)
                except Exception as e:
                    if verbose:
                        console.print(f"[dim]⚠️ 재시도 파싱 실패: {e}[/dim]")

            return f"죄송합니다. 명령 실행에 실패했습니다.\n오류: {error}"
    
    def _infer_tool_from_keywords(self, user_input: str, tool_hint: str = "") -> dict:
        """키워드 기반 Tool 호출 추론 (모델 없이)"""
        user_lower = user_input.lower()
        
        # tool_hint가 있으면 우선 사용
        if tool_hint == "get_weather":
            # 도시명 추출 시도
            cities = ["서울", "seoul", "도쿄", "tokyo", "뉴욕", "new york", "런던", "london", 
                      "부산", "busan", "인천", "대구", "대전", "광주"]
            location = "Seoul"  # 기본값
            for city in cities:
                if city in user_lower:
                    location = city.title()
                    break
            return {"name": "get_weather", "arguments": {"location": location}}
        
        elif tool_hint == "search_web":
            # 검색어 추출 (간단한 휴리스틱)
            query = user_input
            for prefix in ["검색해줘", "찾아봐", "알려줘", "뭐야", "search for", "search"]:
                if prefix in user_lower:
                    query = user_input.replace(prefix, "").strip()
                    break
            return {"name": "search_web", "arguments": {"query": query}}
        
        elif tool_hint == "get_current_time":
            # 타임존 추출
            timezone = "Asia/Seoul"  # 기본값
            if "뉴욕" in user_lower or "new york" in user_lower:
                timezone = "America/New_York"
            elif "도쿄" in user_lower or "tokyo" in user_lower:
                timezone = "Asia/Tokyo"
            elif "런던" in user_lower or "london" in user_lower:
                timezone = "Europe/London"
            return {"name": "get_current_time", "arguments": {"timezone": timezone}}
        
        elif tool_hint == "calculate":
            # 수식 추출
            import re
            match = re.search(r'[\d\s+\-*/().]+', user_input)
            expression = match.group().strip() if match else "0"
            return {"name": "calculate", "arguments": {"expression": expression}}
        
        # tool_hint 없을 때 키워드 기반 폴백 (영어 키워드 포함)
        weather_keywords = ["weather", "날씨", "기온", "온도", "temperature"]
        search_keywords = ["search", "find", "검색", "찾아", "알려줘"]
        time_keywords = ["time", "시간", "몇시", "what time", "current time"]
        
        if any(kw in user_lower for kw in weather_keywords):
            # 도시명 추출
            cities = ["seoul", "서울", "tokyo", "도쿄", "new york", "뉴욕", "london", "런던",
                      "busan", "부산", "incheon", "인천", "osaka", "오사카"]
            location = "Seoul"
            for city in cities:
                if city in user_lower:
                    location = city.title().replace("서울", "Seoul").replace("도쿄", "Tokyo").replace("뉴욕", "New York").replace("런던", "London").replace("부산", "Busan")
                    break
            return {"name": "get_weather", "arguments": {"location": location}}
        
        command_keywords = ["실행", "run", "check", "verify", "version", "버전", "확인", "ls", "dir", "command"]
        if any(kw in user_lower for kw in command_keywords) and ("코드" not in user_lower):
             # 간단한 명령어 추출 시도 (매우 단순화됨)
            cmd = "ver" # 기본값
            if "uv" in user_lower:
                 cmd = "uv --version"
            elif "python" in user_lower:
                 cmd = "python --version"
            elif "dir" in user_lower or "목록" in user_lower:
                 cmd = "dir"
            return {"name": "execute_command", "arguments": {"command": cmd}}

        if any(kw in user_lower for kw in search_keywords) or "uv" in user_lower:
            return {"name": "search_web", "arguments": {"query": user_input}}
        
        if any(kw in user_lower for kw in time_keywords):
            return {"name": "get_current_time", "arguments": {"timezone": "Asia/Seoul"}}
        
        return {"error": "Could not infer tool from keywords"}
    
    def _process_rag_attachments(self, user_input: str, verbose: bool = True) -> tuple[str, str]:
        """
        @[filename] 패턴을 찾아 RAG 처리하고 컨텍스트 반환
        
        Returns:
            (cleaned_user_input, rag_context)
        """
        rag_context = ""
        rag_files = re.findall(r"@\[(.*?)\]", user_input)
        
        if not rag_files:
            return user_input, ""
            
        if verbose:
            console.print(f"[dim]📚 RAG 파일 감지: {rag_files}[/dim]")
        
        # Lazy Loading check
        if not hasattr(self, "_rag_engine") or self._rag_engine is None:
            try:
                from src.rag.engine import RAGEngine
                self._rag_engine = RAGEngine()
            except ImportError as e:
                    console.print(f"[red]⚠️ RAG Engine 로드 실패: {e}[/red]")
                    self._rag_engine = None

        if self._rag_engine:
            for file_ref in rag_files:
                # 파일 경로 보정 (현재 디렉토리 기준)
                file_path = file_ref.strip()
                if not Path(file_path).exists():
                        # 혹시 절대 경로가 아니라면 현재 작업 디렉토리에서 찾기
                        file_path = str(Path(project_root) / file_ref.strip())
                
                if Path(file_path).exists():
                    # 1. Ingest (이미 처리된 경우 스킵됨 - Engine 내부 로직)
                    if verbose:
                            console.print(f"[dim]🔄 문서 처리 중: {Path(file_path).name}...[/dim]")
                    status = self._rag_engine.ingest_file(file_path)
                    if verbose:
                            console.print(f"[dim]   Result: {status}[/dim]")
                    
                    # 2. Query (질문과 관련된 내용 검색)
                    # 질문에서 파일 참조 제거 후 검색
                    clean_query = re.sub(r"@\[(.*?)\]", "", user_input).strip()
                    retrieved = self._rag_engine.query(clean_query)
                    
                    if retrieved:
                            rag_context += f"\n\n[Context from {file_ref}]\n{retrieved}\n"
                else:
                    if verbose:
                            console.print(f"[yellow]⚠️ 파일을 찾을 수 없음: {file_ref}[/yellow]")
        
        # 입력에서 파일 참조 제거
        clean_input = re.sub(r"@\[(.*?)\]", "", user_input).strip()
        
        if rag_context and verbose:
            console.print(f"[dim]📄 RAG 컨텍스트 추가됨 ({len(rag_context)} chars)[/dim]")
            
        return clean_input, rag_context

    def chat(self, user_input: str, rag_context: str = "", verbose: bool = True, return_raw_tool_result: bool = False) -> str:
        """
        사용자 질문에 응답 (Thinking -> Tool Calling -> RAG -> Brain 순위)
        
        Args:
            return_raw_tool_result: True일 경우 Tool 호출 결과를 Brain 통합 없이 그대로 반환
        """
        """
        사용자 입력 처리
        
        Args:
            user_input: 사용자 메시지
            verbose: 처리 과정 출력 여부
            
        Returns:
            최종 응답
        """
        if verbose:
            console.print(f"\n[bold]📝 입력:[/bold] {user_input}")

        # 0.1. [RAG] 파일 참조 감지 (@[filename])
        user_input, rag_context = self._process_rag_attachments(user_input, verbose=verbose)
        if rag_context:
             user_input += f"\n\n--- Reference Material ---\n{rag_context}\n--------------------------\n(Answer strictly based on the Reference Material above if relevant.)"
        
        # 0.5. [Multi-Step Pipeline] route_pipeline() 사용하여 복합 작업 분해
        # 예: "최신 AI 트렌드 검색해서 요약해줘" → [TOOL: search] → [DIRECT: 요약]
        pipeline = self.brain.route_pipeline(processed_input if 'processed_input' in dir() else user_input)
        
        if len(pipeline) > 1:
            # 다중 스텝 파이프라인 실행
            if verbose:
                console.print(f"[dim]🔗 파이프라인 감지: {len(pipeline)}단계 실행[/dim]")
                for step in pipeline:
                    console.print(f"[dim]   Step {step['step']}: {step['route']} - {step.get('description', '')}[/dim]")
            
            step_results = {}  # 각 스텝의 결과 저장
            
            for step in pipeline:
                step_num = step["step"]
                route = step["route"]
                tool_hint = step.get("tool_hint", "")
                
                if verbose:
                    console.print(f"[dim]▶ Step {step_num}: {route}[/dim]")
                
                if route == "TOOL":
                    # Tool 실행
                    result = self._handle_tool_call(
                        user_input, 
                        tool_hint, 
                        step.get("specialist_prompt", ""), 
                        verbose=verbose,
                        return_raw=True  # Raw 결과 필요
                    )
                    step_results[step_num] = result
                    
                elif route == "DIRECT":
                    # 이전 스텝의 결과를 컨텍스트로 사용
                    context_from = step.get("context_from_step", step_num - 1)
                    prev_result = step_results.get(context_from, "")
                    
                    # 결과 포맷팅
                    if isinstance(prev_result, dict):
                        prev_result = str(prev_result.get("result", prev_result))
                    
                    # Brain에게 요약/처리 요청
                    with self._model_lock:
                        final_response = self.brain.integrate_response(user_input, str(prev_result))
                    step_results[step_num] = final_response
                    
                elif route == "REASONER":
                    with self._model_lock:
                        result = self.reasoner.solve(step.get("specialist_prompt", user_input))
                    step_results[step_num] = result
            
            # 마지막 스텝 결과 반환
            final_response = step_results.get(len(pipeline), list(step_results.values())[-1])
            
            if verbose:
                console.print(Panel(
                    Markdown(str(final_response)) if isinstance(final_response, str) else JSON.from_data(final_response),
                    title="[bold green]🔗 파이프라인 완료[/bold green]",
                    border_style="green",
                ))
            
            # 번역 처리 (필요시)
            if self.enable_translation and self._translation_pipeline and isinstance(final_response, str):
                try:
                    target_lang_ctx = self._translation_pipeline.to_english(user_input)
                    if target_lang_ctx.is_translated:
                        final_response = self._translation_pipeline.from_english(final_response, target_lang_ctx)
                except Exception as e:
                    logger.error(f"Pipeline translation failed: {e}")
            
            return final_response
        
        # 0.5.1 [Legacy] 기존 복합 질문 분해 (compare/비교 케이스)
        # "비교", "compare", "vs" 등 키워드가 있으면 분해 시도
        complex_keywords = ["비교", "compare", "vs", "difference", "차이", "어때?", "각각", "separately", "each"] # '어때?'는 애매하지만 일단 테스트
        is_complex = any(k in user_input for k in ["비교", "compare", "vs", "difference", "차이", "각각", "separately", "each"])
        
        if is_complex:
            if verbose:
                console.print("[dim]🧩 복합 질문 감지: 분해 시도 중...[/dim]")
            
            sub_queries = self.brain.decompose_query(user_input)
            
            # 분해가 실제로 일어났는지 확인 (1개 이상이고, 원본과 다을 때)
            if len(sub_queries) > 1:
                if verbose:
                    console.print(f"[dim]🧩 분해 결과: {sub_queries}[/dim]")
                
                context_results = []
                for sub_q in sub_queries:
                    # 각 하위 질문 처리
                    # 재귀 호출 방지를 위해 단순 처리 로직 필요하나, 여기서는 chat() 호출하되
                    # 무한 루프 방지를 위해 is_complex 체크가 중요함.
                    # 하지만 sub_q는 단순할 것이므로 괜찮음.
                    # 다만 chat()은 번역/출력을 또 하므로, 내부 함수 _process_single_turn 같은게 필요.
                    # 여기서는 간단히: route -> handle_tool_call 복붙 로직 사용 (함수 분리 권장하지만 일단 인라인)
                    
                    # 1. Brain이 라우팅 결정 (Sub query)
                    # 번역 필요시 번역
                    sub_processed = sub_q
                    if self.enable_translation and self._translation_pipeline:
                        t_ctx = self._translation_pipeline.to_english(sub_q)
                        if t_ctx.is_translated:
                            sub_processed = t_ctx.english_text

                    route_result = self.brain.route(sub_processed)
                    route = route_result.get("route", "DIRECT")
                    
                    step_result = ""
                    if route == "TOOL":
                         tool_hint = route_result.get("tool_hint", "")
                         arg_hint = route_result.get("specialist_prompt", "")
                         # Tool 실행 및 결과 획득 (포맷팅 전의 Raw Result가 필요하지만, _handle_tool_call은 포맷팅된 텍스트 반환)
                         # 여기선 _handle_tool_call의 결과를 그대로 텍스트로 사용
                         step_result = self._handle_tool_call(sub_q, tool_hint, arg_hint, verbose=True)
                    else:
                         step_result = self.brain.direct_respond(sub_processed)
                    
                    
                    context_results.append(f"Query: {sub_q}\nResult: {step_result[:500]}") # 결과 길이 제한 (500자)
                
                # 결과 통합
                aggregated_context = "\n\n".join(context_results)
                
                # 통합 호출 전 메모리 정리 (간접적)
                if hasattr(self.brain.model, "reset"):
                    self.brain.model.reset()
                    
                final_response = self.brain.integrate_response(user_input, aggregated_context)
                
                if verbose:
                    console.print(Panel(
                        Markdown(final_response),
                        title="[bold green]💬 통합 응답[/bold green]",
                        border_style="green",
                    ))
                
                # 번역: en → original_lang (있다면)
                # 주의: decomposition 로직 시작 전에 translation_ctx를 구했어야 함.
                # 하지만 구조상 chat 함수의 메인 파이프라인(0번 단계)보다 먼저 실행됨.
                # 따라서 여기서 별도로 detect/translate 하거나, 0번 단계를 위로 올려야 함.
                # 리팩토링 최소화를 위해 여기서 간단히 처리.
                
                # (이미 chat 함수 진입 시점에는 processed_input이 없으므로, user_input을 이용)
                # [Critical Fix] If raw output is requested, skip ALL translation logic to avoid dict vs string errors
                if return_raw_tool_result:
                    return final_response

                if self.enable_translation and self._translation_pipeline and isinstance(final_response, str):
                    # Brain now outputs English, so we MUST translate to original language.
                    # Attempt to detect language from the original user_input.
                    try:
                        # Re-detect context if not available (since this is inside decomposition block)
                        target_lang_ctx = self._translation_pipeline.to_english(user_input)
                        
                        if target_lang_ctx.is_translated:
                            # User spoke non-English (e.g. Korean), translate back
                            final_response = self._translation_pipeline.from_english(final_response, target_lang_ctx)
                        else:
                            # User spoke English (or detection failed).
                            # If the system policy enforces Korean, we might consider forcing translation here.
                            # However, the standard logic is to respect the input language or explicit user instruction.
                            # For now, we maintain the "Return in Original Language" policy.
                            pass

                    except Exception as e:
                        logger.error(f"Translation logic failed: {e}")

                return final_response

        # 0. 번역 파이프라인: 다국어 → 영어
        translation_ctx = None
        processed_input = user_input
        
        if self.enable_translation and self._translation_pipeline:
            with self._model_lock:
                translation_ctx = self._translation_pipeline.to_english(user_input)
            if translation_ctx.is_translated:
                processed_input = translation_ctx.english_text
                if verbose:
                    console.print(f"[dim]🌐 번역: {translation_ctx.original_lang} → en[/dim]")
                    console.print(f"[dim]   영어: {processed_input[:50]}...[/dim]")
        
        # 1. Brain이 라우팅 결정 (영어로 된 입력 사용)
        # [Fix] RAG 컨텍스트가 있으면 Tool Calling을 방지하고 강제로 DIRECT 응답 유도
        if rag_context:
             if verbose:
                 console.print("[dim]📄 RAG 컨텍스트 존재: 강제로 DIRECT 모드 전환[/dim]")
             route = "DIRECT"
             specialist_prompt = ""
             tool_hint = ""
        else:
             with self._model_lock:
                 route_result = self.brain.route(processed_input)
             route = route_result.get("route", "DIRECT")
             specialist_prompt = route_result.get("specialist_prompt", "")
             tool_hint = route_result.get("tool_hint", "")
        
        if verbose:
            console.print(f"[dim]🧠 라우팅: {route}[/dim]")
        
        # 2. 라우팅에 따른 처리
        if route == "TOOL":
            # Tool Calling
            if verbose:
                console.print(f"[dim]🔧 Tool 호출: {tool_hint}[/dim]")
            # specialist_prompt를 arg_hint로 전달
            # [Critical Fix] Use processed_input (EN) instead of user_input (KO) for tool calling
            final_response = self._handle_tool_call(processed_input, tool_hint, specialist_prompt, verbose, return_raw=return_raw_tool_result)
            
        elif route == "REASONER" and specialist_prompt:
            # Reasoner 호출
            if verbose:
                console.print("[dim]🤔 Reasoner 호출 중...[/dim]")
            
            with self._model_lock:
                specialist_output = self.reasoner.solve(specialist_prompt)
            
            # PoC: Reasoner 출력 직접 반환 (토큰 절약)
            final_response = specialist_output
        else:
            # Brain이 직접 응답
            if verbose:
                console.print("[dim]🧠 Brain 직접 응답...[/dim]")
            with self._model_lock:
                final_response = self.brain.direct_respond(processed_input)
        
        # 3. 번역 파이프라인: 영어 → 원래 언어
        # [Fix] Raw 결과(dict)는 번역하지 않음 + 타입 체크 강제
        if not return_raw_tool_result and isinstance(final_response, str) and translation_ctx and translation_ctx.is_translated and self._translation_pipeline:
            if verbose:
                console.print(f"[dim]🌐 번역: en → {translation_ctx.original_lang}[/dim]")
            with self._model_lock:
                try:
                    final_response = self._translation_pipeline.from_english(final_response, translation_ctx)
                except Exception as e:
                    logger.error(f"Translation failed (main): {e}")
        if verbose:
            console.print(Panel(
                Markdown(str(final_response)) if isinstance(final_response, str) else JSON.from_data(final_response),
                title="[bold green]💬 응답[/bold green]",
                border_style="green",
            ))
            
            # [Thinking Model Visualization]
            # 만약 Thinking Trace가 포함된 경우 (예: <thinking>...</thinking> 또는 유사 패턴)
            # 별도로 파싱하여 보여주는 로직 추가
            if self.use_thinking and self.show_thinking and isinstance(final_response, str):
                 # Thinking Trace가 있는지 확인하고 있으면 별도 패널로 출력
                 # (현재 모델은 명시적인 태그가 없을 수 있으므로, 일단 전체 출력 유지하되 안내 메시지 추가)
                 console.print("[dim blue]🧠 Thinking Process Visualization Enabled (Raw Output)[/dim blue]")
        
        return final_response

    def _setup_cowork_logger(self):
        """Cowork 전용 로거 설정 (TUI 에러 추적용)"""
        logger = logging.getLogger("cowork")
        logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거
        if logger.handlers:
            logger.handlers.clear()
            
        fh = logging.FileHandler("cowork.log", encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger, fh

    def run_cowork_flow(self, user_goal: str, workspace_root: str = ".", use_tui: bool = True) -> str:
        """
        Tiny Cowork v2.0 실행 (TUI & Parallel 지원)
        """
        from rich.live import Live

        from tiny_moa.cowork.parallel_runner import ParallelRunner
        from tiny_moa.cowork.planner import PlannerAgent
        from tiny_moa.cowork.skills.file_skills import CoworkFileSkill
        from tiny_moa.cowork.task_queue import TaskQueue, TaskStatus
        from tiny_moa.cowork.workers.brain_worker import BrainWorker
        from tiny_moa.cowork.workers.office_worker import OfficeWorker
        from tiny_moa.cowork.workers.researcher import ResearchWorker
        from tiny_moa.cowork.workers.tool_worker import ToolWorker
        from tiny_moa.cowork.workers.writer import WriterWorker
        from tiny_moa.cowork.workspace import WorkspaceContext
        from tiny_moa.ui.dashboard import CoworkDashboard

        # [Fix] Keep reference to handler for cleanup
        logger, log_handler = self._setup_cowork_logger()
        logger.info(f"--- Starting Cowork Session: {user_goal} ---")

        workspace = WorkspaceContext(workspace_root)
        queue = TaskQueue()
        planner = PlannerAgent(self.brain)
        file_skill = CoworkFileSkill(workspace)
        dashboard = CoworkDashboard(user_goal)
        self.dashboard = dashboard

        runner = ParallelRunner(max_workers=4)

        # 0. Pre-process RAG attachments
        # [Fix] Handle @[filename] in Cowork mode
        user_goal, rag_context = self._process_rag_attachments(user_goal, verbose=use_tui)
        if rag_context:
             logger.info(f"RAG Context extracted ({len(rag_context)} chars)")
             if use_tui:
                 dashboard.add_log("RAG Context attached from files.", "System")





        # Worker initialization
        researcher = ResearchWorker("Research-1", logger, self)
        writer = WriterWorker("Writer-1", logger, self.brain, file_skill)
        brain_worker = BrainWorker("Brain-1", logger, self.brain)
        tool_worker = ToolWorker("Tool-1", logger, self)
        office_worker = OfficeWorker("Office-1", logger, self)

        # 0. Intelligent Routing & Fast Track (Optimization)
        # Check if the task is simple (Tool or Direct) using Brain's router
        route_data = self.brain.route(user_goal)
        route = route_data.get("route", "DIRECT")
        
        # Determine if it's a simple text/file summary request (Heuristic Fast track)
        is_simple_summary = any(kw in user_goal.lower() for kw in ["요약", "정리", "summarize", "read", "읽고"]) and len(user_goal) < 50
        
        # [Fix v2] RAG + TOOL 복합 요청 감지
        # RAG 컨텍스트가 있어도 날씨/검색/뉴스 키워드가 있으면 TOOL 라우팅 유지
        tool_keywords = ["날씨", "weather", "검색", "search", "뉴스", "news", "시간", "time"]
        needs_tool = any(kw in user_goal.lower() for kw in tool_keywords)
        
        if rag_context and not needs_tool:
            logger.info("RAG context detected (no tool needed). Forcing DIRECT/Summary mode.")
            route = "DIRECT"
            is_simple_summary = True
        elif rag_context and needs_tool:
            logger.info("RAG + TOOL hybrid detected. Will execute both.")
            # route 유지 (TOOL), is_simple_summary는 False로

        # [Complexity Check]
        # 질문이 길거나 복합적인 연결어가 많으면 LLM Planner를 강제 사용 (Regex 분해보다 정확함)
        complexity_signals = ["그리고", "and", "also", "?", "what about", "vs", "compare"]
        # ?가 2개 이상이거나, 연결어가 2개 이상이면 복잡한 쿼리로 간주
        complexity_score = sum(user_goal.count(sig) for sig in complexity_signals)
        is_complex_query = (len(user_goal) > 60) or (complexity_score >= 2)
        
        # [Office Document Detection]
        # Office 문서 생성 요청은 항상 복잡한 쿼리로 처리 (Brain이 내용 생성해야 함)
        office_keywords = ["ppt", "powerpoint", "발표", "프레젠테이션", "슬라이드",
                          "word", "docx", "보고서", "문서", "제안서",
                          "excel", "xlsx", "엑셀", "스프레드시트", "통계"]
        is_office_request = any(kw in user_goal.lower() for kw in office_keywords)
        if is_office_request:
            is_complex_query = True
            logger.info("Office document request detected. Forcing LLM Planner.")
        
        if is_complex_query:
            logger.info(f"Complex query detected (Length: {len(user_goal)}, Score: {complexity_score}). Enforcing LLM Planner.")

        # Decisions
        # If complex, bypass_llm_planner should be FALSE (i.e., Use Planner)
        bypass_llm_planner = (not is_complex_query) and ((route in ["TOOL", "DIRECT"]) or is_simple_summary)
        
        if use_tui:
            live = Live(dashboard.generate_layout(), refresh_per_second=4, screen=False)
            live.start()
            dashboard.add_log("System initialized.", "System")
            if bypass_llm_planner:
                dashboard.add_log(f"Intelligent bypass enabled (Route: {route}).", "System")
            else:
                dashboard.add_log("Complex task detected. Using LLM Planner.", "System")
            live.update(dashboard.generate_layout())
        
        try:
            # 1. Plan
            context_str = workspace.get_context_description()
            if rag_context:
                context_str += f"\n\n=== Attached File Context ===\n{rag_context}\n============================="



            if use_tui: 
                dashboard.add_log("Analyzing request and creating plan...", "Planner")
                live.update(dashboard.generate_layout())
            
            # [NEW] RAG + TOOL 복합 요청 처리
            if rag_context and needs_tool:
                logger.info("Creating hybrid RAG+TOOL pipeline")
                tasks_data = []
                
                # 1. 문서 분석 태스크
                tasks_data.append({
                    "description": f"Analyze the provided file context and summarize: '{user_goal}'",
                    "agent": "brain"
                })
                
                # 2. 필요한 Tool 태스크 추가
                user_lower = user_goal.lower()
                if any(kw in user_lower for kw in ["날씨", "weather"]):
                    # 도시 추출
                    cities = ["서울", "seoul", "도쿄", "tokyo", "뉴욕", "부산", "인천", "대구"]
                    location = "Seoul"
                    for city in cities:
                        if city in user_lower:
                            location = city.title()
                            break
                    tasks_data.append({
                        "description": f"{location} 날씨",
                        "agent": "tool"
                    })
                
                if any(kw in user_lower for kw in ["뉴스", "news"]):
                    # Smart Decomposition for News: Split by connection words to isolate news context
                    import re
                    # Split by sentence delimiters. Do NOT split by comma to preserve entity lists (e.g. "A, B, C news")
                    clauses = re.split(r'(?:그리고|and|\.|\?|also)\s+', user_goal)
                    news_clauses = [c for c in clauses if any(kw in c.lower() for kw in ["뉴스", "news"])]
                    
                    if not news_clauses:
                        news_clauses = [user_goal] # Fallback
                    
                    for clause in news_clauses:
                        sub_qs = self.brain.decompose_query(clause)
                        for q in sub_qs:
                            tasks_data.append({
                                "description": q,
                                "agent": "tool"
                            })
                
                if any(kw in user_lower for kw in ["검색", "search"]):
                    tasks_data.append({
                        "description": user_goal,  # 원본 쿼리 사용
                        "agent": "tool"
                    })
                
                dashboard.add_log(f"Hybrid plan created with {len(tasks_data)} tasks.", "Planner")
                
            elif bypass_llm_planner and route == "TOOL":
                # Decompose complex questions into simple tool tasks
                sub_queries = self.brain.decompose_query(user_goal)
                logger.info(f"Using TOOL decomposition: {sub_queries}")
                tasks_data = [{"description": q, "agent": "tool"} for q in sub_queries]
                if len(sub_queries) > 1:
                    dashboard.add_log(f"Decomposed into {len(sub_queries)} tool tasks.", "Planner")
            elif bypass_llm_planner and route == "DIRECT" and not is_simple_summary:
                # Simple direct response, but also check for decomposition
                sub_queries = self.brain.decompose_query(user_goal)
                logger.info(f"Using DIRECT decomposition: {sub_queries}")
                tasks_data = [{"description": q, "agent": "brain"} for q in sub_queries]
            elif bypass_llm_planner and is_simple_summary:
                # Heuristic Planning for summary
                logger.info("Using fast-track heuristic plan for summary.")
                
                if rag_context:
                     # If we already have context, we don't need a separate RAG step
                     tasks_data = [
                        {"description": f"Analyze the provided file context and answer the user's question: '{user_goal}'", "agent": "brain"},
                        {"description": "Format the answer clearly", "agent": "writer"}
                     ]
                else:
                    tasks_data = [
                        {"description": f"Locate and read target files related to '{user_goal}'", "agent": "rag"},
                        {"description": "Summarize the extracted content in Korean", "agent": "brain"},
                        {"description": "Save the final summary", "agent": "writer"}
                    ]
            else:
                logger.info("Creating full LLM plan...")
                tasks_data = planner.create_plan(user_goal, context_str)
            
            logger.info(f"Plan created: {tasks_data}")
            
            for t in tasks_data:
                queue.add_task(t.get("description"), t.get("agent", "brain"))
            
            all_tasks = queue.get_all_tasks()
            if use_tui: 
                 dashboard.update_tasks([{"id": t.id, "desc": t.description, "status": t.status.name, "agent": t.agent_type} for t in all_tasks])
                 dashboard.add_log(f"Plan created with {len(tasks_data)} tasks.", "Planner")
                 live.update(dashboard.generate_layout())

            # 2. Execute
            results = []
            
            # [Fix] Inject RAG context into history so workers can see it
            if rag_context:
                results.append(f"[CONTEXT FROM UPLOADED FILES]\n{rag_context}\n[END OF CONTEXT]")
            
            
            # Use ParallelRunner if possible (experimental)
            # Find independent tasks (those with same agent or no clear dependency)
            # For simplicity, we run 'tool' and 'rag' tasks in parallel if multiple.
            # 'brain' and 'writer' usually depend on previous results.
            
            parallelizable = [t for t in all_tasks if t.agent_type in ["tool", "rag"]]
            sequential = [t for t in all_tasks if t.agent_type in ["brain", "writer", "office"]]
            
            # [FIX] Hybrid 모드에서는 순차 태스크(brain)를 먼저 실행
            is_hybrid_mode = rag_context and needs_tool
            if is_hybrid_mode:
                # brain이 먼저, 그 다음 tool
                first_phase = sequential
                second_phase = parallelizable
            else:
                # 기존: 병렬 먼저, 순차 나중
                first_phase = parallelizable
                second_phase = sequential
            
            def execute_single_task(task):
                 try:
                    agent_type = task.agent_type.lower()
                    task_lower = task.description.lower()
                    history = "\n\n".join(results) # Note: Parallel tasks won't have latest history from siblings
                    
                    if use_tui:
                        task.status = TaskStatus.RUNNING
                        # Show more detail in log: Agent + Preview
                        dashboard.add_log(f"[{agent_type.upper()}] Thinking: {task.description[:40]}...", agent_type.capitalize())
                        dashboard.update_tasks([{"id": t.id, "desc": t.description, "status": t.status.name, "agent": t.agent_type} for t in all_tasks])
                        live.update(dashboard.generate_layout())

                    if agent_type == "tool":
                        # For tools, we want to know WHICH tool.
                        # We use chat(return_raw_tool_result=True)
                        res = tool_worker.execute(task.description)
                        if use_tui and isinstance(res, dict):
                             # Extract actual info for log
                             t_name = res.get("tool", "unknown")
                             # Search result inner content is in res['result']
                             t_res = res.get("result", {})
                             if isinstance(t_res, dict) and "results" in t_res:
                                  # News/Search result case
                                  articles = t_res["results"]
                                  dashboard.add_log(f"Tool {t_name.upper()}: Found {len(articles)} items", "Tool")
                                  for art in articles:
                                       title = art.get('title', 'No Title')
                                       url = art.get('url') or art.get('href', 'No URL')
                                       dashboard.add_log(f"ARTICLE: {title}", "Source")
                                       dashboard.add_log(f"   URL: {url}", "Source")
                             elif isinstance(t_res, dict) and "temperature" in t_res:
                                  # Weather case
                                  dashboard.add_log(f"Weather: {t_res['temperature']}, {t_res['condition']}", "Tool")
                             else:
                                  dashboard.add_log(f"API {t_name}: {str(t_res)[:50]}...", "Tool")
                             live.update(dashboard.generate_layout())
                        # If raw, we need to extract the 'result' part for the final integration
                        if isinstance(res, dict) and "result" in res:
                             res = res["result"]
                    elif agent_type == "rag":
                        res = researcher.execute(task.description)
                    elif agent_type == "writer":
                        res = writer.execute(task.description, history=history, user_goal=user_goal)
                    elif agent_type == "office":
                        # Office 문서 생성 (PPT, Word, Excel)
                        res = office_worker.execute(task.description)
                        if use_tui:
                            if isinstance(res, dict) and res.get("success"):
                                dashboard.add_log(f"Office: Created {res.get('path', 'document')}", "Office")
                            live.update(dashboard.generate_layout())
                    else:
                        res = brain_worker.execute(task.description, history=history)
                    
                    # [Critical Fix] Ensure task.result is ALWAYS a string for history/integration
                    task.result = str(res)
                    task.status = TaskStatus.COMPLETED
                    if use_tui:
                        # Log success and a small snippet of the result
                        res_str = str(res)
                        dashboard.add_log(f"Success: {task.id} ({res_str[:50]}...)", agent_type.capitalize())
                        dashboard.update_tasks([{"id": t.id, "desc": t.description, "status": t.status.name, "agent": t.agent_type} for t in all_tasks])
                        live.update(dashboard.generate_layout())
                    return res
                 except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.result = str(e)
                    if use_tui:
                        dashboard.add_log(f"Failed {task.id}: {e}", "Error")
                        dashboard.update_tasks([{"id": t.id, "desc": t.description, "status": t.status.name, "agent": t.agent_type} for t in all_tasks])
                        live.update(dashboard.generate_layout())
                    raise e

            # Run first phase tasks
            if first_phase:
                logger.info(f"Running {len(first_phase)} first-phase tasks.")
                if len(first_phase) > 1 and all(t.agent_type in ["tool", "rag"] for t in first_phase):
                    # Parallel execution for tool/rag
                    t_dicts = [{"id": t.id, "description": t.description, "agent": t.agent_type} for t in first_phase]
                    def runner_wrapper(t_dict):
                        target_task = next(tt for tt in first_phase if tt.id == t_dict['id'])
                        return execute_single_task(target_task)
                    runner.run_tasks(t_dicts, runner_wrapper)
                else:
                    # Sequential execution
                    for t in first_phase: execute_single_task(t)

            # Update results after first phase
            for t in first_phase:
                if t.status == TaskStatus.COMPLETED:
                     results.append(f"[TASK: {t.description}]\nDATA: {t.result}")
            logger.info(f"After first_phase: {len(results)} results collected. Last: {results[-1][:100] if results else 'NONE'}...")

            # Run second phase tasks
            if second_phase:
                logger.info(f"Running {len(second_phase)} second-phase tasks.")
                if len(second_phase) > 1 and all(t.agent_type in ["tool", "rag"] for t in second_phase):
                    # Parallel execution for tool/rag
                    t_dicts = [{"id": t.id, "description": t.description, "agent": t.agent_type} for t in second_phase]
                    def runner_wrapper2(t_dict):
                        target_task = next(tt for tt in second_phase if tt.id == t_dict['id'])
                        return execute_single_task(target_task)
                    runner.run_tasks(t_dicts, runner_wrapper2)
                else:
                    # Sequential execution
                    for t in second_phase: execute_single_task(t)
                    
            # Update results after second phase
            for t in second_phase:
                if t.status == TaskStatus.COMPLETED:
                     results.append(f"[TASK: {t.description}]\nDATA: {t.result}")

            # 3. Final Integration (Synthesis)
            if use_tui: 
                dashboard.add_log("Synthesizing final report...", "Brain")
                live.update(dashboard.generate_layout())
            
            logger.info("Performing final integration...")
            
            # [Optimization] If we have Brain task results (summaries), 
            # we explicitly remove the raw RAG context to prevent it from overwhelming the context window
            # or confusing the model vs the summarized output.
            effective_results = results
            if len(results) > 1 and "[CONTEXT FROM UPLOADED FILES]" in results[0]:
                # Check if we have any valid task outputs (Brain/Tool)
                has_task_output = any("[TASK:" in r for r in results[1:])
                if has_task_output:
                    logger.info("Removing raw RAG context from final integration input as tasks have processed it.")
                    effective_results = results[1:]
            
            input_data = "\n\n".join(effective_results)
            logger.info(f"Input data to Brain: {input_data[:500]}...") # Log first 500 chars to check
            
            with self._model_lock:
                final_report = self.brain.integrate_response(user_goal, input_data)
            
            logger.info(f"Pre-translation output: {final_report}")

            # [English-First Strategy]
            # Brain은 영어를 생성하므로, 만약 사용자 질문이 한국어였다면(또는 번역 파이프라인이 있다면) 한국어로 번역
            if self.enable_translation and self._translation_pipeline and isinstance(final_report, str):
                if use_tui:
                     dashboard.add_log("Translating report to Korean...", "System")
                     live.update(dashboard.generate_layout())
                
                # 타겟 언어 감지를 위해 user_goal 재분석 (cowork flow는 chat과 별개라 직접 수행)
                t_ctx = self._translation_pipeline.to_english(user_goal)
                if t_ctx.is_translated: # user_goal이 영어가 아니었다면 (즉 한국어 등)
                     try:
                         final_report = self._translation_pipeline.from_english(final_report, t_ctx)
                     except Exception as e:
                         logger.error(f"Translation failed (cowork): {e}")

            if use_tui: 
                dashboard.add_log("Flow completed successfully.", "System")
                live.stop()
            
            # Save final report to file (Integrated Result)
            try:
                write_msg = workspace.write_file("docs/cowork_result.md", final_report)
                logger.info(f"Auto-save result: {write_msg}")
                console.print("\n[green]ℹ️ 작업 결과가 저장되었습니다: docs/cowork_result.md[/green]")
            except Exception as e:
                logger.error(f"Failed to auto-save cowork_result.md: {e}")

            logger.info("Cowork flow completed.")
            self.dashboard = None
            return final_report
            
        except Exception as e:
            logger.critical(f"Fatal error in cowork flow: {e}", exc_info=True)
            self.dashboard = None
            if use_tui: live.stop()
            raise e
        finally:
            # [Fix] Clean up log handler
            if log_handler:
                log_handler.close()
                logger.removeHandler(log_handler)


def interactive_mode():
    """대화형 모드"""
    console.print(Panel(
        "[bold]🤖 Tiny MoA 대화형 모드[/bold]\n"
        "🔧 Tool Calling: 날씨, 검색, 계산, 시간\n"
        "🌐 다국어 지원: 한국어, 일본어, 중국어 등\n"
        "종료: 'quit' 또는 'exit'",
        border_style="blue",
    ))
    
    moa = TinyMoA()
    
    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[dim]👋 안녕히 가세요![/dim]")
                break
            
            if not user_input.strip():
                continue
            
            moa.chat(user_input)
            
        except KeyboardInterrupt:
            console.print("\n[dim]👋 안녕히 가세요![/dim]")
            break


if __name__ == "__main__":
    interactive_mode()

