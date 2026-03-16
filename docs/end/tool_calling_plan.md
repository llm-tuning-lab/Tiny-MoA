# 🔧 Tool Calling 구현 계획

> **목표:** Tiny MoA에 실제 Tool Calling 기능 추가 - 날씨, 검색, 계산기 등 외부 API 호출

---

## 📋 현재 상태

### PoC 완료 항목
- ✅ Brain (LFM2.5-1.2B) + Reasoner (Falcon-R-0.6B) 조합
- ✅ 코딩/수학 → Reasoner 라우팅
- ✅ 일반 대화 → Brain 직접 응답
- ✅ llama.cpp 기반 CPU 추론

### 부족한 점
- ❌ "오늘 날씨는?" 같은 질문에 실제 API 호출 불가
- ❌ Tool schema 정의 및 파싱 없음
- ❌ Function Calling 전용 모델 미사용

---

## 🎯 구현 목표

### Phase 1: Tool Schema 정의
```python
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
]
```

### Phase 2: Tool Calling 전용 모델 통합

| 모델 | 파라미터 | 양자화 | 용도 |
|------|----------|--------|------|
| **Falcon-H1-Tiny-Tool-Calling-90M** | 90M | Q8_0 (~0.1GB) | JSON 생성 |
| **LFM2.5-1.2B** (이미 있음) | 1.17B | Q4_K_M | Tool 필요 여부 판단 |

### Phase 3: 라우팅 로직 확장

```
사용자 입력
     │
     ▼
┌─────────────────────────────────────────┐
│           🧠 Brain (LFM2.5-1.2B)        │
│  라우팅 결정:                           │
│  - REASONER: 코딩/수학                  │
│  - DIRECT: 일반 대화                    │
│  - TOOL: 외부 정보 필요 (날씨, 검색 등)  │ ← NEW
└─────────────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    ┌───────────┐       ┌────────────────┐
    │ Reasoner  │       │ Tool Caller    │ ← NEW
    │ (600M)    │       │ (90M)          │
    └───────────┘       └────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Tool Executor   │ ← NEW
                    │ (실제 API 호출)  │
                    └─────────────────┘
```

---

## 📁 파일 구조

```
src/tiny_moa/
├── __init__.py
├── brain.py           # 라우터 (TOOL 라우팅 추가)
├── reasoner.py        # 코딩+수학
├── tool_caller.py     # [NEW] Falcon-Tool-Calling-90M
├── tool_executor.py   # [NEW] 실제 API 호출
├── tools/             # [NEW] 개별 도구 구현
│   ├── __init__.py
│   ├── weather.py
│   ├── search.py
│   └── calculator.py
├── orchestrator.py    # TOOL 라우팅 처리 추가
└── main.py
```

---

## 🔄 실행 흐름 예시

### 입력: "서울 날씨는?"

1. **Brain 라우팅**: `{"route": "TOOL", "tool_hint": "weather"}`
2. **Tool Caller 호출**:
   ```json
   {"name": "get_weather", "arguments": {"location": "Seoul"}}
   ```
3. **Tool Executor 실행**: OpenWeatherMap API 호출
4. **Brain 통합**: "서울의 현재 날씨는 맑음, 기온 22°C입니다."

---

## 📦 의존성 추가

```txt
# requirements.txt (추가)
requests>=2.31.0       # API 호출
python-weather>=2.0.0  # 날씨 (옵션)
duckduckgo-search>=6.0 # 검색 (옵션)
```

---

## ⏱️ 예상 일정

| 단계 | 작업 | 예상 시간 |
|------|------|-----------|
| 1 | Tool schema 정의 | 1시간 |
| 2 | Tool Caller 래퍼 작성 | 2시간 |
| 3 | Tool Executor + 개별 도구 | 3시간 |
| 4 | Brain 라우팅 로직 확장 | 2시간 |
| 5 | Orchestrator 통합 | 2시간 |
| 6 | 테스트 및 디버깅 | 2시간 |
| **총계** | | **~12시간** |

---

## ⚠️ 주의사항

1. **Falcon-Tool-Calling-90M**은 영어만 지원
2. API 키 필요 (OpenWeatherMap, DuckDuckGo는 무료)
3. Rate limit 고려 필요
4. Tool 호출 실패 시 fallback 로직 필요

---

## 📚 참고 자료

- [Falcon-H1-Tiny-Tool-Calling-90M](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M)
- [LFM2.5 Tool Use](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct#tool-use)
- [OpenAI Function Calling Format](https://platform.openai.com/docs/guides/function-calling)



사례 추가:
You: How to turn a project using a pip into a uv environment

📝 입력: How to turn a project using a pip into a uv environment
🧠 라우팅: DIRECT
🧠 Brain 직접 응답...
╭───────────────────────────────── 💬 응답 ──────────────────────────────────╮
│ It seems there might be some confusion in your question. Let's break it    │
│ down:                                                                      │
│                                                                            │
│ You're asking about turning a project that uses pip into a Uv              │
│ (Unity/Visual Studio) environment. But here's what I need to clarify:      │
│                                                                            │
│                   1. [pip] is a Python package manager.                    │
│                                                                            │
│  • It installs packages using pip install <package>.                       │
│  • It doesn't create development environments like Uv or Visual Studio.    │
│                                                                            │
│ If you're trying to run or manage Python projects (like using pip) in an   │
│ IDE such as Visual Studio Code, UvEngine, or another editor/IDE, that's    │
│ not something pip does directly — it’s the tool used by developers during  │
│ development.                                                               │
│                                                                            │
│  2. If you meant to create a Unity project (for example with Unity Hub),   │
│                                 then yes:                                  │
│                                                                            │
│  • You can use pip install -r requirements.txt in your IDE or terminal.    │
│  • Then open the project in Unity and configure it using the Unity Editor. │
│                                                                            │
│ But if you're talking about creating an environment for running tests,     │
│ packaging, or deploying Python code into a Uv environment (like Visual     │
│ Studio), that's not standard.                                              │
│                                                                            │
│                        3. Possible Interpretation:                         │
│                                                                            │
│ You might be referring to setting up a development environment for testing │
│ or running your project using Unity (e.g., with Unity Hub and Visual       │
│ Studio). In that case:                                                     │
│                                                                            │
│  • You can install the necessary tools via pip in your IDE.                │
│  • Then configure the project settings in Unity or Visual Studio.          │
│                                                                            │
│ ────────────────────────────────────────────────────────────────────────── │
│                        If you meant something else:                        │
│                                                                            │
│ Could you clarify what you're trying to do? For example:                   │
│                                                                            │
│  • Are you trying to run a Python project in an IDE?                       │
│  • Are you setting up Unity with Python tools?                             │
│  • Is there a specific package or tool named "Uv" that you're referring    │
│    to?                                                                     │
│                                                                            │
│ Let me know and I’ll give you a more accurate guide!                       │
╰────────────────────────────────────────────────────────────────────────────╯

You:




이 결과는 실패입니다. LFM2.5 Brain 모델이 uv (Rust 기반 Python 패키지 매니저)를 전혀 모르고 있습니다.

🚨 분석
지식 단절 (Knowledge Cutoff):
사용자는 "uv" (Astral이 만든 최신 Python 패키지 매니저)를 물었습니다.
LFM2.5 Brain은 이를 "Unity/Visual Studio" 또는 **"UvEngine"**으로 잘못 해석했습니다 (Hallucination).
이는 모델 학습 데이터에 최신 uv (2024년 급부상) 정보가 없거나 부족하기 때문입니다.
라우팅 오류:
'pip', 'environment' 같은 단어 때문에 코딩 질문임에도 불구하고 DIRECT (Brain)로 라우팅되었습니다.
Brain이 "일반 대화"로 판단하고 직접 답하려다 망친 케이스입니다.
🛠️ 해결책
검색(Web Search) 기능 필수: 최신 기술(uv, LangChain v0.2 등)은 모델이 모를 확률이 100%입니다. Tool Calling 계획(검색 도구 추가)이 더더욱 중요해졌습니다.
라우터 개선: 'pip', 'python', 'environment' 같은 키워드는 무조건 REASONER로 보내거나, SEARCH TOOL을 쓰게 해야 합니다.
결론: Tool Calling 계획대로 "Search Tool"을 빨리 붙여야 이런 최신 질문에 답할 수 있습니다. 지금 상태로는 최신 기술 질문에 취약합니다.