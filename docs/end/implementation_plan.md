# Tiny MoA (Mixture of Agents) PoC 상세 명세서

> **작성일:** 2026년 1월 23일
> **프로젝트명:** Tiny MoA (Mixture of Agents) Proof of Concept
> **목표:** 4B 모델 하나를 띄우는 것보다 효율적으로, 1.2B "Brain" 모델과 90M~600M급 "Specialist" 모델들을 조합하여 더 나은 성능을 달성하는 아키텍처 설계 및 구현

---

## 1. 프로젝트 개요

### 1.1 핵심 아이디어

"GPU Poor"를 위한 **Tiny MoA (Mixture of Agents)** 아키텍처입니다.

- **중앙 사령관 (Brain):** 사용자의 요청을 분석하고, 어떤 전문가 모델을 호출할지 결정
- **전문가 그룹 (Specialists):** 각 분야(코딩, 추론, 도구호출, OCR 등)에 특화된 초소형 모델들

이 구조는 2026년 1월 현재 최신 "작지만 강력한(Small but Powerful)" 모델 트렌드를 활용합니다:
- **LiquidAI LFM2.5 시리즈:** 28T 토큰으로 학습된 1.2B급 고성능 모델
- **TII Falcon-H1-Tiny 시리즈:** 90M~600M급 Hybrid (Mamba+Attention) 아키텍처

### 1.2 왜 이 접근법인가?

| 전통적 접근 | Tiny MoA 접근 |
|------------|---------------|
| 4B~7B 단일 모델 | 1.2B Brain + 90M~600M Specialists |
| VRAM: 8~16GB 필요 | VRAM: 2~4GB로 충분 |
| 단일 모델의 한계 | 분야별 최적화된 전문가 |
| 로딩 시간 김 | 경량 모델로 즉시 응답 |

---

## 2. 모델 분석 및 선정

### 2.1 Brain 모델 후보

#### LiquidAI LFM2.5-1.2B-Thinking (권장)
| 항목 | 사양 |
|------|------|
| **파라미터** | 1.17B |
| **아키텍처** | 16 layers (10 LIV convolution + 6 GQA blocks) |
| **학습 토큰** | 28T tokens |
| **컨텍스트** | 32,768 tokens |
| **지원 언어** | English, Arabic, Chinese, French, German, Japanese, Korean, Spanish |
| **추천 용도** | Agentic tasks, Data extraction, RAG |
| **비추 용도** | Knowledge-intensive tasks, Programming (→ Specialist에게 위임) |

**선정 이유:**
- 1B급에서 GPQA(38.89), IFEval(86.23), BFCLv3(49.12) 압도적 성능
- 한국어 포함 8개 언어 지원
- GGUF, ONNX, MLX 모든 포맷 제공
- CPU에서 116 tok/s (AMD Ryzen) 달성

#### LiquidAI LFM2.5-1.2B-Instruct (대안)
- Thinking 모델과 동일 아키텍처
- 더 짧은 응답 생성 (낮은 latency)
- 일반 지시 따르기에 최적화

### 2.2 Specialist 모델 분석

#### 2.2.1 도구 호출 (Tool Calling)

| 모델 | 파라미터 | 특징 | BFCL Score |
|------|----------|------|------------|
| **Falcon-H1-Tiny-Tool-Calling** | 90M | Hybrid Mamba+Attention, 관련성 판단 94.44% | 41.23% |
| Google FunctionGemma | 270M | Gemma3 기반, 파인튜닝 필요 | - |

**Falcon-H1-Tiny-Tool-Calling 선정 이유:**
- 90M임에도 관련성 판단 정확도 **94.44%** (FunctionGemma 61.10% 압도)
- JSON 함수 호출 포맷 정확히 생성
- 블로그에 따르면 CoT(Chain-of-Thought) 배제로 반복 문제 해결

#### 2.2.2 추론/수학 (Reasoning)

| 모델 | 파라미터 | AIME24 pass@1 | MATH500 | 특징 |
|------|----------|---------------|---------|------|
| **Falcon-H1-Tiny-R-0.6B** | 600M | 75.0% | 94.0% | GRPO 적용, SoTA |
| Falcon-H1-Tiny-R-90M | 90M | 5.0% | 39.7% | 경량 버전 |

**Falcon-H1-Tiny-R-0.6B 선정 이유:**
- AIME24에서 MobileLLM-R1-950M(15.5%)을 5배 압도
- Anti-curriculum 학습: 추론 데이터로 처음부터 사전학습
- GRPO(Group Relative Policy Optimization) 강화학습 적용
- **코딩도 가능:** LiveCodeBench v6에서 39.0% 달성 (MobileLLM-R1 19.9% 압도)

> **💡 Coder(90M) vs Reasoner(0.6B) 선택 기준:**
> | 요청 유형 | 추천 모델 | 이유 |
> |-----------|-----------|------|
> | 함수 완성, 간단한 코드 | Coder (90M) | FIM 지원, 빠름 |
> | 알고리즘 문제, 논리적 구현 | **Reasoner (0.6B)** | 추론력 필요 |
> | AIME/경시대회 스타일 코딩 | **Reasoner (0.6B)** | 수학+코드 복합 |

#### 2.2.3 코딩 (Coding)

| 모델 | 파라미터 | HumanEval+ | MBPP+ | FIM 지원 |
|------|----------|------------|-------|----------|
| **Falcon-H1-Tiny-Coder** | 90M | 14.63% | 34.92% | ✅ |
| Qwen2.5-Coder-0.5B | 500M | 23.17% | 48.67% | ✅ |

**Falcon-H1-Tiny-Coder 선정 이유:**
- Python FIM(Fill-in-the-Middle) 지원
- 90M로 Qwen 0.5B 대비 5배 작지만 기본 코딩 가능
- Continue VS Code 플러그인과 연동 검증됨

#### 2.2.4 다국어 (Multilingual)

| 모델 | 파라미터 | IFEVAL | M-MMLU | 지원 언어 |
|------|----------|--------|--------|-----------|
| **Falcon-H1-Tiny-Multilingual** | 100M | 52.00% | 45.00% | 17개 언어 |
| SmolLM2-135M-Instruct | 135M | 30.69% | 25.63% | - |

**Falcon-H1-Tiny-Multilingual 선정 이유:**
- 17개 언어 지원 (한국어 포함)
- IFEVAL 52% (SmolLM2 대비 21%p 우위)
- 100M으로 극도로 가벼움

#### 2.2.5 OCR/문서 처리

| 모델/도구 | 파라미터 | 특징 | 속도 |
|-----------|----------|------|------|
| **docling** (라이브러리) | - | PDF, DOCX, PPTX, 이미지 등 다양한 포맷 | - |
| **LightOnOCR-2-1B** | 1B | End-to-End VLM, 수식/표 지원 | 5.71 pages/s (H100) |
| GraniteDocling | 258M | IBM, Docling 통합 | - |

**권장 조합:**
1. 기본: `docling` 라이브러리 (모델 없이 문서 파싱)
2. 고급: `LightOnOCR-2-1B` (스캔 문서, 수식 인식 필요시)

#### 2.2.6 오디오 (Audio) - 선택적

| 모델 | 파라미터 | 특징 |
|------|----------|------|
| **LFM2.5-Audio-1.5B** | 1.5B | 음성↔텍스트 Native, 8x 빠른 디토크나이저 |

**선정 이유:**
- 파이프라인 방식(ASR→LLM→TTS) 대비 End-to-End로 지연 최소화
- Multi-turn, Multi-modal 채팅 지원

#### 2.2.7 한국어 특화

| 모델 | 파라미터 | 용도 |
|------|----------|------|
| **HybriKo-117M-LinuxFC-SFT-v2** | 117M | 한국어 Linux 명령어 생성 |
| LFM2.5-1.2B-JP | 1.2B | 일본어 특화 (참고용) |

**HybriKo 활용 이유:**
- 기웅님의 커스텀 모델로 한국어 Linux 명령어 100% 정확도
- Griffin-style Hybrid (RNN+Attention 2:1)

#### 2.2.8 범용 Agent (대안)

| 모델 | 파라미터 | 특징 |
|------|----------|------|
| **Youtu-LLM-2B** | 1.96B | 128K 컨텍스트, Native Agentic, MLA |

**특징:**
- Dense MLA(Multi-head Latent Attention) 아키텍처
- End-to-End Agent 작업 완수 능력

---

## 3. 메모리(VRAM) 견적

### 3.1 양자화 기준 메모리 사용량

모든 수치는 **llama.cpp GGUF** 기준입니다.

| 모델 | FP16 | Q8_0 | Q4_K_M |
|------|------|------|--------|
| LFM2.5-1.2B (Brain) | ~2.4GB | ~1.5GB | ~0.8GB |
| Falcon-H1-Tiny-Tool-Calling (90M) | ~180MB | ~100MB | ~60MB |
| Falcon-H1-Tiny-Coder (90M) | ~180MB | ~100MB | ~60MB |
| Falcon-H1-Tiny-R-0.6B | ~1.2GB | ~700MB | ~400MB |
| Falcon-H1-Tiny-Multilingual (100M) | ~200MB | ~110MB | ~65MB |
| HybriKo-117M | ~240MB | ~130MB | ~75MB |
| LightOnOCR-2-1B | ~2GB | ~1.2GB | ~0.7GB |

### 3.2 구성별 총 메모리 요구량

> **🖥️ PoC 타겟 환경:** Windows CPU-Only, 16GB RAM, Intel Core i5

#### PoC 권장 구성 (CPU-Only, 16GB RAM)
```
Brain:     LFM2.5-1.2B (Q4_K_M)        ~0.8GB
Reasoner:  Falcon-R-0.6B (Q4_K_M)      ~0.4GB  ← 코딩+수학 통합!
Tool:      Falcon-Tool-Calling (Q8_0)  ~0.1GB  (선택적)
KV Cache + OS 여유                      ~0.7GB
─────────────────────────────────────────
총계                                    ~2.0GB
```

**설계 근거:**
- **Falcon-R-0.6B = Coder + Reasoner 통합:** LiveCodeBench 39% + MATH500 94%
- **LFM2.5가 한국어 직접 처리:** 8개 언어 지원으로 Multilingual Specialist 불필요
- **Falcon-Coder(90M) 제외:** Reasoner가 코딩도 잘함

> **⚠️ Thinking vs Instruct 선택:**
> - `LFM2.5-1.2B-Thinking`: 더 긴 추론, 정확도 높을 수 있음
> - `LFM2.5-1.2B-Instruct`: 짧은 응답, 빠름
> - **→ PoC에서 실험 후 결정 예정** (CPU 환경에서 속도 vs 품질 트레이드오프 확인 필요)

**적합 환경:** Windows 16GB RAM (CPU-Only), 맥북 16GB 통합 메모리

---

#### 확장 구성 (GPU 보유 시)
```
PoC 권장 구성                            ~2.0GB
+ LightOnOCR-2-1B (Q4_K_M)              ~0.7GB  (문서 처리)
+ LFM2.5-Audio-1.5B (선택적)             ~1.0GB  (음성)
─────────────────────────────────────────
총계                                     ~3.7GB
```

**적합 환경:** 8GB+ VRAM GPU, 또는 32GB RAM 시스템

### 3.3 동적 로딩 전략

| 모델 유형 | 로딩 전략 | 이유 |
|-----------|-----------|------|
| Brain (LFM2.5-1.2B) | **상시 로딩** | 모든 요청의 진입점 |
| Reasoner (0.6B) | **상시 로딩** | 코딩+수학 빈번, Q4면 0.4GB로 가벼움 |
| Tool-Calling (90M) | **필요시 로딩** | API 호출 시에만 필요 |
| OCR (1B) | **Just-In-Time** | 문서 처리 시에만 필요, 무거움 |
| Audio (1.5B) | **Just-In-Time** | 음성 입출력 시에만 필요 |

---

## 4. 시스템 아키텍처

### 4.1 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 입력                                │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   🧠 Brain (LFM2.5-1.2B-Thinking)                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 1. 의도 분석 (Intent Recognition)                        │    │
│  │ 2. 작업 분해 (Task Decomposition)                        │    │
│  │ 3. 전문가 선택 (Router)                                  │    │
│  │ 4. 결과 통합 (Aggregator)                                │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🔧 Specialist Pool                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │🔨 Tool   │ │💻 Coder  │ │🤔 Reason │ │🌏 Multi  │           │
│  │  (90M)   │ │  (90M)   │ │ (600M)   │ │ (100M)   │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│  │📄 OCR    │ │🎙️ Audio  │ │🇰🇷 Ko    │                        │
│  │  (1B)    │ │ (1.5B)   │ │ (117M)   │                        │
│  └──────────┘ └──────────┘ └──────────┘                        │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     🧠 Brain (결과 통합)                         │
│  - 각 Specialist 출력 종합                                       │
│  - 일관성 검증                                                   │
│  - 최종 응답 생성                                                │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        최종 응답                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 라우팅 로직

Brain 모델이 다음 JSON 형식으로 전문가를 선택합니다:

```json
{
  "thinking": "사용자가 피보나치 함수를 요청했으므로 코딩 전문가에게 전달",
  "agents": [
    {
      "name": "CODER",
      "prompt": "Write a Python function to calculate the nth Fibonacci number",
      "priority": 1
    }
  ],
  "direct_answer": false
}
```

**라우팅 규칙:**

| 키워드/패턴 | 선택되는 Agent |
|-------------|----------------|
| 함수, 코드, 구현, def, class | CODER |
| 계산, 증명, 수학, 논리, AIME | REASONER |
| 검색, API, 호출, function call | TOOL_CALLER |
| 번역, 한국어, 일본어 | MULTILINGUAL |
| PDF, 문서, OCR, 이미지 텍스트 | OCR |
| 음성, 말해줘, TTS | AUDIO |
| Linux, 터미널, bash | HYBRIKO |

### 4.3 Specialist 프롬프트 템플릿

#### Tool-Calling Specialist
```xml
<tools>
[
  {"name": "search", "parameters": {"query": "string"}},
  {"name": "calculate", "parameters": {"expression": "string"}}
]
</tools>

User: {user_request}
Assistant:
```

#### Coder Specialist
```
You are a Python code generation assistant.
- Write clean, efficient code
- Include type hints
- Add docstrings

Task: {coding_task}

```python
```

---

## 5. 기술 스택

### 5.1 추론 프레임워크

| 프레임워크 | 지원 플랫폼 | 장점 |
|------------|-------------|------|
| **llama.cpp** | 모든 플랫폼 | GGUF 표준, 가장 넓은 호환성 |
| **MLX** | Apple Silicon | Metal 최적화, 통합 메모리 활용 |
| **ONNX** | 크로스 플랫폼 | NPU 지원 (AMD, Qualcomm) |
| vLLM | GPU 서버 | 고처리량 배치 추론 |

**PoC 권장:** `llama.cpp` (Python 바인딩 사용)

### 5.2 의존성

```toml
[project]
dependencies = [
    "llama-cpp-python>=0.3.0",
    "huggingface-hub>=0.25.0",
    "docling>=1.0.0",  # 문서 파싱
    "rich>=13.0.0",     # CLI 출력
    "pydantic>=2.0.0",  # 데이터 검증
]
```

---

## 6. PoC 구현 계획

### 6.1 Phase 1: 기본 오케스트레이터 (Week 1)

**목표:** Brain + 2개 Specialist 연동

1. **환경 설정**
   - llama.cpp Python 바인딩 설치
   - GGUF 모델 다운로드 스크립트 작성

2. **ModelPool 클래스 구현**
   ```python
   class ModelPool:
       def __init__(self, config: dict):
           self.brain = self._load_brain()
           self.specialists = self._load_specialists()

       def route_and_execute(self, user_input: str) -> str:
           plan = self.brain.plan(user_input)
           results = [self.specialists[a].execute(a.prompt) for a in plan.agents]
           return self.brain.aggregate(results)
   ```

3. **테스트 시나리오**
   - 단순 코딩: "피보나치 함수 작성해줘" → CODER
   - 수학: "123 + 456 = ?" → Brain 직접 또는 REASONER

### 6.2 Phase 2: 전체 Specialist 통합 (Week 2)

1. **추가 Specialist 연동**
   - Reasoning (0.6B)
   - Multilingual (100M)
   - HybriKo (117M)

2. **동적 로딩 구현**
   ```python
   class LazyModelLoader:
       def __getattr__(self, name):
           if name not in self._loaded:
               self._loaded[name] = self._load(name)
           return self._loaded[name]
   ```

3. **복합 시나리오 테스트**
   - "AIME 2024 문제 풀어줘" → REASONER
   - "이 코드를 한국어로 설명해줘" → CODER → MULTILINGUAL

### 6.3 Phase 3: 문서 처리 확장 (Week 3)

1. **docling 통합**
   - PDF → Markdown 변환
   - 이미지 추출

2. **LightOnOCR 동적 로딩**
   - 스캔 문서일 경우에만 로드
   - 수식 인식 필요시 활용

### 6.4 Phase 4: 최적화 및 벤치마크 (Week 4)

1. **성능 측정**
   - 응답 지연 시간
   - 메모리 사용량 추적
   - 정확도 평가

2. **최적화**
   - 배치 처리
   - KV Cache 공유 검토

---

## 7. 예상 성능

### 7.1 성능 비교 (예상)

| 시나리오 | 단일 4B 모델 | Tiny MoA (1.2B + 전문가) |
|----------|-------------|-------------------------|
| 코딩 작업 | 중상 | **상** (전문 Coder) |
| 수학 추론 | 중 | **상** (전문 Reasoner) |
| 도구 호출 | 중 | **상** (전문 Tool-Caller) |
| 일반 대화 | 상 | 중상 (Brain 단독) |
| 메모리 사용 | 8GB+ | **2~4GB** |
| 로딩 시간 | 10~20초 | **3~5초** |

### 7.2 한계점

1. **복잡한 멀티홉 추론:** Brain의 분해 능력에 의존
2. **지식 집약적 질문:** 1.2B Brain의 지식 한계 (RAG로 보완)
3. **Specialist 간 의존성:** 순차 호출 시 지연 누적

---

## 8. 확장 가능성

### 8.1 추가 가능한 Specialist

| 분야 | 후보 모델 | 비고 |
|------|-----------|------|
| 이미지 이해 | LFM2.5-VL-1.6B | Vision-Language |
| 법률/의료 | 도메인 SFT 모델 | 추가 파인튜닝 필요 |
| 게임/엔터테인먼트 | FunctionGemma Fine-tune | 앱 제어 |

### 8.2 스케일 업

- **클라우드 배포:** vLLM + GPU 클러스터
- **하이브리드:** 경량 모델 로컬 + 무거운 모델 클라우드
- **MCP 서버:** docling MCP 서버로 Agent 연결

---

## 9. 참고 자료

### 9.1 모델 링크

| 모델 | Hugging Face |
|------|--------------|
| LFM2.5-1.2B-Thinking | [LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) |
| LFM2.5-1.2B-Instruct | [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| Falcon-H1-Tiny-Tool-Calling | [tiiuae/Falcon-H1-Tiny-Tool-Calling-90M](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M) |
| Falcon-H1-Tiny-R-0.6B | [tiiuae/Falcon-H1-Tiny-R-0.6B](https://huggingface.co/tiiuae/Falcon-H1-Tiny-R-0.6B) |
| Falcon-H1-Tiny-Coder | [tiiuae/Falcon-H1-Tiny-Coder-90M](https://huggingface.co/collections/tiiuae/falcon-h1-tiny) |
| Falcon-H1-Tiny-Multilingual | [tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct) |
| LightOnOCR-2-1B | [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) |
| FunctionGemma | [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) |
| Youtu-LLM-2B | [tencent/Youtu-LLM-2B](https://huggingface.co/tencent/Youtu-LLM-2B) |
| HybriKo-117M | [Yaongi/HybriKo-117M-LinuxFC-SFT-v2](https://huggingface.co/Yaongi/HybriKo-117M-LinuxFC-SFT-v2) |
| docling | [GitHub](https://github.com/docling-project/docling) |

### 9.2 블로그 및 논문

- [Falcon-H1-Tiny 기술 블로그](https://huggingface.co/spaces/tiiuae/tiny-h1-blogpost)
- [LiquidAI LFM2.5 블로그](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- [Anti-curriculum 학습 전략](https://huggingface.co/spaces/tiiuae/tiny-h1-blogpost#falcon-h1-tiny-r-paving-the-way-for-a-new-pretraining-paradigm-for-reasoning-models)

---

## 10. 결론

**Tiny MoA**는 "GPU Poor" 환경에서도 고품질 AI 경험을 제공할 수 있는 현실적인 접근법입니다.

**핵심 메시지:**
1. **4B 단일 모델 < 1.2B Brain + 전문가 군단** (메모리 효율성)
2. **90M~600M 모델도 특화 학습으로 SoTA 달성** (Falcon-H1-Tiny 증명)
3. **llama.cpp + GGUF로 즉시 구현 가능** (생태계 성숙)

다음 단계로 `src/poc/` 디렉토리에서 PoC 구현을 시작할 수 있습니다.
