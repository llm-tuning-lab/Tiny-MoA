# Tiny-MoA — AGENTS.md

**Generated:** 2026-03-13
**Project:** Tiny MoA v2.1 — CPU-based Mixture of Agents (LFM2.5-1.2B + Falcon-R 600M + Tool Caller 90M)
**Stack:** Python 3.10+ / llama-cpp-python / Rich TUI / Pydantic

---

## OVERVIEW

Proof-of-concept Mixture of Agents system designed for CPU-only environments (~2GB total memory). A 1.2B "Brain" model (LFM2.5-1.2B-Thinking) plans and routes tasks to a 600M Reasoner (Falcon-R) and 90M Tool Caller (Falcon-Tool-Calling). Supports multilingual input via English-first translation strategy, TUI visualization, RAG with ChromaDB, and web search via DuckDuckGo.

---

## STRUCTURE

```
Tiny-MoA/
  pyproject.toml           # Package config (hatchling build, ruff)
  uv.lock                  # uv lockfile
  requirements.txt         # pip fallback dependencies
  README.md                # Korean documentation
  README_EN.md             # English documentation
  LICENSE                  # Apache 2.0
  docs/                    # Plans, roadmaps, vision docs
  scripts/                 # Utility scripts
  src/
    tiny_moa/              # Main package
      main.py              # Entry point (CLI args, interactive mode)
      orchestrator.py      # Central controller (intent → route → respond)
      brain.py             # LFM2.5-1.2B-Thinking wrapper
      reasoner.py          # Falcon-R-0.6B wrapper
      cowork/              # Agentic workflow
        planner.py         # Task planning
        workspace.py       # File system access
        workers/           # Specialized worker agents
      ui/                  # Rich TUI components
    tools/                 # Tool use (search, weather, system)
      executor.py          # Tool execution engine
      schema.py            # Tool definitions
    translation/           # Multilingual translation pipeline
    doc_processing/        # Document conversion (Docling)
      converter.py
    rag/                   # RAG engine + vector store
      engine.py
      store.py
```

---

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Run the system | `src/tiny_moa/main.py` | CLI entry point, `--thinking`, `--tui`, `--interactive` flags |
| Routing logic | `src/tiny_moa/orchestrator.py` | Routes to TOOL / REASONER / DIRECT |
| Brain (thinking) | `src/tiny_moa/brain.py` | LFM2.5-1.2B-Thinking via llama-cpp |
| Reasoner | `src/tiny_moa/reasoner.py` | Falcon-R 600M for coding/math |
| Tool calling | `src/tools/executor.py` | Weather, search, file ops, system info |
| Translation | `src/translation/` | Language detect → English → process → translate back |
| RAG | `src/rag/engine.py` | ChromaDB vector store + sentence-transformers |
| Document processing | `src/doc_processing/converter.py` | Docling-based PDF/DOCX conversion |
| TUI | `src/tiny_moa/ui/` | Rich-based task board visualization |
| Agentic workflow | `src/tiny_moa/cowork/` | Multi-step planning and execution |

---

## CONVENTIONS

**Line length:** 100 chars (ruff)
**Target Python:** 3.10+
**Lint rules:** E, F, I (E501 ignored)
**Package manager:** uv (preferred), pip fallback
**Build system:** hatchling
**Model format:** GGUF via llama-cpp-python
**Language strategy:** English-first (translate input → process in English → translate output)

---

## ANTI-PATTERNS

- Do NOT hardcode model paths — use configurable model directory
- Do NOT assume GPU availability — this is a CPU-only project
- Do NOT add large dependencies without justification — keep memory footprint low
- Do NOT bypass the translation pipeline for non-English input
- Do NOT use `# type: ignore` or `cast()` — fix type errors properly

---

## COMMANDS

```bash
# Run with TUI and thinking mode
uv run python -m tiny_moa.main --thinking --show-thinking --tui --query "your query"

# Interactive mode
uv run python -m tiny_moa.main --interactive

# With file reference (RAG)
uv run python -m tiny_moa.main --tui --query "@[document.pdf] summarize this"

# Lint
make lint

# Format
make format

# Clean
make clean
```

---

## NOTES

- **Total memory:** ~2GB (Brain 0.8GB + Reasoner 0.4GB + Tool Caller 0.1GB)
- **Models:** Download GGUF files to `./models/` directory via `huggingface-cli`
- **No tests directory:** Project has no test suite currently
- **Wheel packages:** `src/tiny_moa`, `src/tools`, `src/translation` (hatch config)
- **RAG storage:** ChromaDB persists to `./rag_storage/`
- **Translation:** Uses deep-translator + kiwipiepy (Korean morphological analysis)
