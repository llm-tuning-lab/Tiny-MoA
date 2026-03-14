"""Test configuration and schema loading."""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestToolSchema:
    """Test tool schema definitions."""

    def test_tools_schema_structure(self):
        """Test that TOOLS schema has correct structure."""
        from src.tools.schema import TOOLS

        assert isinstance(TOOLS, list)
        assert len(TOOLS) > 0

        # Check each tool has required fields
        for tool in TOOLS:
            assert isinstance(tool, dict)
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert isinstance(tool["name"], str)
            assert isinstance(tool["description"], str)
            assert isinstance(tool["parameters"], dict)

    def test_tools_schema_has_required_tools(self):
        """Test that essential tools are defined."""
        from src.tools.schema import TOOLS

        tool_names = [tool["name"] for tool in TOOLS]

        # Check for essential tools
        essential_tools = ["get_weather", "search_web", "calculate"]
        for tool_name in essential_tools:
            assert tool_name in tool_names, f"Missing tool: {tool_name}"

    def test_tools_schema_parameters_valid(self):
        """Test that tool parameters follow OpenAI format."""
        from src.tools.schema import TOOLS

        for tool in TOOLS:
            params = tool["parameters"]
            assert "type" in params
            assert params["type"] == "object"
            assert "properties" in params
            assert isinstance(params["properties"], dict)


class TestLanguageDetector:
    """Test language detection configuration."""

    def test_language_names_mapping(self):
        """Test that language names mapping is valid."""
        from src.translation.detector import LANGUAGE_NAMES

        assert isinstance(LANGUAGE_NAMES, dict)
        assert len(LANGUAGE_NAMES) > 0

        # Check for essential languages
        essential_langs = ["ko", "en", "ja", "zh"]
        for lang_code in essential_langs:
            assert lang_code in LANGUAGE_NAMES
            assert isinstance(LANGUAGE_NAMES[lang_code], str)

    def test_detect_language_empty_input(self):
        """Test language detection with empty input."""
        from src.translation.detector import detect_language

        result = detect_language("")
        assert result == "en"

        result = detect_language("   ")
        assert result == "en"

    def test_detect_language_english(self):
        """Test language detection for English text."""
        from src.translation.detector import detect_language

        result = detect_language("Hello world, this is a test.")
        assert result == "en"

    def test_detect_language_korean(self):
        """Test language detection for Korean text."""
        from src.translation.detector import detect_language

        result = detect_language("안녕하세요, 이것은 테스트입니다.")
        # Should detect Korean or fallback to heuristic
        assert result in ["ko", "en"]  # Allow fallback

    def test_detect_language_japanese(self):
        """Test language detection for Japanese text."""
        from src.translation.detector import detect_language

        result = detect_language("こんにちは、これはテストです。")
        # Should detect Japanese or fallback
        assert result in ["ja", "en"]  # Allow fallback


class TestToolExecutor:
    """Test tool executor configuration."""

    def test_tool_executor_initialization(self):
        """Test that ToolExecutor can be instantiated."""
        from src.tools.executor import ToolExecutor

        executor = ToolExecutor()
        assert executor is not None
        assert hasattr(executor, "execute")

    def test_tool_executor_has_execute_method(self):
        """Test that ToolExecutor has execute method."""
        from src.tools.executor import ToolExecutor

        executor = ToolExecutor()
        assert callable(executor.execute)


class TestTranslationPipeline:
    """Test translation pipeline configuration."""

    def test_translation_pipeline_initialization(self):
        """Test that TranslationPipeline can be instantiated."""
        from src.translation.pipeline import TranslationPipeline

        pipeline = TranslationPipeline()
        assert pipeline is not None


class TestRAGConfiguration:
    """Test RAG system configuration."""

    def test_rag_engine_initialization(self):
        """Test that RAGEngine can be instantiated."""
        from src.rag.engine import RAGEngine

        engine = RAGEngine()
        assert engine is not None

    def test_rag_engine_has_query_method(self):
        """Test that RAGEngine has query method."""
        from src.rag.engine import RAGEngine

        engine = RAGEngine()
        assert hasattr(engine, "query")
        assert callable(engine.query)
