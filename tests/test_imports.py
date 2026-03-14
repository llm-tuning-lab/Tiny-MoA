"""Test that all src modules can be imported without errors."""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestImports:
    """Test importability of all core modules."""

    def test_import_tools_schema(self):
        """Test tools.schema module imports."""
        from src.tools import schema

        assert hasattr(schema, "TOOLS")
        assert isinstance(schema.TOOLS, list)
        assert len(schema.TOOLS) > 0

    def test_import_tools_executor(self):
        """Test tools.executor module imports."""
        from src.tools import executor

        assert hasattr(executor, "ToolExecutor")

    def test_import_tools_caller(self):
        """Test tools.caller module imports."""
        from src.tools.caller import ToolCaller

        assert ToolCaller is not None

    def test_import_translation_detector(self):
        """Test translation.detector module imports."""
        from src.translation import detector

        assert hasattr(detector, "detect_language")
        assert hasattr(detector, "LANGUAGE_NAMES")

    def test_import_translation_pipeline(self):
        """Test translation.pipeline module imports."""
        from src.translation import pipeline

        assert hasattr(pipeline, "TranslationPipeline")

    def test_import_rag_engine(self):
        """Test rag.engine module imports."""
        from src.rag import engine

        assert hasattr(engine, "RAGEngine")

    def test_import_doc_processing_converter(self):
        """Test doc_processing.converter module imports."""
        from src.doc_processing import converter

        assert hasattr(converter, "DocumentConverter")

    def test_import_tiny_moa_brain(self):
        """Test tiny_moa.brain module imports."""
        from src.tiny_moa.brain import Brain

        assert Brain is not None

    def test_import_tiny_moa_reasoner(self):
        """Test tiny_moa.reasoner module imports."""
        from src.tiny_moa.reasoner import Reasoner

        assert Reasoner is not None

    def test_import_tiny_moa_cowork_task_queue(self):
        """Test tiny_moa.cowork.task_queue module imports."""
        from src.tiny_moa.cowork import task_queue

        assert hasattr(task_queue, "TaskQueue")

    def test_import_tiny_moa_cowork_workers_base(self):
        """Test tiny_moa.cowork.workers.base module imports."""
        from src.tiny_moa.cowork.workers import base

        assert hasattr(base, "BaseWorker")

    def test_import_tiny_moa_cowork_workers_brain_worker(self):
        """Test tiny_moa.cowork.workers.brain_worker module imports."""
        from src.tiny_moa.cowork.workers import brain_worker

        assert hasattr(brain_worker, "BrainWorker")

    def test_import_tiny_moa_cowork_workers_tool_worker(self):
        """Test tiny_moa.cowork.workers.tool_worker module imports."""
        from src.tiny_moa.cowork.workers import tool_worker

        assert hasattr(tool_worker, "ToolWorker")
