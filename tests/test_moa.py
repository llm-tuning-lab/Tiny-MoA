"""Test core MoA (Mixture of Agents) logic with mocks."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestBrainModule:
    """Test Brain (thinking model) module."""

    @patch("src.tiny_moa.brain.Llama")
    def test_brain_initialization(self, mock_llama):
        """Test that Brain can be initialized with mocked model."""
        from src.tiny_moa.brain import Brain

        mock_llama.return_value = MagicMock()
        brain = Brain(model_path="mock_model.gguf")
        assert brain is not None

    @patch("src.tiny_moa.brain.Llama")
    def test_brain_has_route_method(self, mock_llama):
        """Test that Brain has route method."""
        from src.tiny_moa.brain import Brain

        mock_llama.return_value = MagicMock()
        brain = Brain(model_path="mock_model.gguf")
        assert hasattr(brain, "route")
        assert callable(brain.route)

    @patch("src.tiny_moa.brain.Llama")
    def test_brain_has_direct_respond_method(self, mock_llama):
        """Test that Brain has direct_respond method."""
        from src.tiny_moa.brain import Brain

        mock_llama.return_value = MagicMock()
        brain = Brain(model_path="mock_model.gguf")
        assert hasattr(brain, "direct_respond")
        assert callable(brain.direct_respond)


class TestReasonerModule:
    """Test Reasoner module."""

    @patch("src.tiny_moa.reasoner.Llama")
    def test_reasoner_initialization(self, mock_llama):
        """Test that Reasoner can be initialized with mocked model."""
        from src.tiny_moa.reasoner import Reasoner

        mock_llama.return_value = MagicMock()
        reasoner = Reasoner(model_path="mock_model.gguf")
        assert reasoner is not None

    @patch("src.tiny_moa.reasoner.Llama")
    def test_reasoner_has_solve_method(self, mock_llama):
        """Test that Reasoner has solve method."""
        from src.tiny_moa.reasoner import Reasoner

        mock_llama.return_value = MagicMock()
        reasoner = Reasoner(model_path="mock_model.gguf")
        assert hasattr(reasoner, "solve")
        assert callable(reasoner.solve)

    @patch("src.tiny_moa.reasoner.Llama")
    def test_reasoner_has_code_method(self, mock_llama):
        """Test that Reasoner has code method."""
        from src.tiny_moa.reasoner import Reasoner

        mock_llama.return_value = MagicMock()
        reasoner = Reasoner(model_path="mock_model.gguf")
        assert hasattr(reasoner, "code")
        assert callable(reasoner.code)


class TestToolCaller:
    """Test tool calling functionality."""

    def test_tool_caller_initialization(self):
        """Test that ToolCaller can be instantiated."""
        from src.tools.caller import ToolCaller

        caller = ToolCaller()
        assert caller is not None

    def test_tool_caller_has_generate_tool_call_method(self):
        """Test that ToolCaller has generate_tool_call method."""
        from src.tools.caller import ToolCaller

        caller = ToolCaller()
        assert hasattr(caller, "generate_tool_call")
        assert callable(caller.generate_tool_call)


class TestTaskQueue:
    """Test task queue module."""

    def test_task_queue_initialization(self):
        """Test that TaskQueue can be instantiated."""
        from src.tiny_moa.cowork.task_queue import TaskQueue

        queue = TaskQueue()
        assert queue is not None

    def test_task_queue_has_add_task_method(self):
        """Test that TaskQueue has add_task method."""
        from src.tiny_moa.cowork.task_queue import TaskQueue

        queue = TaskQueue()
        assert hasattr(queue, "add_task")
        assert callable(queue.add_task)

    def test_task_queue_has_get_pending_tasks_method(self):
        """Test that TaskQueue has get_pending_tasks method."""
        from src.tiny_moa.cowork.task_queue import TaskQueue

        queue = TaskQueue()
        assert hasattr(queue, "get_pending_tasks")
        assert callable(queue.get_pending_tasks)

    def test_task_queue_add_task_cycle(self):
        """Test add_task and get_pending_tasks operations."""
        from src.tiny_moa.cowork.task_queue import TaskQueue

        queue = TaskQueue()
        task = queue.add_task("test task", "brain")

        assert task is not None
        assert task.description == "test task"

        pending = queue.get_pending_tasks()
        assert len(pending) > 0


class TestBaseWorker:
    """Test base worker class."""

    def test_base_worker_has_execute_method(self):
        """Test that BaseWorker has execute method."""
        from src.tiny_moa.cowork.workers.base import BaseWorker

        # BaseWorker is abstract, so we check the class definition
        assert hasattr(BaseWorker, "execute")


class TestBrainWorker:
    """Test brain worker (specialized agent)."""

    def test_brain_worker_initialization(self):
        """Test that BrainWorker can be instantiated."""
        from src.tiny_moa.cowork.workers.brain_worker import BrainWorker
        import logging

        # Mock dependencies
        mock_brain = MagicMock()
        logger = logging.getLogger("test")

        worker = BrainWorker(name="test_brain", logger=logger, brain=mock_brain)
        assert worker is not None

    def test_brain_worker_has_execute_method(self):
        """Test that BrainWorker has execute method."""
        from src.tiny_moa.cowork.workers.brain_worker import BrainWorker
        import logging

        mock_brain = MagicMock()
        logger = logging.getLogger("test")

        worker = BrainWorker(name="test_brain", logger=logger, brain=mock_brain)
        assert hasattr(worker, "execute")
        assert callable(worker.execute)


class TestToolWorker:
    """Test tool worker (specialized agent)."""

    def test_tool_worker_initialization(self):
        """Test that ToolWorker can be instantiated."""
        from src.tiny_moa.cowork.workers.tool_worker import ToolWorker
        import logging

        mock_orchestrator = MagicMock()
        logger = logging.getLogger("test")

        worker = ToolWorker(name="test_tool", logger=logger, orchestrator=mock_orchestrator)
        assert worker is not None

    def test_tool_worker_has_execute_method(self):
        """Test that ToolWorker has execute method."""
        from src.tiny_moa.cowork.workers.tool_worker import ToolWorker
        import logging

        mock_orchestrator = MagicMock()
        logger = logging.getLogger("test")

        worker = ToolWorker(name="test_tool", logger=logger, orchestrator=mock_orchestrator)
        assert hasattr(worker, "execute")
        assert callable(worker.execute)
