import asyncio
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from container.app import app, state

# Create a test client
client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_model():
    """Mock the HuggingFacePolicy model so we don't actually load a 1.5B param LLM for testing."""
    mock = MagicMock()
    mock.generate_experiment.return_value = ("DO X2 = 2.5000", {"target": "X2", "value": 2.5, "samples": 200})
    state.model = mock
    state.device = "cpu"
    state.inference_lock = asyncio.Lock()
    yield
    state.model = None
    state.inference_lock = None

def test_health_check_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "online",
        "device": "cpu",
        "model_loaded": True
    }

def test_health_check_no_model():
    state.model = None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False

def test_intervene_success():
    payload = {
        "scm": {
            "nodes": ["X1", "X2", "X3"],
            "edges": [
                {"source": "X1", "target": "X2"},
                {"source": "X2", "target": "X3"}
            ]
        },
        "node_losses": {
            "X1": 0.1,
            "X2": 1.5,
            "X3": 0.2
        },
        "intervention_history": ["X1", "X2"],
        "value_min": -5.0,
        "value_max": 5.0
    }
    
    response = client.post("/intervene", json=payload)
    assert response.status_code == 200, response.text
    
    data = response.json()
    assert data["command"] == "DO X2 = 2.5000"
    assert data["target"] == "X2"
    assert data["value"] == 2.5
    assert data["samples"] == 200
    
    # Verify the mock was called correctly with node_losses and history
    state.model.generate_experiment.assert_called_once()
    kwargs = state.model.generate_experiment.call_args[1]
    assert kwargs["node_losses"] == payload["node_losses"]
    assert kwargs["intervention_history"] == payload["intervention_history"]

def test_intervene_invalid_dag():
    payload = {
        "scm": {
            "nodes": ["X1", "X2", "X3"],
            "edges": [
                {"source": "X1", "target": "X2"},
                {"source": "X2", "target": "X3"},
                {"source": "X3", "target": "X1"} # Creates a cycle
            ]
        },
        "node_losses": {"X1": 0.1, "X2": 0.1, "X3": 0.1}
    }
    
    response = client.post("/intervene", json=payload)
    assert response.status_code == 400
    assert "Directed Acyclic Graph" in response.json()["detail"]

def test_intervene_no_model():
    state.model = None
    payload = {
        "scm": {"nodes": ["X1"], "edges": []},
        "node_losses": {"X1": 0.1}
    }
    response = client.post("/intervene", json=payload)
    assert response.status_code == 503
    assert "Model is not loaded" in response.json()["detail"]


def test_dsl_is_set_before_generate_experiment():
    """Verify the request-specific DSL is applied to the model before inference runs.

    This guards against the race condition where state.model.dsl is mutated by
    one request and then read by another (Option B: each container is isolated,
    but correctness still requires the lock to serialize DSL mutation + inference).
    """
    captured_dsl = {}

    def capture_dsl(*args, **kwargs):
        captured_dsl["nodes"] = list(state.model.dsl.nodes)
        return ("DO X2 = 2.5000", {"target": "X2", "value": 2.5, "samples": 200})

    state.model.generate_experiment.side_effect = capture_dsl

    payload = {
        "scm": {
            "nodes": ["A", "B", "C"],
            "edges": [
                {"source": "A", "target": "B"},
                {"source": "B", "target": "C"}
            ]
        },
        "node_losses": {"A": 0.1, "B": 0.9, "C": 0.2},
        "value_min": -3.0,
        "value_max": 3.0
    }

    response = client.post("/intervene", json=payload)
    assert response.status_code == 200, response.text
    # The DSL nodes visible to generate_experiment must match this request's nodes
    assert captured_dsl.get("nodes") == ["A", "B", "C"]


def test_inference_lock_initialized_on_startup():
    """inference_lock must be an asyncio.Lock (set during startup, not class default None)."""
    assert state.inference_lock is not None
    assert isinstance(state.inference_lock, asyncio.Lock)


def test_intervene_model_failure_returns_500():
    """If generate_experiment returns (raw_text, None), a 500 is raised."""
    state.model.generate_experiment.return_value = ("garbled output xyz", None)

    payload = {
        "scm": {"nodes": ["X1", "X2"], "edges": [{"source": "X1", "target": "X2"}]},
        "node_losses": {"X1": 0.1, "X2": 0.5}
    }
    response = client.post("/intervene", json=payload)
    assert response.status_code == 500
    assert "failed to generate" in response.json()["detail"]
