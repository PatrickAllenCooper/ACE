import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from container.app import app, state

# Create a test client
client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_model():
    """Mock the HuggingFacePolicy model so we don't actually load a 1.5B param LLM for testing"""
    mock = MagicMock()
    # Mock generation method to return a predictable valid command
    mock.generate_experiment.return_value = ("DO X2 = 2.5000", {"target": "X2", "value": 2.5, "samples": 200})
    state.model = mock
    state.device = "cpu"
    yield
    # Cleanup
    state.model = None

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
