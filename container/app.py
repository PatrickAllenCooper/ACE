import asyncio
import logging
import os
import torch
import networkx as nx
from typing import List, Dict, Optional, Any
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Ensure we can import from the parent directory where ACE core files are
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace_experiments import HuggingFacePolicy, ExperimentalDSL, StudentSCM, CausalModel

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ace-api")

app = FastAPI(
    title="Active Causal Experimentalist (ACE) API",
    description="Inference API to generate interventions for Causal Discovery using DPO",
    version="1.0.0"
)

# ----------------------------------------------------------------
# API Key Authentication
# ----------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(key: str = Security(_api_key_header)):
    expected = os.environ.get("API_KEY")
    if not expected:
        # No key configured — fail closed, do not allow unauthenticated access
        raise HTTPException(status_code=503, detail="API key not configured on server.")
    if key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return key

# ----------------------------------------------------------------
# Pydantic Models for the API
# ----------------------------------------------------------------

class EdgeDef(BaseModel):
    source: str
    target: str

class SCMDefinition(BaseModel):
    nodes: List[str] = Field(..., description="List of node names, e.g., ['X1', 'X2', 'X3']")
    edges: List[EdgeDef] = Field(..., description="List of directed edges in the DAG")

class InterventionRequest(BaseModel):
    scm: SCMDefinition
    node_losses: Dict[str, float] = Field(..., description="Current evaluation loss for each node mechanism")
    intervention_history: Optional[List[str]] = Field(default=None, description="History of past intervened nodes (helps prevent getting stuck)")
    value_min: Optional[float] = -5.0
    value_max: Optional[float] = 5.0

class InterventionResponse(BaseModel):
    command: str
    target: str
    value: float
    samples: int

# ----------------------------------------------------------------
# Global State for the Model
# ----------------------------------------------------------------

class AppState:
    device: str = "cpu"
    model: HuggingFacePolicy = None
    # Serializes DSL mutation + inference so concurrent requests to the same
    # replica cannot corrupt each other's graph context (state.model.dsl is shared).
    inference_lock: asyncio.Lock = None

state = AppState()

# ----------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up ACE API server...")
    state.inference_lock = asyncio.Lock()
    state.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {state.device}")
    
    # Define a default DSL just to initialize the policy (we'll create specific ones per request)
    # The HuggingFacePolicy only really needs the vocabulary and value ranges from the DSL.
    try:
        # NOTE: We use a placeholder DSL here just to load the model.
        # The true nodes are injected at request time.
        placeholder_dsl = ExperimentalDSL(nodes=["X1", "X2"])
        
        # We use the same model as in experiments
        model_name = "Qwen/Qwen2.5-1.5B"
        hf_token = os.environ.get("HF_TOKEN")
        
        logger.info(f"Loading base LLM policy ({model_name})...")
        state.model = HuggingFacePolicy(
            model_name=model_name,
            dsl=placeholder_dsl,
            device=state.device,
            token=hf_token
        )
        logger.info("LLM Policy loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load LLM during startup: {e}")
        # We don't crash here so the /health endpoint can report the issue if needed


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "device": state.device,
        "model_loaded": state.model is not None
    }


@app.post("/intervene", response_model=InterventionResponse, dependencies=[Depends(require_api_key)])
async def generate_intervention(req: InterventionRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or still initializing.")
        
    try:
        # 1. Reconstruct the SCM Graph from the request
        edges = [(edge.source, edge.target) for edge in req.scm.edges]
        
        # Verify DAG properties
        graph = nx.DiGraph(edges)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Provided SCM edges do not form a Directed Acyclic Graph (DAG).")

        # Create a dummy base model to hold the structure
        class RequestCausalModel(CausalModel):
            def __init__(self):
                super().__init__(edges)
        
        base_scm = RequestCausalModel()
        
        # Create StudentSCM to hold the parameters (used by the policy prompt in base ACE)
        student_scm = StudentSCM(base_scm)
        student_scm.to(state.device)
        
        # 2. Build Request-Specific DSL
        request_dsl = ExperimentalDSL(
            nodes=req.scm.nodes,
            value_min=req.value_min,
            value_max=req.value_max
        )

        logger.info(f"Generating intervention for graph with {len(req.scm.nodes)} nodes and {len(edges)} edges.")

        # 3. Generate Experiment
        # Acquire lock before mutating the shared DSL on the model so concurrent
        # requests to this replica cannot overwrite each other's graph context.
        async with state.inference_lock:
            state.model.dsl = request_dsl
            cmd_str, plan = state.model.generate_experiment(
                scm_student=student_scm,
                node_losses=req.node_losses,
                intervention_history=req.intervention_history,
                max_new_tokens=64
            )
        
        if plan is None:
             raise HTTPException(status_code=500, detail=f"Model failed to generate a valid intervention. Raw output: {cmd_str}")

        return InterventionResponse(
            command=cmd_str,
            target=plan["target"],
            value=plan["value"],
            samples=plan.get("samples", 200)
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))
