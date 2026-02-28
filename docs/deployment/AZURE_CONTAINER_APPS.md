# Deploying ACE to Azure Container Apps

This guide outlines how to deploy the containerized version of ACE (Active Causal Experimentalist) to Azure Container Apps (ACA) with GPU support.

## Prerequisites
- Azure CLI installed (`az login`)
- An active Azure Subscription
- Docker Desktop installed globally

## Part 1: Build and Push Docker Image

1. **Log in to Azure Container Registry (ACR)**
   Create an ACR if you don't have one:
   ```bash
   az acr create --resource-group myResourceGroup --name myACRRegistry --sku Basic
   ```
   Log in to the registry:
   ```bash
   az acr login --name myACRRegistry
   ```

2. **Build and Tag the Image**
   From the ACE repository root directory:
   ```bash
   cd /path/to/ACE
   docker build -t myACRRegistry.azurecr.io/ace-api:latest -f container/Dockerfile .
   ```

3. **Push to ACR**
   ```bash
   docker push myACRRegistry.azurecr.io/ace-api:latest
   ```

## Part 2: Deploy to Azure Container Apps with GPU Support

Azure Container Apps supports GPU workloads via dedicated workload profiles. You must create an environment with a workload profile that supports GPUs (e.g., `NC A100 v4` series).

1. **Create the Environment with a GPU Workload Profile**
   ```bash
   az containerapp env create \
     --name ace-gpu-env \
     --resource-group myResourceGroup \
     --location eastus \
     --enable-workload-profiles
   ```
   *(Note: Ensure your selected region supports GPU workload profiles in ACA).*

2. **Add a GPU Workload Profile**
   Reserve a GPU profile for your environment (e.g., `NC4as_T4_v3` for a T4 GPU, or A100 variants if you have quota):
   ```bash
   az containerapp env workload-profile add \
     --name ace-gpu-env \
     --resource-group myResourceGroup \
     --workload-profile-name gpu-profile \
     --workload-profile-type NC4as_T4_v3 \
     --min-nodes 0 \
     --max-nodes 1
   ```

3. **Deploy the Container App**
   ```bash
   az containerapp create \
     --name ace-api \
     --resource-group myResourceGroup \
     --environment ace-gpu-env \
     --workload-profile-name gpu-profile \
     --image myACRRegistry.azurecr.io/ace-api:latest \
     --target-port 8000 \
     --ingress external \
     --registry-server myACRRegistry.azurecr.io \
     --env-vars "HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN_HERE>" "PYTHONUNBUFFERED=1" \
     --cpu 4.0 \
     --memory 16.0Gi
   ```
   *The HuggingFace token is required because `Qwen/Qwen2.5-1.5B` might require authentication or is best pulled with a valid token.*

## Part 3: Using the API

Once deployed, query your container's FQDN (Fully Qualified Domain Name).

**Health Check**
```bash
curl https://<YOUR_APP_FQDN>/health
```

**Generate an Intervention**
Send an `InterventionRequest` defining the current state of a graph:

```bash
curl -X POST "https://<YOUR_APP_FQDN>/intervene" \
     -H "Content-Type: application/json" \
     -d '{
           "scm": {
             "nodes": ["X1", "X2", "X3"],
             "edges": [
               {"source": "X1", "target": "X2"},
               {"source": "X2", "target": "X3"}
             ]
           },
           "node_losses": {
             "X1": 0.05,
             "X2": 0.80,
             "X3": 1.50
           },
           "intervention_history": ["X1", "X1", "X3"]
         }'
```

**Expected Response**:
```json
{
  "command": "DO X2 = 1.3410",
  "target": "X2",
  "value": 1.341,
  "samples": 200
}
```
