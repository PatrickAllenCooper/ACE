# Deploying ACE to Azure Container Apps

This guide covers everything required to build, push, and deploy the containerized ACE API to Azure Container Apps (ACA) — including multi-instance deployment for scaling arbitrary parallel experiments.

---

## Prerequisites

### 1. Install Azure CLI (Windows)

```powershell
# Via winget (Windows 10/11 recommended)
winget install --exact --id Microsoft.AzureCLI

# Or via MSI installer — download from:
# https://aka.ms/installazurecliwindows
# Then restart your terminal.

# Verify installation
az version
```

### 2. Install Docker Desktop

Docker Desktop must be running for all local build steps.
Download from: https://www.docker.com/products/docker-desktop

### 3. Log in to Azure

```powershell
az login
# A browser will open — authenticate with your Microsoft account.

# Confirm your active subscription
az account show --output table

# If you have multiple subscriptions, select the correct one:
az account set --subscription "<SUBSCRIPTION_ID_OR_NAME>"
```

---

## Configuration Variables

All commands below use these variables. Set them once in your PowerShell session before running anything:

```powershell
# === EDIT THESE ===
$SUBSCRIPTION_ID   = "<your-azure-subscription-id>"
$RESOURCE_GROUP    = "ace-rg"
$LOCATION          = "eastus"          # Must support ACA GPU workload profiles
$ACR_NAME          = "aceregistry"     # Must be globally unique, alphanumeric only
$ACA_ENV_NAME      = "ace-env"
$APP_NAME          = "ace-api"
$GPU_PROFILE_NAME  = "gpu-t4"
$GPU_PROFILE_TYPE  = "NC4as_T4_v3"    # T4 GPU — cheapest option with good throughput
# For A100: "NC24ads_A100_v4"

# Image tag
$IMAGE_TAG         = "latest"
$FULL_IMAGE        = "$ACR_NAME.azurecr.io/ace-api:$IMAGE_TAG"
```

---

## Part 1: One-Time Azure Infrastructure Setup

Run these once to provision the registry and container app environment.

### 1a. Create Resource Group

```powershell
az group create `
  --name $RESOURCE_GROUP `
  --location $LOCATION
```

### 1b. Create Azure Container Registry (ACR)

```powershell
az acr create `
  --resource-group $RESOURCE_GROUP `
  --name $ACR_NAME `
  --sku Basic `
  --admin-enabled true
```

Retrieve the ACR login server for reference:

```powershell
az acr show --name $ACR_NAME --query loginServer --output tsv
# Expected: aceregistry.azurecr.io
```

### 1c. Create the Container Apps Environment with GPU Workload Profiles

```powershell
az containerapp env create `
  --name $ACA_ENV_NAME `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --enable-workload-profiles
```

### 1d. Add the GPU Workload Profile

This reserves GPU capacity in the environment. `--min-nodes 0` enables scale-to-zero when idle (cost-saving).

```powershell
az containerapp env workload-profile add `
  --name $ACA_ENV_NAME `
  --resource-group $RESOURCE_GROUP `
  --workload-profile-name $GPU_PROFILE_NAME `
  --workload-profile-type $GPU_PROFILE_TYPE `
  --min-nodes 0 `
  --max-nodes 3
```

> **Region note:** GPU workload profiles in ACA are available in `eastus`, `westus3`, `northeurope`, and select others. Check availability at: https://learn.microsoft.com/en-us/azure/container-apps/workload-profiles-overview

---

## Part 2: Build and Push the Docker Image

Run from the **ACE repository root**.

### 2a. Log in to ACR

```powershell
az acr login --name $ACR_NAME
```

### 2b. Build the Image

```powershell
# From the repo root — the Dockerfile expects this context
docker build `
  --tag $FULL_IMAGE `
  --file container/Dockerfile `
  .
```

### 2c. Push to ACR

```powershell
docker push $FULL_IMAGE
```

Verify the image landed:

```powershell
az acr repository list --name $ACR_NAME --output table
az acr repository show-tags --name $ACR_NAME --repository ace-api --output table
```

---

## Part 3: Deploy the Container App

### 3a. Initial Deployment

```powershell
# Retrieve the ACR admin credentials
$ACR_PASSWORD = az acr credential show `
  --name $ACR_NAME `
  --query "passwords[0].value" `
  --output tsv

az containerapp create `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --environment $ACA_ENV_NAME `
  --workload-profile-name $GPU_PROFILE_NAME `
  --image $FULL_IMAGE `
  --target-port 8000 `
  --ingress external `
  --registry-server "$ACR_NAME.azurecr.io" `
  --registry-username $ACR_NAME `
  --registry-password $ACR_PASSWORD `
  --env-vars "PYTHONUNBUFFERED=1" "HF_TOKEN=<YOUR_HF_TOKEN>" `
  --cpu 4.0 `
  --memory 16.0Gi `
  --min-replicas 0 `
  --max-replicas 1
```

> The `HF_TOKEN` env var is used by the startup code in `container/app.py`. `Qwen/Qwen2.5-1.5B` is public but a token avoids rate limits.

### 3b. Get the Deployed App URL

```powershell
$FQDN = az containerapp show `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --query "properties.configuration.ingress.fqdn" `
  --output tsv

Write-Host "ACE API is live at: https://$FQDN"
```

### 3c. Test the Deployment

```powershell
# Health check
Invoke-RestMethod -Uri "https://$FQDN/health" -Method Get | ConvertTo-Json

# Generate an intervention (example 3-node chain graph)
$body = @{
    scm = @{
        nodes = @("X1", "X2", "X3")
        edges = @(
            @{ source = "X1"; target = "X2" }
            @{ source = "X2"; target = "X3" }
        )
    }
    node_losses = @{
        X1 = 0.05
        X2 = 1.20
        X3 = 0.10
    }
    intervention_history = @("X1", "X1")
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "https://$FQDN/intervene" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json
```

---

## Part 4: Deploying Multiple Parallel Instances

For running multiple independent ACE experiments in parallel (e.g., different seeds, different graph topologies), use **Option B: isolated named apps**. This is the recommended pattern for ACE.

### Why Not Option A (Scaled Replicas)?

Option A scales a single app to N replicas and load-balances requests across them. It is not safe for ACE in its current form. The `/intervene` handler mutates `state.model.dsl` — a shared global — before every inference call. Two simultaneous requests routed to the same replica would overwrite each other's graph context, producing silently wrong interventions. The `asyncio.Lock` in `app.py` prevents this within a single replica, but ACA's load balancer can route concurrent requests to any replica, and each replica has its own lock with no cross-replica coordination.

Option A would be correct only if `HuggingFacePolicy.generate_experiment` accepted the DSL as an argument rather than reading from `self.dsl`. That is a future improvement to the core library.

### Option B: Separate Named App Per Experiment (Recommended)

Deploy fully independent container apps — one per experiment. Each instance has its own model, its own DSL, its own logs, and can be torn down independently when the experiment is done.

```powershell
function Deploy-AceInstance {
    param(
        [string]$InstanceName,
        [string]$HfToken,
        [int]$MaxReplicas = 1
    )

    $ACR_PASSWORD = az acr credential show `
        --name $ACR_NAME `
        --query "passwords[0].value" `
        --output tsv

    az containerapp create `
        --name $InstanceName `
        --resource-group $RESOURCE_GROUP `
        --environment $ACA_ENV_NAME `
        --workload-profile-name $GPU_PROFILE_NAME `
        --image $FULL_IMAGE `
        --target-port 8000 `
        --ingress external `
        --registry-server "$ACR_NAME.azurecr.io" `
        --registry-username $ACR_NAME `
        --registry-password $ACR_PASSWORD `
        --env-vars "PYTHONUNBUFFERED=1" "HF_TOKEN=$HfToken" `
        --cpu 4.0 `
        --memory 16.0Gi `
        --min-replicas 0 `
        --max-replicas $MaxReplicas

    $fqdn = az containerapp show `
        --name $InstanceName `
        --resource-group $RESOURCE_GROUP `
        --query "properties.configuration.ingress.fqdn" `
        --output tsv

    Write-Host "[$InstanceName] Live at: https://$fqdn"
    return $fqdn
}

# Deploy 3 isolated instances
$hfToken = "<YOUR_HF_TOKEN>"
Deploy-AceInstance -InstanceName "ace-exp-seed42"  -HfToken $hfToken
Deploy-AceInstance -InstanceName "ace-exp-seed123" -HfToken $hfToken
Deploy-AceInstance -InstanceName "ace-exp-seed456" -HfToken $hfToken
```

Tear them all down when experiments complete:

```powershell
@("ace-exp-seed42", "ace-exp-seed123", "ace-exp-seed456") | ForEach-Object {
    az containerapp delete --name $_ --resource-group $RESOURCE_GROUP --yes
}
```

---

## Part 5: Updating the Deployment After Code Changes

When `ace_experiments.py` or `container/app.py` changes:

```powershell
# 1. Rebuild and push updated image
docker build --tag $FULL_IMAGE --file container/Dockerfile .
docker push $FULL_IMAGE

# 2. Force ACA to pull the new image (rolling restart)
az containerapp update `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --image $FULL_IMAGE
```

For versioned deploys (recommended for reproducibility), tag by git SHA:

```powershell
$GIT_SHA = git rev-parse --short HEAD
$VERSIONED_IMAGE = "$ACR_NAME.azurecr.io/ace-api:$GIT_SHA"

docker build --tag $VERSIONED_IMAGE --file container/Dockerfile .
docker push $VERSIONED_IMAGE

az containerapp update `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --image $VERSIONED_IMAGE
```

---

## Part 6: Monitoring and Logs

### Live Log Streaming

```powershell
az containerapp logs show `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --follow `
  --tail 50
```

### Revision History

ACA tracks every deployment as an immutable revision. List them:

```powershell
az containerapp revision list `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --output table
```

Roll back to a previous revision:

```powershell
az containerapp revision activate `
  --revision <REVISION_NAME> `
  --resource-group $RESOURCE_GROUP
```

### Container Metrics (CPU, Memory, GPU)

```powershell
# Open Azure Monitor metrics in browser
az monitor metrics list `
  --resource $(az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query id --output tsv) `
  --metric "CpuUsageNanoCores" `
  --output table
```

---

## Part 7: Cost Control

| GPU SKU | vCPUs | RAM | Approx. Cost |
|---|---|---|---|
| `NC4as_T4_v3` | 4 | 28 GB | ~$0.50/hr per replica |
| `NC24ads_A100_v4` | 24 | 220 GB | ~$3.60/hr per replica |

Scale-to-zero (`--min-replicas 0`) means you pay nothing when no requests are in-flight. The model re-loads on the first request after scale-down (~60-90 seconds cold start for Qwen2.5-1.5B).

Delete all resources when done:

```powershell
az group delete --name $RESOURCE_GROUP --yes --no-wait
```

---

## Troubleshooting

**Container fails to start (503 on /health):**
- The LLM model load takes 60-90 seconds after container start. Poll `/health` until `model_loaded: true`.
- Check logs: `az containerapp logs show --name $APP_NAME --resource-group $RESOURCE_GROUP --tail 100`

**ACR authentication error during `containerapp create`:**
- Ensure `--admin-enabled true` was set on the ACR.
- Re-fetch the password: `az acr credential show --name $ACR_NAME`

**GPU not available inside container (`device: cpu` in /health):**
- Verify the `--workload-profile-name` matches the GPU profile added to the environment.
- GPU profiles require the `NC*` SKU family — `Consumption` profiles are CPU-only.

**`az containerapp env workload-profile add` fails with quota error:**
- GPU quota for ACA must be requested separately from VM quota. File a support ticket in the Azure portal: Help + Support > New Support Request > Quota > Container Apps GPU.
