<#
.SYNOPSIS
    Deploy the ACE container API to Azure Container Apps.

.DESCRIPTION
    Handles the full deploy lifecycle: build, push, create/update container app.
    Supports deploying multiple isolated instances for parallel experiments.

.PARAMETER Action
    One of: setup | build | deploy | scale | instances | teardown | logs | status

.PARAMETER InstanceName
    Optional. Override the default app name for isolated instance deploys.

.PARAMETER Replicas
    Number of replicas when scaling (default: 1).

.PARAMETER HfToken
    HuggingFace API token. Falls back to $env:HF_TOKEN if not provided.

.PARAMETER ImageTag
    Docker image tag to build/push/deploy (default: "latest").
    Use "git" to automatically use the current git short SHA.

.EXAMPLE
    # First-time setup (run once per subscription)
    .\scripts\deploy_azure.ps1 -Action setup

.EXAMPLE
    # Build and push image, then deploy/update the default app
    .\scripts\deploy_azure.ps1 -Action build
    .\scripts\deploy_azure.ps1 -Action deploy

.EXAMPLE
    # Deploy 3 isolated instances for parallel seeds
    .\scripts\deploy_azure.ps1 -Action instances -Replicas 3

.EXAMPLE
    # Stream live logs
    .\scripts\deploy_azure.ps1 -Action logs

.EXAMPLE
    # Tear down everything (deletes resource group — irreversible)
    .\scripts\deploy_azure.ps1 -Action teardown
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("setup", "build", "deploy", "scale", "instances", "teardown", "logs", "status")]
    [string]$Action,

    [string]$InstanceName = "",
    [int]$Replicas = 1,
    [string]$HfToken = "",
    [string]$ImageTag = "latest"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ============================================================
# CONFIGURATION — edit these to match your Azure environment
# ============================================================
$RESOURCE_GROUP   = "ace-rg"
$LOCATION         = "eastus"
$ACR_NAME         = "aceregistry"          # Must be globally unique, alphanumeric only
$ACA_ENV_NAME     = "ace-env"
$APP_NAME         = "ace-api"
$GPU_PROFILE_NAME = "gpu-t4"
$GPU_PROFILE_TYPE = "NC4as_T4_v3"         # T4 GPU. For A100: "NC24ads_A100_v4"
$CPU_CORES        = "4.0"
$MEMORY_GI        = "16.0Gi"
# ============================================================

function Get-ImageName {
    param([string]$Tag)
    if ($Tag -eq "git") {
        $sha = git rev-parse --short HEAD 2>$null
        if (-not $sha) { throw "Not inside a git repository or git not found." }
        $Tag = $sha
    }
    return "$ACR_NAME.azurecr.io/ace-api:$Tag"
}

function Require-AzCli {
    if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
        Write-Error "Azure CLI not found. Install with: winget install --exact --id Microsoft.AzureCLI"
        exit 1
    }
}

function Require-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker not found. Install Docker Desktop from https://www.docker.com/products/docker-desktop"
        exit 1
    }
}

function Get-AcrPassword {
    return (az acr credential show `
        --name $ACR_NAME `
        --query "passwords[0].value" `
        --output tsv)
}

function Get-AppFqdn {
    param([string]$Name = $APP_NAME)
    return (az containerapp show `
        --name $Name `
        --resource-group $RESOURCE_GROUP `
        --query "properties.configuration.ingress.fqdn" `
        --output tsv 2>$null)
}

# ============================================================
# ACTION: setup — provision all Azure infrastructure (run once)
# ============================================================
function Invoke-Setup {
    Require-AzCli

    Write-Host "`n[1/4] Creating resource group '$RESOURCE_GROUP' in '$LOCATION'..."
    az group create `
        --name $RESOURCE_GROUP `
        --location $LOCATION `
        --output none

    Write-Host "[2/4] Creating Azure Container Registry '$ACR_NAME'..."
    az acr create `
        --resource-group $RESOURCE_GROUP `
        --name $ACR_NAME `
        --sku Basic `
        --admin-enabled true `
        --output none

    Write-Host "[3/4] Creating Container Apps Environment '$ACA_ENV_NAME'..."
    az containerapp env create `
        --name $ACA_ENV_NAME `
        --resource-group $RESOURCE_GROUP `
        --location $LOCATION `
        --enable-workload-profiles `
        --output none

    Write-Host "[4/4] Adding GPU workload profile '$GPU_PROFILE_NAME' ($GPU_PROFILE_TYPE)..."
    az containerapp env workload-profile add `
        --name $ACA_ENV_NAME `
        --resource-group $RESOURCE_GROUP `
        --workload-profile-name $GPU_PROFILE_NAME `
        --workload-profile-type $GPU_PROFILE_TYPE `
        --min-nodes 0 `
        --max-nodes 5 `
        --output none

    Write-Host "`nSetup complete."
    Write-Host "ACR login server: $(az acr show --name $ACR_NAME --query loginServer --output tsv)"
}

# ============================================================
# ACTION: build — build Docker image and push to ACR
# ============================================================
function Invoke-Build {
    Require-AzCli
    Require-Docker

    $image = Get-ImageName -Tag $ImageTag

    Write-Host "`n[1/3] Logging in to ACR '$ACR_NAME'..."
    az acr login --name $ACR_NAME

    # Confirm we are at the repo root (Dockerfile uses repo-root build context)
    if (-not (Test-Path "container/Dockerfile")) {
        throw "Run this script from the ACE repository root directory."
    }

    Write-Host "[2/3] Building image '$image'..."
    docker build --tag $image --file container/Dockerfile .

    Write-Host "[3/3] Pushing '$image' to ACR..."
    docker push $image

    Write-Host "`nImage pushed: $image"
}

# ============================================================
# ACTION: deploy — create or update the container app
# ============================================================
function Invoke-Deploy {
    param([string]$Name = $APP_NAME)
    Require-AzCli

    $image    = Get-ImageName -Tag $ImageTag
    $token    = if ($HfToken) { $HfToken } else { $env:HF_TOKEN }
    $password = Get-AcrPassword

    $envVars = "PYTHONUNBUFFERED=1"
    if ($token) { $envVars += " HF_TOKEN=$token" }

    # Check if app already exists
    $exists = az containerapp show `
        --name $Name `
        --resource-group $RESOURCE_GROUP `
        --query "name" `
        --output tsv 2>$null

    if ($exists) {
        Write-Host "`nUpdating existing container app '$Name' with image '$image'..."
        az containerapp update `
            --name $Name `
            --resource-group $RESOURCE_GROUP `
            --image $image `
            --output none
    } else {
        Write-Host "`nCreating container app '$Name' with image '$image'..."
        az containerapp create `
            --name $Name `
            --resource-group $RESOURCE_GROUP `
            --environment $ACA_ENV_NAME `
            --workload-profile-name $GPU_PROFILE_NAME `
            --image $image `
            --target-port 8000 `
            --ingress external `
            --registry-server "$ACR_NAME.azurecr.io" `
            --registry-username $ACR_NAME `
            --registry-password $password `
            --env-vars $envVars `
            --cpu $CPU_CORES `
            --memory $MEMORY_GI `
            --min-replicas 0 `
            --max-replicas 1 `
            --output none
    }

    $fqdn = Get-AppFqdn -Name $Name
    Write-Host "`nDeployed: https://$fqdn"
    Write-Host "Health:   https://$fqdn/health"
}

# ============================================================
# ACTION: scale — set replica count on the default app
# ============================================================
function Invoke-Scale {
    Require-AzCli
    Write-Host "`nScaling '$APP_NAME' to $Replicas replica(s)..."
    az containerapp update `
        --name $APP_NAME `
        --resource-group $RESOURCE_GROUP `
        --min-replicas $Replicas `
        --max-replicas $Replicas `
        --output none
    Write-Host "Done. Current replicas: $Replicas"
}

# ============================================================
# ACTION: instances — deploy N isolated named instances
# ============================================================
function Invoke-Instances {
    Require-AzCli

    $names = @()
    for ($i = 1; $i -le $Replicas; $i++) {
        $names += "ace-instance-$i"
    }

    Write-Host "`nDeploying $($names.Count) isolated ACE instance(s)..."
    $endpoints = @{}

    foreach ($name in $names) {
        Write-Host "`n  -> Deploying '$name'..."
        Invoke-Deploy -Name $name
        $endpoints[$name] = "https://$(Get-AppFqdn -Name $name)"
    }

    Write-Host "`nAll instances deployed:"
    $endpoints.GetEnumerator() | ForEach-Object {
        Write-Host "  $($_.Key): $($_.Value)"
    }

    Write-Host "`nTo tear down all instances:"
    $list = ($names | ForEach-Object { "`"$_`"" }) -join ", "
    Write-Host "  @($list) | ForEach-Object { az containerapp delete --name `$_ --resource-group $RESOURCE_GROUP --yes }"
}

# ============================================================
# ACTION: status — show running apps and their endpoints
# ============================================================
function Invoke-Status {
    Require-AzCli
    Write-Host "`nContainer Apps in resource group '$RESOURCE_GROUP':"
    az containerapp list `
        --resource-group $RESOURCE_GROUP `
        --query "[].{Name:name, FQDN:properties.configuration.ingress.fqdn, Replicas:properties.template.scale.minReplicas}" `
        --output table

    $fqdn = Get-AppFqdn -Name $APP_NAME
    if ($fqdn) {
        Write-Host "`nHealth check for '$APP_NAME':"
        try {
            $resp = Invoke-RestMethod -Uri "https://$fqdn/health" -Method Get -TimeoutSec 10
            $resp | ConvertTo-Json
        } catch {
            Write-Host "  Could not reach /health — container may still be starting."
        }
    }
}

# ============================================================
# ACTION: logs — stream live logs from the default app
# ============================================================
function Invoke-Logs {
    Require-AzCli
    $name = if ($InstanceName) { $InstanceName } else { $APP_NAME }
    Write-Host "`nStreaming logs from '$name' (Ctrl+C to stop)..."
    az containerapp logs show `
        --name $name `
        --resource-group $RESOURCE_GROUP `
        --follow `
        --tail 50
}

# ============================================================
# ACTION: teardown — delete the entire resource group (irreversible)
# ============================================================
function Invoke-Teardown {
    Require-AzCli
    Write-Warning "This will DELETE resource group '$RESOURCE_GROUP' and ALL resources inside it."
    $confirm = Read-Host "Type the resource group name to confirm"
    if ($confirm -ne $RESOURCE_GROUP) {
        Write-Host "Teardown cancelled."
        return
    }
    Write-Host "Deleting resource group '$RESOURCE_GROUP'..."
    az group delete --name $RESOURCE_GROUP --yes --no-wait
    Write-Host "Deletion initiated (runs in background, takes ~5 minutes)."
}

# ============================================================
# Dispatch
# ============================================================
switch ($Action) {
    "setup"     { Invoke-Setup }
    "build"     { Invoke-Build }
    "deploy"    { if ($InstanceName) { Invoke-Deploy -Name $InstanceName } else { Invoke-Deploy } }
    "scale"     { Invoke-Scale }
    "instances" { Invoke-Instances }
    "status"    { Invoke-Status }
    "logs"      { Invoke-Logs }
    "teardown"  { Invoke-Teardown }
}
