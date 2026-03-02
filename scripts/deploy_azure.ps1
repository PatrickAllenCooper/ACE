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
    # Deploy 3 isolated instances for parallel seeds (recommended multi-instance pattern)
    .\scripts\deploy_azure.ps1 -Action instances -Replicas 3

.EXAMPLE
    # Stream live logs from a specific named instance
    .\scripts\deploy_azure.ps1 -Action logs -InstanceName ace-instance-2

.EXAMPLE
    # Stream live logs from the default app
    .\scripts\deploy_azure.ps1 -Action logs

.EXAMPLE
    # Tear down everything (deletes resource group - irreversible)
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
# CONFIGURATION - edit these to match your Azure environment
# ============================================================
$RESOURCE_GROUP   = "ace-rg"
$LOCATION         = "eastus"
$ACR_NAME         = "aceregistrypcooper"    # Must be globally unique, alphanumeric only
$ACA_ENV_NAME     = "ace-env"
$APP_NAME         = "ace-api"
$GPU_PROFILE_NAME = "gpu-t4"                      # Name used when referencing the profile
$GPU_PROFILE_TYPE = "Consumption-GPU-NC8as-T4"   # SKU type. For A100: Consumption-GPU-NC24-A100
$CPU_CORES        = "8.0"
$MEMORY_GI        = "56.0Gi"
$API_KEY          = $env:ACE_API_KEY   # Set this in your shell: $env:ACE_API_KEY = "..."
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
    try {
        return (az containerapp show `
            --name $Name `
            --resource-group $RESOURCE_GROUP `
            --query "properties.configuration.ingress.fqdn" `
            --output tsv 2>$null)
    } catch { return $null }
}

# ============================================================
# ACTION: setup - provision all Azure infrastructure (run once)
# ============================================================
function Invoke-Setup {
    Require-AzCli

    Write-Host "`n[0/4] Registering required resource providers (fires in background, ~1-2 min)..."
    foreach ($provider in @("Microsoft.ContainerRegistry", "Microsoft.App", "Microsoft.OperationalInsights", "Microsoft.ContainerService")) {
        az provider register --namespace $provider --output none
        Write-Host "  Queued: $provider"
    }
    Write-Host "  Waiting for providers to reach Registered state..."
    $required = @("Microsoft.ContainerRegistry", "Microsoft.App", "Microsoft.OperationalInsights")
    do {
        Start-Sleep -Seconds 10
        $states = $required | ForEach-Object {
            az provider show --namespace $_ --query registrationState --output tsv
        }
        $pending = @($states | Where-Object { $_ -ne "Registered" }).Count
        Write-Host "  $pending provider(s) still registering..."
    } while ($pending -gt 0)
    Write-Host "  All providers registered."

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

    Write-Host "[4/4] Adding GPU workload profile '$GPU_PROFILE_NAME' (type: $GPU_PROFILE_TYPE)..."
    az containerapp env workload-profile add `
        --name $ACA_ENV_NAME `
        --resource-group $RESOURCE_GROUP `
        --workload-profile-name $GPU_PROFILE_NAME `
        --workload-profile-type $GPU_PROFILE_TYPE `
        --output none
    Write-Host "  GPU profile '$GPU_PROFILE_NAME' added."

    Write-Host "`nSetup complete."
    Write-Host "ACR login server: $(az acr show --name $ACR_NAME --query loginServer --output tsv)"
}

# ============================================================
# ACTION: build - build image in ACR (no local Docker required)
# Uses ACR Tasks: the build runs in Azure, output goes to your registry.
# ============================================================
function Invoke-Build {
    Require-AzCli

    $tag = if ($ImageTag -eq "git") { git rev-parse --short HEAD } else { $ImageTag }

    # Confirm we are at the repo root (Dockerfile uses repo-root build context)
    if (-not (Test-Path "container/Dockerfile")) {
        throw "Run this script from the ACE repository root directory."
    }

    Write-Host "`nBuilding 'ace-api:$tag' via ACR Tasks (runs in Azure, no local Docker needed)..."
    Write-Host "This takes 10-20 minutes on first build due to the PyTorch base image."
    Write-Host "Streaming build logs:"

    az acr build `
        --registry $ACR_NAME `
        --image "ace-api:$tag" `
        --file container/Dockerfile `
        .

    Write-Host "`nImage built and pushed: $ACR_NAME.azurecr.io/ace-api:$tag"
}

# ============================================================
# ACTION: deploy - create or update the container app
# ============================================================
function Invoke-Deploy {
    param([string]$Name = $APP_NAME)
    Require-AzCli

    $image    = Get-ImageName -Tag $ImageTag
    $token    = if ($HfToken) { $HfToken } else { $env:HF_TOKEN }
    $password = Get-AcrPassword

    $envVars = "PYTHONUNBUFFERED=1"
    if ($token) { $envVars += " HF_TOKEN=$token" }
    if ($API_KEY) { $envVars += " API_KEY=$API_KEY" }

    # Check if app already exists (suppress errors - not found is expected on first deploy)
    $exists = $null
    try {
        $exists = az containerapp show `
            --name $Name `
            --resource-group $RESOURCE_GROUP `
            --query "name" `
            --output tsv 2>$null
    } catch { $exists = $null }

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
# ACTION: scale - set replica count on the default app
# NOTE: Scaling a single app to multiple replicas is NOT the recommended
# pattern for ACE. The model DSL is shared global state and the asyncio.Lock
# only serializes within a single replica. Use "instances" instead.
# This action is retained for cases where the core library is updated to
# accept DSL as a per-call argument.
# ============================================================
function Invoke-Scale {
    Require-AzCli
    if ($Replicas -gt 1) {
        Write-Warning "Scaling to multiple replicas is unsafe with the current shared DSL design."
        Write-Warning "Use '-Action instances -Replicas $Replicas' for parallel experiments instead."
        $confirm = Read-Host "Continue anyway? (y/N)"
        if ($confirm -ne "y") { return }
    }
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
# ACTION: instances - deploy N isolated named instances
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
    $list = ($names | ForEach-Object { '"{0}"' -f $_ }) -join ", "
    Write-Host "  @($list) | ForEach-Object { az containerapp delete --name `$_ --resource-group $RESOURCE_GROUP --yes }"
}

# ============================================================
# ACTION: status - show running apps and their endpoints
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
            Write-Host "  Could not reach /health - container may still be starting."
        }
    }
}

# ============================================================
# ACTION: logs - stream live logs from the default app
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
# ACTION: teardown - delete the entire resource group (irreversible)
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
