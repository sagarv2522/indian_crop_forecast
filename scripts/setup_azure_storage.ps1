# scripts/setup_azure_storage.ps1
# Idempotent Azure Blob Storage provisioning script for Crop Price Forecasting Platform
# Designed for Azure for Students (5 GB Standard_LRS free tier)

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$ResourceGroup = "rg-crop-forecasting",

    [Parameter(Mandatory = $false)]
    [string]$Location = "centralindia",

    [Parameter(Mandatory = $false)]
    [string]$StorageAccountName,

    [Parameter(Mandatory = $false)]
    [string[]]$Containers = @("crops-data", "crops-raw", "crops-models", "crops-monitoring"),

    [Parameter(Mandatory = $false)]
    [switch]$UpdateEnv = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    if (Test-Path "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd") {
        $env:Path = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin;" + $env:Path
    }
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Azure Blob Storage Setup (Crop Forecasting Platform)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verify az CLI installation
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Error "Azure CLI ('az') is not installed or not in PATH.`nPlease install it using: winget install -e --id Microsoft.AzureCLI`nAfter installation, restart your terminal and run 'az login'."
    exit 1
}

# 2. Verify az login and Azure for Students subscription
Write-Host "`n[1/5] Checking Azure authentication & subscription..." -ForegroundColor Yellow
$accountJson = az account show --output json 2>$null
if (-not $accountJson) {
    Write-Host "Not signed in to Azure CLI. Launching 'az login'..." -ForegroundColor Yellow
    az login --output none
    $accountJson = az account show --output json
}

$currentAccount = $accountJson | ConvertFrom-Json
Write-Host "  Active Subscription Name : $($currentAccount.name)" -ForegroundColor Green
Write-Host "  Subscription ID          : $($currentAccount.id)" -ForegroundColor Green

# Check if active subscription matches Azure for Students
if ($currentAccount.name -notmatch "Students") {
    Write-Warning "Active subscription does not contain 'Students'. Looking for 'Azure for Students' in available subscriptions..."
    $subList = az account list --output json | ConvertFrom-Json
    $studentSub = $subList | Where-Object { $_.name -match "Students" } | Select-Object -First 1
    if ($studentSub) {
        Write-Host "  Switching to subscription: $($studentSub.name) ($($studentSub.id))..." -ForegroundColor Cyan
        az account set --subscription $studentSub.id
    } else {
        Write-Warning "No subscription with 'Students' in the name found. Proceeding with active subscription: $($currentAccount.name)"
    }
}

# 3. Create Resource Group (Idempotent)
Write-Host "`n[2/5] Ensuring Resource Group '$ResourceGroup' in '$Location'..." -ForegroundColor Yellow
az group create --name $ResourceGroup --location $Location --output none
Write-Host "  Resource Group '$ResourceGroup' is ready." -ForegroundColor Green

# 4. Generate Storage Account Name if not specified
if (-not $StorageAccountName) {
    # Generate unique 3-24 char lowercase alphanumeric name
    $randomSuffix = -join ((97..122) | Get-Random -Count 6 | ForEach-Object { [char]$_ })
    $StorageAccountName = "cropstorage$randomSuffix"
} else {
    $StorageAccountName = $StorageAccountName.ToLower().Trim()
}

Write-Host "`n[3/5] Checking/Creating Storage Account '$StorageAccountName' (Standard_LRS, StorageV2)..." -ForegroundColor Yellow
$saExists = az storage account check-name --name $StorageAccountName --query "nameAvailable" --output tsv

if ($saExists -eq "true") {
    Write-Host "  Creating Storage Account '$StorageAccountName'..." -ForegroundColor Cyan
    az storage account create `
        --name $StorageAccountName `
        --resource-group $ResourceGroup `
        --location $Location `
        --sku Standard_LRS `
        --kind StorageV2 `
        --min-tls-version TLS1_2 `
        --allow-blob-public-access false `
        --output none
    Write-Host "  Storage Account created successfully." -ForegroundColor Green
} else {
    # Check if it belongs to our RG
    $existing = az storage account show --name $StorageAccountName --resource-group $ResourceGroup --output json 2>$null
    if ($existing) {
        Write-Host "  Storage Account '$StorageAccountName' already exists in '$ResourceGroup'. Reusing." -ForegroundColor Green
    } else {
        Write-Error "Storage Account name '$StorageAccountName' is taken globally. Please provide a different name via -StorageAccountName."
        exit 1
    }
}

# 5. Retrieve Storage Keys & Create Containers
Write-Host "`n[4/5] Creating Blob Containers..." -ForegroundColor Yellow
$keysJson = az storage account keys list --account-name $StorageAccountName --resource-group $ResourceGroup --output json
$keys = $keysJson | ConvertFrom-Json
$primaryKey = $keys[0].value

foreach ($container in $Containers) {
    $cExists = az storage container exists --account-name $StorageAccountName --account-key $primaryKey --name $container --query "exists" --output tsv
    if ($cExists -ne "true") {
        Write-Host "  Creating container: '$container'..." -ForegroundColor Cyan
        az storage container create --account-name $StorageAccountName --account-key $primaryKey --name $container --output none
    } else {
        Write-Host "  Container '$container' already exists." -ForegroundColor Green
    }
}

# 6. Generate 1-Year Container SAS Token (Read + List) for DuckDB
Write-Host "`n[5/5] Generating SAS Token for 'crops-data'..." -ForegroundColor Yellow
$expiryDate = (Get-Date).AddYears(1).ToString("yyyy-MM-ddTHH:mm:ssZ")
$sasToken = az storage container generate-sas `
    --account-name $StorageAccountName `
    --account-key $primaryKey `
    --name "crops-data" `
    --permissions rl `
    --expiry $expiryDate `
    --https-only `
    --output tsv

$blobBaseUrl = "https://$StorageAccountName.blob.core.windows.net/crops-data"
$connString = az storage account show-connection-string --name $StorageAccountName --resource-group $ResourceGroup --query "connectionString" --output tsv

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  PROVISIONING SUCCESSFUL!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Storage Account Name : $StorageAccountName"
Write-Host "  Resource Group       : $ResourceGroup"
Write-Host "  Location             : $Location"
Write-Host "  Blob Base URL        : $blobBaseUrl"
Write-Host "  Containers Created   : $($Containers -join ', ')"

if ($UpdateEnv) {
    $envPath = Join-Path (Get-Item $PSScriptRoot).Parent.FullName ".env"
    if (Test-Path $envPath) {
        Write-Host "`nUpdating local .env file..." -ForegroundColor Cyan
        $envContent = Get-Content -Path $envPath -Raw

        # Update or add AZURE_STORAGE_ACCOUNT
        if ($envContent -match "AZURE_STORAGE_ACCOUNT=") {
            $envContent = $envContent -replace 'AZURE_STORAGE_ACCOUNT=.*', "AZURE_STORAGE_ACCOUNT=$StorageAccountName"
        } else {
            $envContent += "`nAZURE_STORAGE_ACCOUNT=$StorageAccountName"
        }

        # Update or add AZURE_BLOB_BASE_URL
        if ($envContent -match "AZURE_BLOB_BASE_URL=") {
            $envContent = $envContent -replace 'AZURE_BLOB_BASE_URL=.*', "AZURE_BLOB_BASE_URL=$blobBaseUrl"
        } else {
            $envContent += "`nAZURE_BLOB_BASE_URL=$blobBaseUrl"
        }

        # Update or add AZURE_SAS_TOKEN
        if ($envContent -match "AZURE_SAS_TOKEN=") {
            $envContent = $envContent -replace 'AZURE_SAS_TOKEN=.*', "AZURE_SAS_TOKEN=$sasToken"
        } else {
            $envContent += "`nAZURE_SAS_TOKEN=$sasToken"
        }

        # Update or add AZURE_STORAGE_CONNECTION_STRING
        if ($envContent -match "AZURE_STORAGE_CONNECTION_STRING=") {
            $envContent = $envContent -replace 'AZURE_STORAGE_CONNECTION_STRING=.*', "AZURE_STORAGE_CONNECTION_STRING=$connString"
        } else {
            $envContent += "`nAZURE_STORAGE_CONNECTION_STRING=$connString"
        }

        Set-Content -Path $envPath -Value $envContent.Trim() -Encoding utf8
        Write-Host "  Updated $envPath successfully." -ForegroundColor Green
    }
}

Write-Host "`nNote: Credentials have been written directly to your local .env (untracked)." -ForegroundColor Yellow
Write-Host "Setup completed.`n"
