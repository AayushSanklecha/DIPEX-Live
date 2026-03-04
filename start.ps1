# ============================================================
#  DIPEX — One-Click Full Stack Launcher
#  Double-click start.bat OR run:  powershell -File start.ps1
# ============================================================

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $projectDir

$banner = @"

  ██████╗ ██╗██████╗ ███████╗██╗  ██╗
  ██╔══██╗██║██╔══██╗██╔════╝╚██╗██╔╝
  ██║  ██║██║██████╔╝█████╗   ╚███╔╝
  ██║  ██║██║██╔═══╝ ██╔══╝   ██╔██╗
  ██████╔╝██║██║     ███████╗██╔╝ ██╗
  ╚═════╝ ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝
  Data Intelligence Platform — Full Stack Launcher
"@

Write-Host $banner -ForegroundColor Cyan

# ── Preflight: auto-start Docker Desktop if not running ───────────────────────
Write-Host "`n[1/4] Checking Docker..." -ForegroundColor Yellow

$dockerReady = $false
docker info 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    $dockerReady = $true
    Write-Host "      Docker is running ✓" -ForegroundColor Green
}
else {
    Write-Host "      Docker Desktop is not running — starting it now..." -ForegroundColor Yellow

    # Common Docker Desktop install paths
    $ddPaths = @(
        "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
        "$env:LOCALAPPDATA\Docker\Docker Desktop.exe"
    )
    $dd = $ddPaths | Where-Object { Test-Path $_ } | Select-Object -First 1

    if ($dd) {
        Start-Process $dd
        Write-Host "      Waiting for Docker Desktop to be ready (up to 90s)..." -ForegroundColor DarkGray
        $waited = 0
        while ($waited -lt 90) {
            Start-Sleep -Seconds 5
            $waited += 5
            docker info 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { $dockerReady = $true; break }
            Write-Host "      ($waited s) waiting for Docker engine..." -ForegroundColor DarkGray
        }
    }

    if ($dockerReady) {
        Write-Host "      Docker is ready ✓" -ForegroundColor Green
    }
    else {
        Write-Host ""
        Write-Host "  ╔══════════════════════════════════════════════════╗" -ForegroundColor Red
        Write-Host "  ║  Docker Desktop could not be started.            ║" -ForegroundColor Red
        Write-Host "  ║  Please open Docker Desktop manually, wait for   ║" -ForegroundColor Red
        Write-Host "  ║  it to say 'Engine running', then try again.     ║" -ForegroundColor Red
        Write-Host "  ╚══════════════════════════════════════════════════╝" -ForegroundColor Red
        Read-Host "`nPress Enter to exit"
        exit 1
    }
}

# ── Pull / build images ───────────────────────────────────────────────────────
Write-Host "`n[2/4] Building & pulling images (first run may take a few minutes)..." -ForegroundColor Yellow
docker compose pull --ignore-buildable 2>&1 | Out-Null
docker compose build --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "      ERROR: docker compose build failed." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "      Images ready ✓" -ForegroundColor Green

# ── Start all services ────────────────────────────────────────────────────────
Write-Host "`n[3/4] Starting all services..." -ForegroundColor Yellow
docker compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "      ERROR: docker compose up failed." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "      All containers started ✓" -ForegroundColor Green

# ── Wait for the API to become healthy ────────────────────────────────────────
Write-Host "`n[4/4] Waiting for DIPEX API to be ready..." -ForegroundColor Yellow
$maxWait = 90   # seconds
$elapsed = 0
$ready = $false

while ($elapsed -lt $maxWait) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    }
    catch { }
    Start-Sleep -Seconds 3
    $elapsed += 3
    Write-Host "      ($elapsed s) still starting..." -ForegroundColor DarkGray
}

if (-not $ready) {
    Write-Host "      API did not become healthy in ${maxWait}s — check logs with:  docker compose logs dipex-api" -ForegroundColor Red
}
else {
    Write-Host "      API is healthy ✓" -ForegroundColor Green
}

# ── Print summary & open browser tabs ─────────────────────────────────────────
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  DIPEX is running!  Opening browser..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Dashboard URL:    http://localhost:3000" -ForegroundColor Green
Write-Host ""
Write-Host "  To stop everything:  docker compose down" -ForegroundColor DarkYellow
Write-Host "  To view logs:        docker compose logs -f" -ForegroundColor DarkYellow
Write-Host ""

# Open browser tabs in Chrome
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path $chrome)) {
    $chrome = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
}
Start-Sleep -Seconds 2
if (Test-Path $chrome) {
    Start-Process $chrome "http://localhost:3000"
}
else {
    Start-Process "http://localhost:3000"
    Write-Host "  (Chrome not found at default path — opened in default browser)" -ForegroundColor DarkYellow
}

Write-Host "============================================================" -ForegroundColor Cyan
Read-Host "`nPress Enter to exit this window (containers keep running)"
