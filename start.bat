@echo off
:: ─────────────────────────────────────────────────────────────────
:: DIPEX — One-click startup script
:: ─────────────────────────────────────────────────────────────────
title DIPEX Startup

echo.
echo  ██████╗ ██╗██████╗ ███████╗██╗  ██╗
echo  ██╔══██╗██║██╔══██╗██╔════╝╚██╗██╔╝
echo  ██║  ██║██║██████╔╝█████╗   ╚███╔╝
echo  ██║  ██║██║██╔═══╝ ██╔══╝   ██╔██╗
echo  ██████╔╝██║██║     ███████╗██╔╝ ██╗
echo  ╚═════╝ ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝
echo  Data Intelligence Pipeline   v1.0
echo.
echo [1/4] Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)
echo       Docker OK.

echo.
echo [2/4] Pulling latest images (first run may take a few minutes)...
docker compose pull --quiet

echo.
echo [3/4] Starting all services...
docker compose up -d --build

if errorlevel 1 (
    echo ERROR: docker compose failed. Run "docker compose logs" to debug.
    pause
    exit /b 1
)

echo.
echo [4/4] Waiting for services to be healthy...
timeout /t 15 /nobreak >nul

echo.
echo ─────────────────────────────────────────
echo  DIPEX is running!
echo.
echo  Dashboard  → http://localhost:3000
echo  API        → http://localhost:8000
echo  API Docs   → http://localhost:8000/docs
echo  Kafka UI   → http://localhost:8080
echo ─────────────────────────────────────────
echo.

:: Open dashboard only
start "" http://localhost:3000

echo  Logs: docker compose logs -f
echo  Stop: docker compose down
echo.
