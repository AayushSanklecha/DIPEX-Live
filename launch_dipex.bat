@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

:: ════════════════════════════════════════════════════════════════════
::  DIPEX — Launch Script  (v2 — Updated 2026-04)
::  Starts the full DIPEX stack via Docker Compose:
::    • PostgreSQL  (port 5433)
::    • MongoDB     (port 27018)
::    • Zookeeper   (internal)
::    • Kafka       (port 9092)
::    • Kafka UI    (port 8080)
::    • DIPEX API   (port 8000)
::    • Dashboard   (port 3000)
:: ════════════════════════════════════════════════════════════════════

title DIPEX Launcher v2

cls
echo.
echo  ██████╗ ██╗██████╗ ███████╗██╗  ██╗
echo  ██╔══██╗██║██╔══██╗██╔════╝╚██╗██╔╝
echo  ██║  ██║██║██████╔╝█████╗   ╚███╔╝
echo  ██║  ██║██║██╔═══╝ ██╔══╝   ██╔██╗
echo  ██████╔╝██║██║     ███████╗██╔╝ ██╗
echo  ╚═════╝ ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝
echo.
echo  Data Intelligence Pipeline  ^|  Launch Script v2
echo  ════════════════════════════════════════════════
echo.

:: ── Step 1: Verify Docker Desktop is running ───────────────────────
echo  [1/6]  Checking Docker Desktop...
docker info >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Docker Desktop is NOT running.
    echo          Please open Docker Desktop, wait for it to fully start,
    echo          then run this script again.
    echo.
    pause
    exit /b 1
)
echo          Docker Desktop  OK
echo.

:: ── Step 2: Check for port conflicts ──────────────────────────────
echo  [2/6]  Checking for port conflicts...
set CONFLICT=0

netstat -aon | find ":3000 " | find "LISTENING" >nul 2>&1
if not errorlevel 1 ( echo          [WARN] Port 3000 already in use  ^(Dashboard^) & set CONFLICT=1 )

netstat -aon | find ":8000 " | find "LISTENING" >nul 2>&1
if not errorlevel 1 ( echo          [WARN] Port 8000 already in use  ^(API^) & set CONFLICT=1 )

netstat -aon | find ":8080 " | find "LISTENING" >nul 2>&1
if not errorlevel 1 ( echo          [WARN] Port 8080 already in use  ^(Kafka UI^) & set CONFLICT=1 )

netstat -aon | find ":9092 " | find "LISTENING" >nul 2>&1
if not errorlevel 1 ( echo          [WARN] Port 9092 already in use  ^(Kafka^) & set CONFLICT=1 )

if "%CONFLICT%"=="1" (
    echo.
    echo          Some ports are already in use. This might be from a previous
    echo          DIPEX run. Attempting to stop old containers first...
    docker compose down >nul 2>&1
    timeout /t 3 /nobreak >nul
    echo          Old containers stopped. Continuing...
    echo.
) else (
    echo          All ports are free  OK
    echo.
)

:: ── Step 3: Pull latest images ─────────────────────────────────────
echo  [3/6]  Pulling / updating Docker images...
echo          ^(First run may take several minutes — cached on subsequent runs^)
echo.
docker compose pull --ignore-buildable --quiet
echo.
echo          Images ready  OK
echo.

:: ── Step 4: Build and start all services ───────────────────────────
echo  [4/6]  Building and starting all DIPEX services...
echo.
docker compose up -d --build
if errorlevel 1 (
    echo.
    echo  [ERROR] docker compose failed to start.
    echo          Run the following to see what went wrong:
    echo              docker compose logs
    echo.
    pause
    exit /b 1
)
echo.
echo          All containers launched  OK
echo.

:: ── Step 5: Wait for services to become healthy ────────────────────
echo  [5/6]  Waiting for services to become healthy...
echo          ^(This typically takes 20-40 seconds on first run^)
echo.

set /a WAIT=0
:WAIT_LOOP
timeout /t 5 /nobreak >nul
set /a WAIT+=5

:: Check API health endpoint
curl -s -o nul -w "%%{http_code}" http://localhost:8000/health 2>nul | find "200" >nul
if not errorlevel 1 goto HEALTHY

if %WAIT% GEQ 90 (
    echo.
    echo          [WARN] API health check timed out after 90s.
    echo          Services may still be starting. Check logs with:
    echo              docker compose logs dipex-api
    echo.
    goto SHOW_URLS
)

echo          Still waiting... ^(%WAIT%s elapsed^)
goto WAIT_LOOP

:HEALTHY
echo          API is healthy  OK
echo.

:: ── Step 6: Show service status ────────────────────────────────────
:SHOW_URLS
echo  [6/6]  Verifying container status...
echo.
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
echo.

echo  ════════════════════════════════════════════════════════════════
echo.
echo   DIPEX is running! Access your services below:
echo.
echo   ^> Dashboard (Frontend)  :   http://localhost:3000
echo   ^> Backend API           :   http://localhost:8000
echo   ^> API Docs (Swagger)    :   http://localhost:8000/docs
echo   ^> API Docs (ReDoc)      :   http://localhost:8000/redoc
echo   ^> Kafka UI              :   http://localhost:8080
echo   ^> PostgreSQL            :   localhost:5433  (user: dipex)
echo   ^> MongoDB               :   localhost:27018 (user: dipex)
echo.
echo   Useful commands:
echo     docker compose logs -f            ^(stream all logs^)
echo     docker compose logs -f dipex-api  ^(stream API logs only^)
echo     docker compose ps                 ^(check container status^)
echo     docker compose down               ^(stop all services^)
echo     docker compose down -v            ^(stop + wipe volumes^)
echo.
echo  ════════════════════════════════════════════════════════════════
echo.

:: Open dashboard in default browser
start "" http://localhost:3000

echo   Dashboard opened in your browser.
echo   Press any key to close this launcher window...
echo   ^(Services will continue running in the background^)
echo.
pause >nul
endlocal
