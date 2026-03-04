@echo off
title DIPEX — Stop All Services
cd /d "%~dp0"
echo Stopping all DIPEX containers...
docker compose down
echo.
echo All services stopped.
pause
