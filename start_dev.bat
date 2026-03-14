@echo off
title DIPEX Docker Server

echo ===================================================
echo Starting FULL DIPEX Stack via Docker
echo ===================================================
echo.

echo Starting ALL services (Databases, Kafka, API, and Dashboard)...
echo First run might take a minute, but subsequent runs will be INSTANT because of caching!
docker compose up -d --build

echo.
echo ===================================================
echo All services are running!
echo.
echo  - Dashboard (Frontend) :    http://localhost:3000
echo  - Backend API          :    http://localhost:8000
echo  - Kafka UI             :    http://localhost:8080
echo.
echo To see logs, run: docker compose logs -f
echo To stop, run:     docker compose down
echo.
echo Press any key to close this launcher window...
echo ===================================================
pause >nul
