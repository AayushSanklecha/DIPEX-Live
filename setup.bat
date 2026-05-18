@echo off
:: =============================================================================
::  ADAP Analytics Platform — One-Shot Environment Setup (Windows)
::  Run this ONCE after cloning the repo before starting the backend.
::
::  Usage:
::    setup.bat
:: =============================================================================

echo.
echo =========================================================
echo   ADAP Analytics Platform -- Environment Setup (Windows)
echo =========================================================
echo.

:: ── 1. Python dependencies ────────────────────────────────────
echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 goto error
echo       Done. Python packages installed.
echo.

:: ── 2. spaCy language model (REQUIRED for NLP Column Analyzer) ─
echo [2/4] Downloading spaCy English language model...
python -m spacy download en_core_web_sm
if errorlevel 1 goto error
echo       Done. spaCy en_core_web_sm downloaded.
echo.

:: ── 3. Create required directories ───────────────────────────
echo [3/4] Creating required runtime directories...
if not exist "reports" mkdir reports
if not exist "data\snapshots" mkdir data\snapshots
if not exist "data\uploads" mkdir data\uploads
if not exist "data\tmp" mkdir data\tmp
if not exist "data\bronze" mkdir data\bronze
if not exist "data\silver" mkdir data\silver
if not exist "data\gold" mkdir data\gold
if not exist "data\model_registry" mkdir data\model_registry
if not exist "audit" mkdir audit
if not exist "models" mkdir models
echo       Done. Runtime directories created.
echo.

:: ── 4. .env file ─────────────────────────────────────────────
echo [4/4] Checking .env file...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env
        echo       .env created from .env.example -- please fill in your API keys.
    ) else (
        echo       WARNING: .env.example not found. Please create a .env file manually.
    )
) else (
    echo       .env already exists -- skipping.
)
echo.

echo =========================================================
echo   Setup complete! You can now start the platform:
echo.
echo   Backend:   uvicorn api.app:app --reload --port 8000
echo   Frontend:  cd frontend ^&^& npm install ^&^& npm run dev
echo =========================================================
echo.
goto end

:error
echo.
echo ERROR: Setup failed. Please check the error above.
exit /b 1

:end
