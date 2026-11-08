@echo off
setlocal EnableExtensions
title NIRF Insights Dashboard
rem Always run from the folder this file lives in
cd /d "%~dp0"

echo.
echo  ==================================================
echo    NIRF Insights Dashboard
echo  ==================================================
echo.

rem ---- 1. Find Python --------------------------------------------------------
set "PY=python"
%PY% --version >nul 2>&1
if errorlevel 1 set "PY=py"
%PY% --version >nul 2>&1
if errorlevel 1 (
    echo  [X] Python was not found on this computer.
    echo      Install it from https://www.python.org/downloads/ and tick "Add to PATH".
    goto :fail
)

rem ---- 2. Make sure the packages are installed (first run only) -------------
%PY% -c "import streamlit, pandas, numpy, plotly, matplotlib, seaborn, scipy, statsmodels, dotenv; from google import genai" >nul 2>&1
if errorlevel 1 (
    echo  [..] Installing required packages - this happens only once...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo  [X] Package installation failed. Check your internet connection.
        goto :fail
    )
)
echo  [OK] Python and packages ready

rem ---- 3. Check the Gemini key (only the Ask AI page needs it) --------------
if exist ".env" (
    echo  [OK] .env found - AI assistant enabled
) else (
    echo  [!]  No .env file - the dashboard works, but Ask AI will show setup steps
)

rem ---- 4. Pick a free port (8501, or the next free one) ---------------------
set PORT=8501
:checkport
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
    set /a PORT+=1
    goto :checkport
)

rem ---- 5. Open the browser as soon as the server is ready -------------------
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 90;$i++){try{if((Invoke-WebRequest -UseBasicParsing 'http://localhost:%PORT%/_stcore/health' -TimeoutSec 2).StatusCode -eq 200){Start-Process 'http://localhost:%PORT%';break}}catch{};Start-Sleep -Milliseconds 700}"

echo.
echo  [GO] Starting at http://localhost:%PORT%  - your browser will open automatically.
echo       Keep this window open while presenting. Close it (or press Ctrl+C) to stop.
echo.
%PY% -m streamlit run app.py --server.port %PORT% --server.headless true
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo  Something went wrong - see the message above.
pause
exit /b 1
