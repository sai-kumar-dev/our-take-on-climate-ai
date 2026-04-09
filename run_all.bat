@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0" || exit /b 1

set "APP_ROOT=%cd%"
set "VENV_PYTHON=%APP_ROOT%\.venv\Scripts\python.exe"
set "API_HOST=127.0.0.1"
set "API_PORT=8000"
set "UI_PORT=8501"
set "API_URL=http://%API_HOST%:%API_PORT%"
set "HEALTH_URL=%API_URL%/health"
set "METRICS_URL=%API_URL%/metrics"
set "UI_URL=http://%API_HOST%:%UI_PORT%"
set "RUNTIME_DIR=%APP_ROOT%\logs\runtime"
set "PID_DIR=%APP_ROOT%\.tmp"
set "API_PID_FILE=%PID_DIR%\api.pid"
set "UI_PID_FILE=%PID_DIR%\ui.pid"
set "API_OUT_LOG=%RUNTIME_DIR%\api.out.log"
set "API_ERR_LOG=%RUNTIME_DIR%\api.err.log"
set "UI_OUT_LOG=%RUNTIME_DIR%\ui.out.log"
set "UI_ERR_LOG=%RUNTIME_DIR%\ui.err.log"

if defined APP_API_PORT set "API_PORT=%APP_API_PORT%"
if defined APP_UI_PORT set "UI_PORT=%APP_UI_PORT%"
if defined APP_API_HOST set "API_HOST=%APP_API_HOST%"
set "API_URL=http://%API_HOST%:%API_PORT%"
set "HEALTH_URL=%API_URL%/health"
set "METRICS_URL=%API_URL%/metrics"
set "UI_URL=http://%API_HOST%:%UI_PORT%"

set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=start"
if not defined APP_OPEN_BROWSER set "APP_OPEN_BROWSER=1"

call :ensure_dirs
if errorlevel 1 exit /b 1

if /I "%ACTION%"=="start" call :start_stack & exit /b !ERRORLEVEL!
if /I "%ACTION%"=="stop" call :stop_stack & exit /b !ERRORLEVEL!
if /I "%ACTION%"=="restart" call :restart_stack & exit /b !ERRORLEVEL!
if /I "%ACTION%"=="status" call :status_stack & exit /b !ERRORLEVEL!
if /I "%ACTION%"=="watch" call :watch_stack & exit /b !ERRORLEVEL!
if /I "%ACTION%"=="logs" call :tail_logs "%~2" & exit /b !ERRORLEVEL!
if /I "%ACTION%"=="doctor" call :doctor_stack & exit /b !ERRORLEVEL!
if /I "%ACTION%"=="help" call :usage & exit /b 0

echo [error] Unknown action: %ACTION%
call :usage
exit /b 1

:ensure_dirs
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%" >nul 2>&1
if not exist "%PID_DIR%" mkdir "%PID_DIR%" >nul 2>&1
exit /b 0

:usage
echo Usage:
echo   run_all.bat start
echo   run_all.bat stop
echo   run_all.bat restart
echo   run_all.bat status
echo   run_all.bat watch
echo   run_all.bat logs [api^|ui^|all]
echo   run_all.bat doctor
echo.
echo Defaults:
echo   API: %API_URL%
echo   UI : %UI_URL%
echo.
echo Optional environment overrides:
echo   APP_API_HOST, APP_API_PORT, APP_UI_PORT
echo   APP_OPEN_BROWSER=0 to suppress automatic browser launch
exit /b 0

:open_browser
if /I "%APP_OPEN_BROWSER%"=="0" exit /b 0
start "" "%~1" >nul 2>&1
if errorlevel 1 echo [info] Open %~1 in your browser.
exit /b 0

:require_python
if exist "%VENV_PYTHON%" exit /b 0
echo [error] Missing virtualenv Python: "%VENV_PYTHON%"
echo Activate or create the project venv first.
exit /b 1

:start_stack
call :require_python
if errorlevel 1 exit /b 1

call :start_api
if errorlevel 1 exit /b 1

call :wait_for_http "%HEALTH_URL%" "API health" 90
if errorlevel 1 (
    call :print_recent_log "%API_ERR_LOG%" "API error log"
    call :stop_api >nul 2>&1
    exit /b 1
)

call :start_ui
if errorlevel 1 (
    call :stop_api >nul 2>&1
    exit /b 1
)

call :wait_for_http "%UI_URL%" "UI" 120
if errorlevel 1 (
    call :print_recent_log "%UI_ERR_LOG%" "UI error log"
    call :stop_ui >nul 2>&1
    call :stop_api >nul 2>&1
    exit /b 1
)

echo [ok] Stack is ready.
echo [info] API: %API_URL%
echo [info] UI : %UI_URL%
echo [info] Logs:
echo        %API_OUT_LOG%
echo        %API_ERR_LOG%
echo        %UI_OUT_LOG%
echo        %UI_ERR_LOG%
echo [info] Use "run_all.bat status" for a quick check or "run_all.bat stop" to shut everything down.
call :open_browser "%UI_URL%"
exit /b 0

:restart_stack
call :stop_stack
call :start_stack
exit /b !ERRORLEVEL!

:start_api
call :read_pid "%API_PID_FILE%" API_PID
if defined API_PID (
    call :is_pid_running "!API_PID!"
    if not errorlevel 1 (
        echo [info] API already running on PID !API_PID!
        exit /b 0
    )
    del /q "%API_PID_FILE%" >nul 2>&1
)

echo [info] Starting API on %API_URL% ...
"%VENV_PYTHON%" "%APP_ROOT%\src\process_launcher.py" ^
    --pid-file "%API_PID_FILE%" ^
    --stdout-log "%API_OUT_LOG%" ^
    --stderr-log "%API_ERR_LOG%" ^
    --cwd "%APP_ROOT%" ^
    -- "%VENV_PYTHON%" -m uvicorn src.app_api_entry:app --host %API_HOST% --port %API_PORT%
if errorlevel 1 (
    echo [error] Failed to start API.
    exit /b 1
)
call :read_pid "%API_PID_FILE%" API_PID
echo [ok] API launched with PID !API_PID!
exit /b 0

:start_ui
call :read_pid "%UI_PID_FILE%" UI_PID
if defined UI_PID (
    call :is_pid_running "!UI_PID!"
    if not errorlevel 1 (
        echo [info] UI already running on PID !UI_PID!
        exit /b 0
    )
    del /q "%UI_PID_FILE%" >nul 2>&1
)

echo [info] Starting UI on %UI_URL% ...
"%VENV_PYTHON%" "%APP_ROOT%\src\process_launcher.py" ^
    --pid-file "%UI_PID_FILE%" ^
    --stdout-log "%UI_OUT_LOG%" ^
    --stderr-log "%UI_ERR_LOG%" ^
    --cwd "%APP_ROOT%" ^
    --env "API_BASE_URL=%API_URL%" ^
    -- "%VENV_PYTHON%" -m streamlit run src\ui_app_source.py --server.port %UI_PORT% --server.headless true
if errorlevel 1 (
    echo [error] Failed to start UI.
    exit /b 1
)
call :read_pid "%UI_PID_FILE%" UI_PID
echo [ok] UI launched with PID !UI_PID!
exit /b 0

:wait_for_http
set "WAIT_URL=%~1"
set "WAIT_LABEL=%~2"
set "WAIT_SECONDS=%~3"
echo [info] Waiting for %WAIT_LABEL% at %WAIT_URL% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$deadline = (Get-Date).AddSeconds(%WAIT_SECONDS%); " ^
    "while ((Get-Date) -lt $deadline) { " ^
    "  try { " ^
    "    $response = Invoke-WebRequest -Uri '%WAIT_URL%' -UseBasicParsing -TimeoutSec 4; " ^
    "    if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) { exit 0 } " ^
    "  } catch { } " ^
    "  Start-Sleep -Seconds 2 " ^
    "} " ^
    "exit 1"
if errorlevel 1 (
    echo [error] %WAIT_LABEL% did not become ready in %WAIT_SECONDS% seconds.
    exit /b 1
)
echo [ok] %WAIT_LABEL% is ready.
exit /b 0

:stop_stack
call :stop_ui
call :stop_api
echo [ok] Stop request completed.
exit /b 0

:stop_api
call :stop_service "API" "%API_PID_FILE%"
exit /b !ERRORLEVEL!

:stop_ui
call :stop_service "UI" "%UI_PID_FILE%"
exit /b !ERRORLEVEL!

:stop_service
set "SERVICE_NAME=%~1"
set "SERVICE_PID_FILE=%~2"
call :read_pid "%SERVICE_PID_FILE%" SERVICE_PID
if not defined SERVICE_PID (
    echo [info] %SERVICE_NAME% is not running.
    exit /b 0
)

call :is_pid_running "%SERVICE_PID%"
if errorlevel 1 (
    del /q "%SERVICE_PID_FILE%" >nul 2>&1
    echo [info] %SERVICE_NAME% PID file was stale and has been removed.
    exit /b 0
)

echo [info] Stopping %SERVICE_NAME% on PID %SERVICE_PID% ...
taskkill /PID %SERVICE_PID% /T /F >nul 2>&1
if errorlevel 1 (
    call :is_pid_running "%SERVICE_PID%"
    if errorlevel 1 (
        del /q "%SERVICE_PID_FILE%" >nul 2>&1
        echo [ok] %SERVICE_NAME% stopped.
        exit /b 0
    )
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "try { Stop-Process -Id %SERVICE_PID% -Force -ErrorAction Stop; exit 0 } catch { exit 1 }"
    if errorlevel 1 (
        call :is_pid_running "%SERVICE_PID%"
        if errorlevel 1 (
            del /q "%SERVICE_PID_FILE%" >nul 2>&1
            echo [ok] %SERVICE_NAME% stopped.
            exit /b 0
        )
        echo [warning] Could not stop %SERVICE_NAME% cleanly. You may need to close it manually.
        exit /b 1
    )
)
del /q "%SERVICE_PID_FILE%" >nul 2>&1
echo [ok] %SERVICE_NAME% stopped.
exit /b 0

:status_stack
echo [info] Root: %APP_ROOT%
echo [info] API : %API_URL%
echo [info] UI  : %UI_URL%
call :print_service_status "API" "%API_PID_FILE%" "%HEALTH_URL%"
call :print_service_status "UI" "%UI_PID_FILE%" "%UI_URL%"
call :print_api_details
call :print_metrics
exit /b 0

:print_service_status
set "SERVICE_NAME=%~1"
set "SERVICE_PID_FILE=%~2"
set "SERVICE_URL=%~3"
call :read_pid "%SERVICE_PID_FILE%" SERVICE_PID
if defined SERVICE_PID (
    call :is_pid_running "%SERVICE_PID%"
    if errorlevel 1 (
        echo [stale] %SERVICE_NAME% PID file exists but process is not running.
        del /q "%SERVICE_PID_FILE%" >nul 2>&1
        echo [info] Removed stale %SERVICE_NAME% PID file.
    ) else (
        echo [live] %SERVICE_NAME% process PID %SERVICE_PID%
    )
)
if not defined SERVICE_PID echo [down] %SERVICE_NAME% process not running.

call :http_ping "%SERVICE_URL%"
if errorlevel 1 (
    echo [down] %SERVICE_NAME% endpoint not reachable: %SERVICE_URL%
) else (
    echo [up]   %SERVICE_NAME% endpoint reachable: %SERVICE_URL%
)
exit /b 0

:print_api_details
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { " ^
    "  $payload = Invoke-RestMethod -Uri '%HEALTH_URL%' -TimeoutSec 4; " ^
    "  Write-Host ('[info] Model version: ' + $payload.model_version); " ^
    "  Write-Host ('[info] Mode         : ' + $payload.mode); " ^
    "  Write-Host ('[info] Artifact dir : ' + $payload.artifact_dir); " ^
    "} catch { }"
exit /b 0

:print_metrics
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { " ^
    "  $metrics = Invoke-RestMethod -Uri '%METRICS_URL%' -TimeoutSec 4; " ^
    "  Write-Host ('[info] Requests     : ' + $metrics.total_requests); " ^
    "  Write-Host ('[info] Predictions  : ' + $metrics.prediction_requests); " ^
    "  Write-Host ('[info] Avg latency  : ' + $metrics.avg_latency_ms + ' ms'); " ^
    "} catch { }"
exit /b 0

:watch_stack
echo [info] Watching stack status. Press Ctrl+C to stop.
:watch_loop
cls
call :status_stack
timeout /t 5 /nobreak >nul
goto watch_loop

:tail_logs
set "LOG_TARGET=%~1"
if "%LOG_TARGET%"=="" set "LOG_TARGET=all"

if /I "%LOG_TARGET%"=="api" (
    start "API Logs" powershell -NoExit -Command "Get-Content -Path '%API_OUT_LOG%','%API_ERR_LOG%' -Wait"
    exit /b 0
)
if /I "%LOG_TARGET%"=="ui" (
    start "UI Logs" powershell -NoExit -Command "Get-Content -Path '%UI_OUT_LOG%','%UI_ERR_LOG%' -Wait"
    exit /b 0
)
if /I "%LOG_TARGET%"=="all" (
    start "API Logs" powershell -NoExit -Command "Get-Content -Path '%API_OUT_LOG%','%API_ERR_LOG%' -Wait"
    start "UI Logs" powershell -NoExit -Command "Get-Content -Path '%UI_OUT_LOG%','%UI_ERR_LOG%' -Wait"
    exit /b 0
)

echo [error] Unknown logs target: %LOG_TARGET%
echo [info] Use: run_all.bat logs api
echo [info]      run_all.bat logs ui
echo [info]      run_all.bat logs all
exit /b 1

:doctor_stack
call :require_python
if errorlevel 1 exit /b 1

echo [info] Python executable: %VENV_PYTHON%
"%VENV_PYTHON%" -V
if errorlevel 1 (
    echo [error] Python environment is not usable.
    exit /b 1
)

if exist "%APP_ROOT%\artifacts\data_new_training\trained_model.pkl" (
    echo [ok] Found production model artifact.
)
if not exist "%APP_ROOT%\artifacts\data_new_training\trained_model.pkl" (
    echo [warning] Production model artifact is missing.
)

call :status_stack

echo [info] Port listeners:
netstat -ano | findstr /R /C:":%API_PORT% .*LISTENING" /C:":%UI_PORT% .*LISTENING"
exit /b 0

:print_recent_log
if not exist "%~1" exit /b 0
echo [info] Recent %~2:
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Get-Content -Path '%~1' -Tail 40"
exit /b 0

:http_ping
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { " ^
    "  $null = Invoke-WebRequest -Uri '%~1' -UseBasicParsing -TimeoutSec 4; " ^
    "  exit 0 " ^
    "} catch { " ^
    "  exit 1 " ^
    "}"
exit /b %ERRORLEVEL%

:read_pid
set "%~2="
if not exist "%~1" exit /b 0
set /p PID_VALUE=<"%~1"
set "%~2=%PID_VALUE%"
exit /b 0

:is_pid_running
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "if (Get-Process -Id %~1 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
exit /b %ERRORLEVEL%
