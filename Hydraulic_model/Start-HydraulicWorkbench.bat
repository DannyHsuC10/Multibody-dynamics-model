@echo off
setlocal
pushd "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -X utf8 -m Hydraulic_model.workbench
) else if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
  "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -X utf8 -m Hydraulic_model.workbench
) else (
  python -X utf8 -m Hydraulic_model.workbench
)
set "exit_status=%ERRORLEVEL%"
popd
if not "%exit_status%"=="0" pause
exit /b %exit_status%
