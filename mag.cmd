@echo off
setlocal
rem Bulletproof launcher: calls the venv Python directly so no venv activation,
rem PATH change, or PowerShell execution policy is involved. setlocal keeps any
rem env vars we set here scoped to this run, not the caller's shell.
rem Load .env (KEY=value lines; # comments skipped) so API keys need not be set
rem by hand. The test suite invokes the CLI directly, not this launcher, so this
rem never leaks secrets into tests.
if exist "%~dp0.env" for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~dp0.env") do set "%%A=%%B"
rem Storage defaults to a user-writable folder unless already set.
if "%MAGNETOR_DATA_ROOT%"=="" set "MAGNETOR_DATA_ROOT=%LOCALAPPDATA%\Magnetor\data"
"%~dp0.venv\Scripts\python.exe" -m magnetor.cli %*
