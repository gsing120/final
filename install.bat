@echo off
REM Gemma Advanced Trading System Windows Installation Script
REM This script installs the Gemma Advanced Trading System and its dependencies

echo === Gemma Advanced Trading System Installer ===
echo Starting installation process...

REM Create installation directory
set INSTALL_DIR=%USERPROFILE%\GemmaTrading
mkdir "%INSTALL_DIR%" 2>nul
echo Created installation directory: %INSTALL_DIR%

REM Check Python version
echo Checking Python version...
python --version 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.8 or higher before continuing.
    echo You can download Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating Python virtual environment...
python -m venv "%INSTALL_DIR%\venv"
call "%INSTALL_DIR%\venv\Scripts\activate.bat"
echo Virtual environment created and activated

REM Install Python dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn matplotlib seaborn plotly torch tensorflow fastapi uvicorn sqlalchemy redis pytest pytest-cov
echo Dependencies installed successfully

REM Copy application files
echo Copying application files...
xcopy /E /I /Y backend "%INSTALL_DIR%\backend"
xcopy /E /I /Y frontend "%INSTALL_DIR%\frontend"
xcopy /E /I /Y docs "%INSTALL_DIR%\docs"
mkdir "%INSTALL_DIR%\data" 2>nul
mkdir "%INSTALL_DIR%\logs" 2>nul
mkdir "%INSTALL_DIR%\config" 2>nul

REM Create default configuration
echo Creating default configuration...
(
echo [General]
echo debug = false
echo log_level = INFO
echo data_dir = data
echo log_dir = logs
echo.
echo [Market]
echo default_market = US
echo trading_hours_start = 09:30
echo trading_hours_end = 16:00
echo pre_market_start = 04:00
echo post_market_end = 20:00
echo.
echo [API]
echo alpha_vantage_key = YOUR_ALPHA_VANTAGE_KEY
echo moomoo_app_key = YOUR_MOOMOO_APP_KEY
echo moomoo_app_secret = YOUR_MOOMOO_APP_SECRET
echo.
echo [Risk]
echo max_position_size = 0.05
echo max_portfolio_risk = 0.02
echo default_stop_loss = 0.02
echo default_take_profit = 0.06
echo max_correlation = 0.7
echo.
echo [AI]
echo gemma_model_path = models/gemma3
echo use_gpu = true
echo inference_threads = 4
) > "%INSTALL_DIR%\config\config.ini"

REM Create launcher script
echo Creating launcher script...
(
echo @echo off
echo call "%INSTALL_DIR%\venv\Scripts\activate.bat"
echo cd /d "%INSTALL_DIR%"
echo python backend\main.py
echo pause
) > "%INSTALL_DIR%\start_gemma_trading.bat"

REM Create desktop shortcut
echo Creating desktop shortcut...
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\Gemma Trading.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\start_gemma_trading.bat'; $Shortcut.IconLocation = '%INSTALL_DIR%\frontend\assets\icon.ico,0'; $Shortcut.Description = 'AI-powered trading platform'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Save()"

echo Installation completed successfully!
echo You can start the application by:
echo 1. Double-clicking the desktop shortcut
echo 2. Running %INSTALL_DIR%\start_gemma_trading.bat
echo.
echo Please edit %INSTALL_DIR%\config\config.ini to configure your API keys and preferences.
echo Refer to the documentation in %INSTALL_DIR%\docs for usage instructions.

pause
