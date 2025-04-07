; NSIS Installer Script for Gemma Advanced Trading System
; This script creates a Windows installer for the Gemma Advanced Trading System

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"

; General settings
Name "Gemma Advanced Trading System"
OutFile "GemmaAdvancedTrading_Setup.exe"
InstallDir "$PROGRAMFILES64\Gemma Trading"
InstallDirRegKey HKCU "Software\Gemma Trading" ""
RequestExecutionLevel admin

; Interface settings
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${NSISDIR}\Contrib\Graphics\Wizard\win.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP "${NSISDIR}\Contrib\Graphics\Header\win.bmp"
!define MUI_HEADERIMAGE_RIGHT

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Languages
!insertmacro MUI_LANGUAGE "English"

; Installer sections
Section "Gemma Advanced Trading System" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  
  ; Check for Python installation
  nsExec::ExecToStack 'python --version'
  Pop $0
  Pop $1
  ${If} $0 != 0
    MessageBox MB_OK|MB_ICONEXCLAMATION "Python not detected. Please install Python 3.8 or higher before continuing."
    Abort
  ${EndIf}
  
  ; Create directories
  CreateDirectory "$INSTDIR\backend"
  CreateDirectory "$INSTDIR\frontend"
  CreateDirectory "$INSTDIR\docs"
  CreateDirectory "$INSTDIR\data"
  CreateDirectory "$INSTDIR\logs"
  CreateDirectory "$INSTDIR\config"
  CreateDirectory "$INSTDIR\models"
  
  ; Copy files
  File /r "backend\*.*"
  File /r "frontend\*.*"
  File /r "docs\*.*"
  
  ; Create default configuration
  FileOpen $0 "$INSTDIR\config\config.ini" w
  FileWrite $0 "[General]$\r$\n"
  FileWrite $0 "debug = false$\r$\n"
  FileWrite $0 "log_level = INFO$\r$\n"
  FileWrite $0 "data_dir = data$\r$\n"
  FileWrite $0 "log_dir = logs$\r$\n"
  FileWrite $0 "$\r$\n"
  FileWrite $0 "[Market]$\r$\n"
  FileWrite $0 "default_market = US$\r$\n"
  FileWrite $0 "trading_hours_start = 09:30$\r$\n"
  FileWrite $0 "trading_hours_end = 16:00$\r$\n"
  FileWrite $0 "pre_market_start = 04:00$\r$\n"
  FileWrite $0 "post_market_end = 20:00$\r$\n"
  FileWrite $0 "$\r$\n"
  FileWrite $0 "[API]$\r$\n"
  FileWrite $0 "alpha_vantage_key = YOUR_ALPHA_VANTAGE_KEY$\r$\n"
  FileWrite $0 "moomoo_app_key = YOUR_MOOMOO_APP_KEY$\r$\n"
  FileWrite $0 "moomoo_app_secret = YOUR_MOOMOO_APP_SECRET$\r$\n"
  FileWrite $0 "$\r$\n"
  FileWrite $0 "[Risk]$\r$\n"
  FileWrite $0 "max_position_size = 0.05$\r$\n"
  FileWrite $0 "max_portfolio_risk = 0.02$\r$\n"
  FileWrite $0 "default_stop_loss = 0.02$\r$\n"
  FileWrite $0 "default_take_profit = 0.06$\r$\n"
  FileWrite $0 "max_correlation = 0.7$\r$\n"
  FileWrite $0 "$\r$\n"
  FileWrite $0 "[AI]$\r$\n"
  FileWrite $0 "gemma_model_path = models/gemma3$\r$\n"
  FileWrite $0 "use_gpu = true$\r$\n"
  FileWrite $0 "inference_threads = 4$\r$\n"
  FileClose $0
  
  ; Create launcher script
  FileOpen $0 "$INSTDIR\start_gemma_trading.bat" w
  FileWrite $0 "@echo off$\r$\n"
  FileWrite $0 "cd /d $\"$INSTDIR$\"$\r$\n"
  FileWrite $0 "python -m venv venv$\r$\n"
  FileWrite $0 "call venv\Scripts\activate.bat$\r$\n"
  FileWrite $0 "pip install -r requirements.txt$\r$\n"
  FileWrite $0 "python backend\main.py$\r$\n"
  FileWrite $0 "pause$\r$\n"
  FileClose $0
  
  ; Create requirements.txt
  FileOpen $0 "$INSTDIR\requirements.txt" w
  FileWrite $0 "numpy>=1.20.0$\r$\n"
  FileWrite $0 "pandas>=1.3.0$\r$\n"
  FileWrite $0 "scikit-learn>=1.0.0$\r$\n"
  FileWrite $0 "matplotlib>=3.4.0$\r$\n"
  FileWrite $0 "seaborn>=0.11.0$\r$\n"
  FileWrite $0 "plotly>=5.0.0$\r$\n"
  FileWrite $0 "torch>=1.9.0$\r$\n"
  FileWrite $0 "tensorflow>=2.6.0$\r$\n"
  FileWrite $0 "fastapi>=0.68.0$\r$\n"
  FileWrite $0 "uvicorn>=0.15.0$\r$\n"
  FileWrite $0 "sqlalchemy>=1.4.0$\r$\n"
  FileWrite $0 "redis>=4.0.0$\r$\n"
  FileWrite $0 "pytest>=6.2.0$\r$\n"
  FileWrite $0 "pytest-cov>=2.12.0$\r$\n"
  FileClose $0
  
  ; Create shortcuts
  CreateDirectory "$SMPROGRAMS\Gemma Trading"
  CreateShortcut "$SMPROGRAMS\Gemma Trading\Gemma Advanced Trading.lnk" "$INSTDIR\start_gemma_trading.bat" "" "$INSTDIR\frontend\assets\icon.ico" 0
  CreateShortcut "$SMPROGRAMS\Gemma Trading\Uninstall.lnk" "$INSTDIR\uninstall.exe"
  CreateShortcut "$DESKTOP\Gemma Advanced Trading.lnk" "$INSTDIR\start_gemma_trading.bat" "" "$INSTDIR\frontend\assets\icon.ico" 0
  
  ; Write uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"
  
  ; Write registry keys
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\GemmaTrading" "DisplayName" "Gemma Advanced Trading System"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\GemmaTrading" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\GemmaTrading" "QuietUninstallString" "$\"$INSTDIR\uninstall.exe$\" /S"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\GemmaTrading" "InstallLocation" "$\"$INSTDIR$\""
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\GemmaTrading" "DisplayIcon" "$\"$INSTDIR\frontend\assets\icon.ico$\""
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\GemmaTrading" "Publisher" "Gemma Trading"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\GemmaTrading" "DisplayVersion" "1.0.0"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\GemmaTrading" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\GemmaTrading" "NoRepair" 1
  
  ; Calculate and store installation size
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\GemmaTrading" "EstimatedSize" "$0"
SectionEnd

Section "Desktop Shortcut" SecDesktop
  CreateShortcut "$DESKTOP\Gemma Advanced Trading.lnk" "$INSTDIR\start_gemma_trading.bat" "" "$INSTDIR\frontend\assets\icon.ico" 0
SectionEnd

Section "Install Python Dependencies" SecPython
  nsExec::ExecToLog '"$INSTDIR\start_gemma_trading.bat" /FIRSTRUN'
SectionEnd

; Uninstaller section
Section "Uninstall"
  ; Remove files and directories
  RMDir /r "$INSTDIR\backend"
  RMDir /r "$INSTDIR\frontend"
  RMDir /r "$INSTDIR\docs"
  RMDir /r "$INSTDIR\venv"
  Delete "$INSTDIR\config\config.ini"
  Delete "$INSTDIR\start_gemma_trading.bat"
  Delete "$INSTDIR\requirements.txt"
  Delete "$INSTDIR\uninstall.exe"
  
  ; Remove shortcuts
  Delete "$DESKTOP\Gemma Advanced Trading.lnk"
  Delete "$SMPROGRAMS\Gemma Trading\Gemma Advanced Trading.lnk"
  Delete "$SMPROGRAMS\Gemma Trading\Uninstall.lnk"
  RMDir "$SMPROGRAMS\Gemma Trading"
  
  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\GemmaTrading"
  DeleteRegKey HKLM "Software\Gemma Trading"
  
  ; Remove installation directory if empty
  RMDir "$INSTDIR"
SectionEnd

; Section descriptions
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} "Core files for the Gemma Advanced Trading System."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "Create a shortcut on the desktop."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecPython} "Install required Python dependencies (recommended)."
!insertmacro MUI_FUNCTION_DESCRIPTION_END
