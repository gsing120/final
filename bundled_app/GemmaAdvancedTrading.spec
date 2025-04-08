# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['fixed_frontend_app.py'],
    pathex=[],
    binaries=[],
    datas=[('static', 'static'), ('templates', 'templates'), ('charts', 'charts'), ('docs', 'docs')],
    hiddenimports=['pandas', 'numpy', 'matplotlib', 'yfinance', 'flask', 'flask_cors', 'scikit-learn'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='GemmaAdvancedTrading',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
