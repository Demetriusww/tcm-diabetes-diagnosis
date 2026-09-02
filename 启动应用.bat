@echo off
chcp 65001 >nul
echo ========================================
echo   🌿 症智明辨 - 中医智能辨证系统
echo   快速启动脚本
echo ========================================
echo.

echo [1/3] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)
echo ✅ Python 环境正常

echo.
echo [2/3] 检查依赖包...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  正在安装依赖包，请稍候...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖安装失败
        pause
        exit /b 1
    )
)
echo ✅ 依赖包已安装

echo.
echo [3/3] 启动应用...
echo.
echo 🎉 应用即将启动，浏览器会自动打开
echo 如果未自动打开，请访问：http://localhost:8501
echo.
echo 按 Ctrl+C 可停止应用
echo.

cd /d "%~dp0"
streamlit run app.py --server.headless true

pause
