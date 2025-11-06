@echo off
setlocal enabledelayedexpansion

REM ==== 設定 ====
set "input_folder=C:\Users\tk71\Downloads\gerrard-hall\images"
set "output_folder=C:\Users\tk71\Downloads\SimpleGaussianSplat_tk71\colmap\images"
set "width=640"
REM ==============

if not exist "%output_folder%" (
    mkdir "%output_folder%"
)

for %%F in ("%input_folder%\*.jpg" "%input_folder%\*.jpeg" "%input_folder%\*.png" "%input_folder%\*.bmp") do (
    set "input_file=%%~fF"
    set "filename=%%~nxF"
    set "output_file=%output_folder%\!filename!"
    echo 処理中: !input_file!
    ffmpeg -y -i "!input_file!" -map_metadata 0 -vf "scale=%width%:-1" "!output_file!"
)

echo.
echo ==== リサイズ完了（Exif情報も保持）====
pause
