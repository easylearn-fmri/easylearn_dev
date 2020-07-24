@echo off
$dir = 'D:\My_Codes\easylearn-fmri\eslearn'

for /f "delims=" %%a in ('dir /ad /b /s $dir\^|sort /r') do (

rd "%%a">nul 2>nul &&echo 空目录"%%a"成功删除！

)

pause