# FBT shortcut on the desktop: runs the hub with pythonw (no console; its output goes to
# data\logs\app.log), and only the FBT window shows: closing it quits FBT. setup.ps1 runs this too.
# windows\FBT.cmd is the same with a console, for looking at the output live.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
& "$root\.venv\Scripts\python.exe" "$root\windows\make_icon.py" | Out-Null
$lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "FBT.lnk"
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = "$root\.venv\Scripts\pythonw.exe"
$s.Arguments = "-m fbtapp --browser"
$s.WorkingDirectory = $root
$s.IconLocation = "$root\windows\FBT.ico,0"
$s.Description = "FBT: трекінг тіла з телефонів для VRChat"
$s.Save()
Write-Host "Ярлик: $lnk"
