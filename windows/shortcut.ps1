# FBT shortcut on the desktop: starts windows\FBT.cmd with its console minimised, so only the
# FBT window shows (close the console from the taskbar to quit). setup.ps1 runs this too.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
& "$root\.venv\Scripts\python.exe" "$root\windows\make_icon.py" | Out-Null
$lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "FBT.lnk"
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath = "$root\windows\FBT.cmd"
$s.WorkingDirectory = $root
$s.IconLocation = "$root\windows\FBT.ico,0"
$s.WindowStyle = 7
$s.Description = "FBT: трекінг тіла з телефонів для VRChat"
$s.Save()
Write-Host "Ярлик: $lnk"
