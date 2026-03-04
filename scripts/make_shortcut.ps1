$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
$link = $ws.CreateShortcut("$desktop\Start DIPEX.lnk")
$link.TargetPath = 'C:\Users\sankl\Desktop\dipex\start.bat'
$link.WorkingDirectory = 'C:\Users\sankl\Desktop\dipex'
$link.IconLocation = 'C:\Windows\System32\shell32.dll,162'
$link.Save()
Write-Host "Shortcut created on Desktop."
