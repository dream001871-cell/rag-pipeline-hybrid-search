$ErrorActionPreference = 'Stop'
$runtimeDir = Join-Path $PSScriptRoot '.local'
$launcherPath = Join-Path $PSScriptRoot 'scripts\run_local.py'

foreach ($serviceName in @('api', 'dashboard')) {
    $pidFile = Join-Path $runtimeDir "$serviceName.pid"
    if (-not (Test-Path -LiteralPath $pidFile)) {
        continue
    }
    $servicePid = [int](Get-Content -LiteralPath $pidFile -Raw)
    $serviceProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$servicePid"
    if ($serviceProcess) {
        # 进程号可能被重用，只停止由本项目入口启动的服务。
        if ($serviceProcess.CommandLine -notlike "*$launcherPath*$serviceName*") {
            throw "进程 $servicePid 不属于本项目的 $serviceName 服务。"
        }
        # Windows 虚拟环境的 python.exe 可能另起解释器，需要停止整个服务进程树。
        & taskkill.exe /PID $servicePid /T /F
        if ($LASTEXITCODE -ne 0) {
            throw "停止 $serviceName 失败（进程号 $servicePid）。"
        }
        Write-Host "$serviceName : 已停止，进程号 $servicePid。"
    }
    Remove-Item -LiteralPath $pidFile
}
