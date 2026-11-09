$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$pythonPath = Join-Path $projectDir '.venv\Scripts\python.exe'
$launcherPath = Join-Path $projectDir 'scripts\run_local.py'
$runtimeDir = Join-Path $projectDir '.local'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '请先创建 .venv 并安装 requirements.txt 中的依赖。'
}
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null

foreach ($serviceName in @('api', 'dashboard')) {
    $pidFile = Join-Path $runtimeDir "$serviceName.pid"
    if (Test-Path -LiteralPath $pidFile) {
        $servicePid = [int](Get-Content -LiteralPath $pidFile -Raw)
        $existingProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$servicePid"
        if ($existingProcess -and $existingProcess.CommandLine -like "*$launcherPath*$serviceName*") {
            Write-Host "$serviceName : 已启动，进程号 $servicePid。"
            continue
        }
    }
    $servicePort = if ($serviceName -eq 'api') { 8000 } else { 8501 }
    $listener = Get-NetTCPConnection -LocalPort $servicePort -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        Write-Host "$serviceName : 端口 $servicePort 已有服务监听。"
        continue
    }

    # 后台运行并保留日志，避免打开额外终端窗口。
    $serviceProcess = Start-Process -FilePath $pythonPath `
        -ArgumentList "`"$launcherPath`" $serviceName" `
        -WorkingDirectory $projectDir -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtimeDir "$serviceName.stdout.log") `
        -RedirectStandardError (Join-Path $runtimeDir "$serviceName.stderr.log")
    Set-Content -LiteralPath $pidFile -Value $serviceProcess.Id
    Write-Host "$serviceName : 已启动，进程号 $($serviceProcess.Id)。"
}

Write-Host '接口文档： http://127.0.0.1:8000/docs'
Write-Host '检索界面： http://127.0.0.1:8501'
Write-Host '首次启动需要下载向量和重排模型，请查看 .local 下的日志。'
