$python = "D:\PythonProjects\WANPM_FP\venv\Scripts\python.exe"
$base = "D:\PythonProjects\WANPM_FP\Fractional"
$env:PYTHONIOENCODING = "utf-8"

$scripts = @(
    @{ dir = "fFP_1d_1well";            file = "fFP_1d_1well.py" },
    @{ dir = "fFP_1d_2well";            file = "fFP_1d_2well.py" },
    @{ dir = "fFP_1d_steady_doublepeak"; file = "fFP_1d_steady_doublepeak.py" },
    @{ dir = "fFP_1d_steady_ou";        file = "fFP_1d_steady_ou.py" },
    @{ dir = "fFP_1d_triplewell";       file = "fFP_1d_triplewell.py" },
    @{ dir = "fFP_20d_doublewell";      file = "fFP_20d_doublewell.py" },
    @{ dir = "fFP_2d_ring";             file = "fFP_2d_ring.py" },
    @{ dir = "fFP_2d_steady_doublepeak"; file = "fFP_2d_steady_doublepeak.py" },
    @{ dir = "fFP_2d_steady_ring";      file = "fFP_2d_steady_ring.py" },
    @{ dir = "fFP_nd_ou";               file = "fFP_nd_ou.py" },
    @{ dir = "fFP_nd_1well";            file = "FP_nd_1well.py" }
)

$total = $scripts.Count
$i = 0

foreach ($s in $scripts) {
    $i++
    $fullPath = Join-Path $base "$($s.dir)\$($s.file)"
    $workDir  = Join-Path $base $s.dir
    $logFile  = Join-Path $workDir "run.log"
    $errFile  = "$logFile.err"

    Write-Host ""
    Write-Host "[$i/$total] Starting: $($s.file)  $(Get-Date -Format 'HH:mm:ss')"
    Write-Host "  Log: $logFile"

    # Create a temporary PowerShell script to run the command with proper redirection
    $tmpScript = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.ps1'
    @"
cd '$workDir'
`$output = & '$python' '$fullPath' 2>`&1
`$output | Out-File '$logFile' -Encoding utf8
exit `$LASTEXITCODE
"@ | Set-Content $tmpScript -Encoding utf8

    # Run the temp script and capture exit code
    & powershell -ExecutionPolicy Bypass -File $tmpScript
    $exitCode = $LASTEXITCODE

    # Clean up temp file
    Remove-Item $tmpScript -ErrorAction SilentlyContinue

    if ($exitCode -eq 0) {
        Write-Host "[$i/$total] DONE: $($s.file)  $(Get-Date -Format 'HH:mm:ss')"
    } else {
        Write-Host "[$i/$total] ERROR (exit $exitCode): $($s.file) -- check $errFile"
        # Continue to next script even on error
    }
}

Write-Host ""
Write-Host "All done! $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
