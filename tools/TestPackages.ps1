param(
    [string]$Version,
    [Parameter(Mandatory = $true)][string]$Packages,
    [string]$Output,
    [string]$DotNetSdk
)
. "$PSScriptRoot/ReleaseCommon.ps1"
$Version = Get-ReleaseVersion $Version
$Packages = (Resolve-Path -LiteralPath $Packages).Path
Test-ReleaseManifest $Packages $Version | Out-Null
if (!$Output) { $Output = Join-Path $releaseRoot "artifacts/package-checks/$Version-$([Guid]::NewGuid().ToString('N'))" }
$Output = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$project = Join-Path $Output 'PackageChecks.csproj'
Copy-Item -LiteralPath (Join-Path $releaseRoot 'tests/FunAsrNet.PackageChecks/FunAsrNet.PackageChecks.csproj') -Destination $project
Copy-Item -LiteralPath (Join-Path $releaseRoot 'tests/FunAsrNet.PackageChecks/Program.cs') -Destination (Join-Path $Output 'Program.cs')
$packageCache = Join-Path $Output '.packages'
Invoke-ReleaseDotNet -Sdk $DotNetSdk -Arguments @('restore', $project, '--source', $Packages, '--packages', $packageCache, "-p:Version=$Version", '-p:NuGetAudit=false', '--nologo')
Invoke-ReleaseDotNet -Sdk $DotNetSdk -Arguments @('build', $project, '-c', 'Release', '--no-restore', "-p:Version=$Version", '--nologo')
Invoke-ReleaseDotNet -Arguments @((Join-Path $Output 'bin/Release/net10.0/PackageChecks.dll'))
Write-Output "Consumed the frozen $Version packages from $Packages using a separate package cache."
