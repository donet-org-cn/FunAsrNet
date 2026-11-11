param(
    [string]$Version,
    [Parameter(Mandatory = $true)][string]$Packages,
    [int]$WaitSeconds = 900
)
. "$PSScriptRoot/ReleaseCommon.ps1"
$Version = Get-ReleaseVersion $Version
$Packages = (Resolve-Path -LiteralPath $Packages).Path
$manifest = Test-ReleaseManifest $Packages $Version
$downloadDirectory = Join-Path $Packages 'public-downloads'
New-Item -ItemType Directory -Force -Path $downloadDirectory | Out-Null
$deadline = [DateTime]::UtcNow.AddSeconds($WaitSeconds)
$verified = @()
foreach ($package in $manifest.packages) {
    $id = $package.id.ToLowerInvariant()
    $url = "https://api.nuget.org/v3-flatcontainer/$id/$Version/$id.$Version.nupkg"
    $download = Join-Path $downloadDirectory $package.file
    while ($true) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $download -TimeoutSec 30
            break
        } catch {
            if ([DateTime]::UtcNow -ge $deadline) { throw "NuGet package is not downloadable before the deadline: $($package.id). $($_.Exception.Message)" }
            Write-Output "Waiting for NuGet indexing: $($package.id) $Version."
            Start-Sleep -Seconds 15
        }
    }
    $actual = @(Get-PackageEntries $download)
    if ($actual.Count -ne @($package.entries).Count) { throw "NuGet payload entry count differs: $($package.id)" }
    foreach ($expected in $package.entries) {
        $entry = @($actual | Where-Object { $_.path -ceq $expected.path })
        if ($entry.Count -ne 1 -or $entry[0].bytes -ne $expected.bytes -or $entry[0].sha256 -cne $expected.sha256) {
            throw "NuGet payload differs: $($package.id)/$($expected.path)"
        }
    }
    $verified += [pscustomobject]@{ id = $package.id; version = $Version; url = $url; payloadMatches = $true; checkedEntries = $actual.Count; downloadedSha256 = (Get-FileHash -LiteralPath $download -Algorithm SHA256).Hash.ToLowerInvariant() }
    Write-Output "Verified public NuGet payload: $($package.id) $Version."
}
Write-ReleaseText (Join-Path $Packages 'published-verification.json') (($verified | ConvertTo-Json -Depth 5) + "`n")
