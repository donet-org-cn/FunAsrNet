$ErrorActionPreference = 'Stop'
$releaseRoot = Split-Path -Parent $PSScriptRoot
Add-Type -AssemblyName System.IO.Compression.FileSystem

# 使用仓库版本作为唯一来源；只接受不带预发布后缀的正式 SemVer。
function Get-ReleaseVersion {
    param([string]$Requested)
    [xml]$properties = Get-Content -LiteralPath (Join-Path $releaseRoot 'Directory.Build.props') -Raw
    $configured = $properties.SelectSingleNode('/Project/PropertyGroup/Version').InnerText
    if (!$Requested) { $Requested = $configured }
    if ($Requested -notmatch '^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$') { throw 'A stable major.minor.patch version is required.' }
    if ($Requested -cne $configured) { throw "Requested version $Requested differs from Directory.Build.props ($configured)." }
    return $Requested
}

# 可显式选择已安装的 SDK DLL；默认遵循仓库 global.json。
function Invoke-ReleaseDotNet {
    param([string[]]$Arguments, [string]$Sdk)
    if ($Sdk) { & dotnet $Sdk @Arguments } else { & dotnet @Arguments }
    if ($LASTEXITCODE -ne 0) { throw "dotnet failed with exit code $LASTEXITCODE." }
}

# 归档与清单统一写为无 BOM 的 UTF-8 和 LF。
function Write-ReleaseText {
    param([string]$Path, [string]$Text)
    [IO.File]::WriteAllText($Path, $Text.Replace("`r`n", "`n"), [Text.UTF8Encoding]::new($false))
}

# NuGet 仓库签名只新增 .signature.p7s；比对其余每个 ZIP 成员的原始内容。
function Get-PackageEntries {
    param([string]$Path)
    $zip = [IO.Compression.ZipFile]::OpenRead($Path)
    try {
        foreach ($entry in $zip.Entries | Sort-Object FullName) {
            if ($entry.FullName.EndsWith('/') -or $entry.FullName -eq '.signature.p7s') { continue }
            $stream = $entry.Open()
            $hasher = [Security.Cryptography.SHA256]::Create()
            try {
                $digest = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
                [pscustomobject]@{ path = $entry.FullName; bytes = $entry.Length; sha256 = $digest }
            } finally { $hasher.Dispose(); $stream.Dispose() }
        }
    } finally { $zip.Dispose() }
}

# 验证交接给安装测试和发布作业的四个包没有被重建或替换。
function Test-ReleaseManifest {
    param([string]$Packages, [string]$Version)
    $manifest = Get-Content -LiteralPath (Join-Path $Packages 'manifest.json') -Raw | ConvertFrom-Json
    if ($manifest.version -cne $Version -or @($manifest.files).Count -ne 4 -or @($manifest.packages).Count -ne 2) {
        throw 'Unexpected package manifest.'
    }
    $expectedNames = @("FunAsrNet.$Version.nupkg", "FunAsrNet.$Version.snupkg", "FunAsrNet.Vulkan.$Version.nupkg", "FunAsrNet.Vulkan.$Version.snupkg")
    if (@(Compare-Object $expectedNames @($manifest.files.path)).Count -ne 0) { throw 'Unexpected package file list.' }
    foreach ($file in $manifest.files) {
        $path = Join-Path $Packages $file.path
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $file.sha256) {
            throw "Package hash mismatch: $($file.path)"
        }
    }
    return $manifest
}
