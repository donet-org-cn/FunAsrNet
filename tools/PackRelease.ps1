param(
    [string]$Version,
    [string]$Output,
    [string]$DotNetSdk
)
. "$PSScriptRoot/ReleaseCommon.ps1"
$Version = Get-ReleaseVersion $Version
if (!$Output) { $Output = Join-Path $releaseRoot "artifacts/packages/$Version" }
$Output = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$expectedNames = @("FunAsrNet.$Version.nupkg", "FunAsrNet.$Version.snupkg", "FunAsrNet.Vulkan.$Version.nupkg", "FunAsrNet.Vulkan.$Version.snupkg")
foreach ($existing in Get-ChildItem -LiteralPath $Output -File | Where-Object { $_.Extension -in @('.nupkg', '.snupkg') }) {
    if ($existing.Name -notin $expectedNames) { throw "Output contains an unrelated package: $($existing.Name)" }
}
foreach ($project in @('FunAsrNet', 'FunAsrNet.Vulkan')) {
    Invoke-ReleaseDotNet -Sdk $DotNetSdk -Arguments @('pack', (Join-Path $releaseRoot "src/$project/$project.csproj"), '-c', 'Release', '-o', $Output, "-p:Version=$Version", '-p:ContinuousIntegrationBuild=true', '--nologo')
}
$packages = @()
foreach ($id in @('FunAsrNet', 'FunAsrNet.Vulkan')) {
    $file = "$id.$Version.nupkg"
    $path = Join-Path $Output $file
    $zip = [IO.Compression.ZipFile]::OpenRead($path)
    try {
        foreach ($required in @("$id.nuspec", 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', "lib/net10.0/$id.dll", "lib/net10.0/$id.xml")) {
            if (!$zip.GetEntry($required)) { throw "$file is missing $required." }
        }
        if ($id -eq 'FunAsrNet.Vulkan' -and !$zip.GetEntry('licenses/SimdPaddleOCR/LICENSE')) { throw 'Missing Vulkan binding license.' }
        if (@($zip.Entries | Where-Object { $_.FullName -match '\.(gguf|onnx|safetensors)$|^runtimes/' }).Count -ne 0) { throw 'Unexpected model or native runtime payload.' }
        if (@($zip.Entries | Where-Object { $_.FullName.EndsWith('.dll') }).Count -ne 1) { throw 'Unexpected assemblies in package.' }
        $reader = [IO.StreamReader]::new($zip.GetEntry("$id.nuspec").Open())
        try { [xml]$spec = $reader.ReadToEnd() } finally { $reader.Dispose() }
        $metadata = $spec.package.metadata
        if ($metadata.id -cne $id -or $metadata.version -cne $Version -or $metadata.license.InnerText -ne 'Apache-2.0') {
            throw "Incorrect NuGet identity, version or license: $file"
        }
        if ($metadata.repository.url -ne 'https://github.com/donet-org-cn/FunAsrNet') { throw 'Incorrect source repository.' }
        $dependencies = @($metadata.dependencies.group.dependency | Where-Object { $_ })
        if ($id -eq 'FunAsrNet' -and $dependencies.Count -ne 0) { throw 'Core package must not acquire runtime dependencies.' }
        if ($id -eq 'FunAsrNet.Vulkan') {
            if ($dependencies.Count -ne 1 -or $dependencies[0].id -ne 'FunAsrNet') { throw 'Unexpected GPU dependency graph.' }
            if (($dependencies[0].version -replace '\s', '') -notin @($Version, "[$Version,)", "[$Version]")) { throw 'GPU dependency version differs.' }
        }
    } finally { $zip.Dispose() }
    $symbols = [IO.Compression.ZipFile]::OpenRead((Join-Path $Output "$id.$Version.snupkg"))
    try {
        if (!$symbols.GetEntry("lib/net10.0/$id.pdb")) { throw "Missing portable PDB for $id." }
    } finally { $symbols.Dispose() }
    $packages += [pscustomobject]@{ id = $id; version = $Version; file = $file; entries = @(Get-PackageEntries $path) }
}
$files = @($expectedNames | Sort-Object | ForEach-Object {
    $path = Join-Path $Output $_
    [pscustomobject]@{ path = $_; bytes = (Get-Item -LiteralPath $path).Length; sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
})
$commit = & git -C $releaseRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot determine source commit.' }
$manifest = [ordered]@{ version = $Version; commit = $commit; files = $files; packages = $packages }
Write-ReleaseText (Join-Path $Output 'manifest.json') (($manifest | ConvertTo-Json -Depth 8) + "`n")
Write-ReleaseText (Join-Path $Output 'SHA256SUMS.txt') (($files | ForEach-Object { "$($_.sha256)  $($_.path)" }) -join "`n")
Test-ReleaseManifest $Output $Version | Out-Null
Write-Output "Verified two NuGet packages and two symbol packages for $Version."
