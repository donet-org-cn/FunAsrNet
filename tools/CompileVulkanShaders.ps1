param(
    [Parameter(Mandatory = $true)][string]$Compiler,
    [string]$OutputDirectory,
    [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repository 'artifacts/vulkan-shaders' }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$compilerPath = (Resolve-Path -LiteralPath $Compiler).Path
& $compilerPath --version
if ($LASTEXITCODE -ne 0) { throw '无法运行 GLSL 编译器。' }
$variants = @(
    @{ Name = 'SpeechShader'; Method = 'Binary'; Define = $null; Suffix = 'Binary' },
    @{ Name = 'QuantizedShader'; Method = 'Binary'; Define = $null; Suffix = 'Binary' },
    @{ Name = 'CooperativeQuantizedShader'; Method = 'Binary'; Define = $null; Suffix = 'Binary' },
    @{ Name = 'CooperativeQuantizedShader'; Method = 'IntegerDotBinary'; Define = '-DINTEGER_DOT=1'; Suffix = 'IntegerDotBinary' },
    @{ Name = 'FloatVectorShader'; Method = 'WideBinary'; Define = '-DWIDE=1'; Suffix = 'WideBinary' },
    @{ Name = 'AttentionTiledShader'; Method = 'Binary'; Define = $null; Suffix = 'Binary' },
    @{ Name = 'AttentionShader'; Method = 'Binary'; Define = $null; Suffix = 'Binary' },
    @{ Name = 'FloatLinearShader'; Method = 'Binary'; Define = $null; Suffix = 'Binary' },
    @{ Name = 'FloatVectorShader'; Method = 'Binary'; Define = $null; Suffix = 'Binary' }
)
foreach ($variant in $variants) {
    $sourcePath = Join-Path $repository ("src/FunAsrNet.Vulkan/" + $variant.Name + '.cs')
    $sourceText = [IO.File]::ReadAllText($sourcePath)
    $match = [regex]::Match($sourceText, '(?s)Source\s*=\s*"""(.*?)"""')
    if (-not $match.Success) { throw "缺少 GLSL 来源 $sourcePath" }
    $glslPath = Join-Path $OutputDirectory ($variant.Name + '.' + $variant.Suffix + '.comp')
    $spirvPath = Join-Path $OutputDirectory ($variant.Name + '.' + $variant.Suffix + '.spv')
    [IO.File]::WriteAllText($glslPath, $match.Groups[1].Value.Replace("`r`n", "`n"), [Text.UTF8Encoding]::new($false))
    $compilerArgs = @('-V', '--target-env', 'vulkan1.1', '-S', 'comp')
    if ($variant.Define) { $compilerArgs += $variant.Define }
    $compilerArgs += @($glslPath, '-o', $spirvPath)
    & $compilerPath @compilerArgs
    if ($LASTEXITCODE -ne 0) { throw "编译失败 $glslPath" }
    $binary = [Convert]::ToBase64String([IO.File]::ReadAllBytes($spirvPath))
    $generated = @"
namespace FunAsrNet.Vulkan;

/// <summary>由同名 GLSL 来源离线编译得到的 SPIR-V。</summary>
internal static partial class $($variant.Name)
{
    /// <summary>返回离线编译的计算程序，运行时无需编译器。</summary>
    /// <returns>创建 Vulkan 模块所需的独立字节数组。</returns>
    internal static byte[] $($variant.Method)() => Convert.FromBase64String("$binary");
}
"@
    $targetPath = Join-Path $repository ("src/FunAsrNet.Vulkan/" + $variant.Name + '.' + $variant.Suffix + '.cs')
    if ($VerifyOnly) {
        $existing = [regex]::Match([IO.File]::ReadAllText($targetPath), 'FromBase64String\("([^"]+)"\)')
        if (-not $existing.Success -or $existing.Groups[1].Value -cne $binary) { throw "来源与嵌入 SPIR-V 不一致 $targetPath" }
        Write-Output "Verified $($variant.Name).$($variant.Suffix)"
    } else {
        [IO.File]::WriteAllText($targetPath, $generated.Replace("`r`n", "`n").TrimEnd() + "`n", [Text.UTF8Encoding]::new($false))
    }
    Get-FileHash -LiteralPath $spirvPath -Algorithm SHA256
}
