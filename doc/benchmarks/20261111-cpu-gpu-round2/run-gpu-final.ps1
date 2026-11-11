$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'E:\temp\FunAsrNet'
$root = 'E:\temp\FunAsrNet\artifacts\cpu-gpu-round2-20261111'
$dest = Join-Path $root 'gpu-final'
if (Test-Path -LiteralPath $dest) { throw 'Formal output already exists.' }
New-Item -ItemType Directory -Path $dest | Out-Null
$env:FUNASRNET_DISABLE_PROMPT_CACHE = '1'
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $dest 'cpu.json') -Encoding utf8
& nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,temperature.gpu --format=csv | Set-Content -LiteralPath (Join-Path $dest 'gpu-before.csv') -Encoding utf8
foreach ($precision in @('original','q8')) {
    if ($precision -eq 'original') {
        $encoder = 'E:\temp\FunAsrNet\artifacts\cpu-gpu-20261111\original\models\funasr-encoder-original-f32.gguf'
        $decoder = 'E:\temp\FunAsrNet\artifacts\cpu-gpu-20261111\original\models\qwen3-original-f32.gguf'
    } else {
        $encoder = 'E:\temp\FunAsrNet\artifacts\cpp-asr-20261111\models\funasr-encoder-f16.gguf'
        $decoder = 'E:\temp\FunAsrNet\artifacts\cpp-asr-20261111\models\qwen3-0.6b-q8_0.gguf'
    }
    foreach ($backend in @('gpu')) {
        foreach ($inputName in @('sample-6s','sample-repeated-24s')) {
            $inputPath = 'E:\temp\FunAsrNet\artifacts\cpp-asr-20261111\inputs\' + $inputName + '.f32'
            for ($round=0; $round -lt 2; $round++) {
                $order = if ($round -eq 0) { @('before','after') } else { @('after','before') }
                foreach ($variant in $order) {
                    $tag = "$precision-$backend-$inputName-$variant-$round"
                    Write-Output "Running $tag"
                    $runner = if ($variant -eq 'after') { Join-Path $root "final/AsrRunner.dll" } else { Join-Path $root "before/AsrRunner.dll" }
                    $invokeArgs = @($runner,$encoder,$decoder,$inputPath,'3')
                    if ($backend -eq 'gpu') { $invokeArgs += '--gpu' }
                    @{tag=$tag;command=@('dotnet')+$invokeArgs;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $dest 'commands.jsonl') -Encoding utf8
                    & dotnet @invokeArgs > (Join-Path $dest "$tag.jsonl")
                    if ($LASTEXITCODE -ne 0) { throw "Failed $tag, exit $LASTEXITCODE" }
                    & nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader | Add-Content -LiteralPath (Join-Path $dest 'gpu-progress.csv') -Encoding utf8
                }
            }
        }
    }
}
Write-Output 'Formal CPU/GPU and original/Q8 comparison completed.'
