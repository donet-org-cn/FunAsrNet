$ErrorActionPreference='Stop'
Set-Location -LiteralPath 'E:\temp\FunAsrNet'
$root='E:\temp\FunAsrNet\artifacts\cpu-gpu-20261111'
$dest=Join-Path $root 'native-confirmation'
if(Test-Path -LiteralPath $dest){throw 'Output exists.'}
New-Item -ItemType Directory -Path $dest | Out-Null
$encoder='E:\temp\FunAsrNet\artifacts\cpp-asr-20261111\models\funasr-encoder-f16.gguf'
$decoder='E:\temp\FunAsrNet\artifacts\cpp-asr-20261111\models\qwen3-0.6b-q8_0.gguf'
foreach($subject in @('sample-6s','sample-repeated-24s')) {
    $inputPath='E:\temp\FunAsrNet\artifacts\cpp-asr-20261111\inputs\'+$subject+'.f32'
    $index=0
    foreach($engine in @('cpp','csharp','csharp','cpp')) {
        $tag="$subject-$engine-$index"
        Write-Output "Running $tag"
        if($engine -eq 'cpp') {
            $exe='E:\temp\FunAsrNet\artifacts\cpp-asr-20261111\native-cpu\build\funasr-native-cpu-reference.exe'
            $callArgs=@($encoder,$decoder,$inputPath,'3')
        } else {
            $exe='dotnet'
            $callArgs=@((Join-Path $root 'after/AsrRunner.dll'),$encoder,$decoder,$inputPath,'3')
        }
        @{tag=$tag;command=@($exe)+$callArgs;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $dest 'commands.jsonl') -Encoding utf8
        & $exe @callArgs > (Join-Path $dest "$tag.jsonl") 2> (Join-Path $dest "$tag.stderr")
        if($LASTEXITCODE -ne 0){throw "Failed $tag"}
        $index++
    }
}
Write-Output 'Native CPU comparison completed.'
