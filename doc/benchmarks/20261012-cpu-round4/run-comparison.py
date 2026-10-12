import ctypes
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

ctypes.windll.kernel32.SetErrorMode(0x8003)
root = Path.cwd()
output = root / 'artifacts/cpu-round4/final'
output.mkdir(exist_ok=True)
current = root / 'tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll'
before = root / 'artifacts/cpu-round4/before/FunAsrNet.Benchmarks.dll'
ort = root / 'tools/FunAsrNet.NativeReference/bin/Release/net10.0/FunAsrNet.NativeReference.dll'
cpp = root / 'artifacts/native-cpu/build-vnni/funasr-native-cpu-reference.exe'
models = root / 'artifacts/local-models'
inputs = root / 'artifacts/native-gpu/inputs'
environment = {key:value for key,value in os.environ.items() if not key.startswith('FUNASRNET_')}
environment.update(FUNASRNET_DISABLE_PROMPT_CACHE='1',OMP_NUM_THREADS='4')
order = [('a','before'),('b','after'),('c','native'),('d','native'),('e','after'),('f','before')]
manifest = {'beforeCore':hashlib.sha256((before.parent/'FunAsrNet.dll').read_bytes()).hexdigest(),'afterCore':hashlib.sha256((current.parent/'FunAsrNet.dll').read_bytes()).hexdigest(),'nativeCpp':hashlib.sha256(cpp.read_bytes()).hexdigest()}
(output/'build-manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
def run(name,command):
    print('Starting '+name,flush=True)
    with (output/(name+'.jsonl')).open('wb') as stdout,(output/(name+'.stderr')).open('wb') as stderr:
        subprocess.run([str(part) for part in command],stdout=stdout,stderr=stderr,env=environment,timeout=1200,check=True)
    rows = [json.loads(line) for line in (output/(name+'.jsonl')).read_text(encoding='utf-8-sig').splitlines()]
    measured = [row['seconds'] for row in rows if row['kind'] in ('asr','tts') and not row['warmup']]
    print(name+' median='+str(statistics.median(measured)),flush=True)
for kind,subjects in [('tts',['huayan','aishell3']),('asr',['sip-short-pcma','sip-zh-pcmu','sip-repeated-pcma'])]:
    for subject in subjects:
        repeats = '3' if subject=='sip-repeated-pcma' else '5'
        for block,variant in order:
            name = kind+'-'+subject+'-'+block+'-'+variant
            wave = output/(name+'.f32')
            if kind=='tts':
                ids = inputs/(subject+'.i64')
                if variant=='native':
                    model = models/('zh_CN-huayan-medium.onnx' if subject=='huayan' else 'aishell3/model.onnx')
                    command = ['dotnet',ort,model,ids,subject,repeats,wave,'--provider','cpu','--threads','4','--affinity','0x55']
                else:
                    command = ['dotnet',before if variant=='before' else current,'tts',models,ids,subject,repeats,wave,'--threads','4','--affinity','0x55']
            else:
                pcm = inputs/(subject+'.f32')
                if variant=='native':
                    command = [cpp,models/'funasr-encoder-f16.gguf',models/'qwen3-0.6b-q8_0.gguf',pcm,repeats]
                else:
                    command = ['dotnet',before if variant=='before' else current,'asr',models,pcm,repeats,'--affinity','0x55']
            run(name,command)
        if kind=='tts':
            for reference in ['d-native','f-before']:
                with (output/('parity-'+subject+'-'+reference+'.jsonl')).open('wb') as stdout:
                    subprocess.run(['dotnet',str(current),'compare',str(output/('tts-'+subject+'-'+reference+'.f32')),str(output/('tts-'+subject+'-e-after.f32'))],stdout=stdout,check=True)
print('COMPLETE',flush=True)
