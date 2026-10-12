import json, os, statistics, subprocess
from pathlib import Path
root = Path.cwd()
out = root/'artifacts/cpu-round4/final/confirmation'
out.mkdir(exist_ok=False)
env = {k:v for k,v in os.environ.items() if not k.startswith('FUNASRNET_')}
before = root/'artifacts/cpu-round4/before/FunAsrNet.Benchmarks.dll'
after = root/'tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll'
ort = root/'tools/FunAsrNet.NativeReference/bin/Release/net10.0/FunAsrNet.NativeReference.dll'
models = root/'artifacts/local-models'
inputs = root/'artifacts/native-gpu/inputs'
for subject in ['huayan','aishell3']:
    for block,variant in zip('abcdef',['native','after','before','before','after','native']):
        name = 'tts-'+subject+'-'+block+'-'+variant
        wave = out/(name+'.f32')
        if variant == 'native':
            model = models/('zh_CN-huayan-medium.onnx' if subject == 'huayan' else 'aishell3/model.onnx')
            command = ['dotnet',ort,model,inputs/(subject+'.i64'),subject,'10',wave,'--provider','cpu','--threads','4','--affinity','0x55']
        else:
            command = ['dotnet',before if variant == 'before' else after,'tts',models,inputs/(subject+'.i64'),subject,'10',wave,'--threads','4','--affinity','0x55']
        with (out/(name+'.jsonl')).open('wb') as stdout:
            subprocess.run([str(p) for p in command],stdout=stdout,env=env,check=True,timeout=300)
        rows=[json.loads(line) for line in (out/(name+'.jsonl')).read_text(encoding='utf-8-sig').splitlines()]
        print(name,statistics.median(r['seconds'] for r in rows if r['kind']=='tts' and not r['warmup']),flush=True)
    for reference in ['f-native','d-before']:
        with (out/('parity-'+subject+'-'+reference+'.jsonl')).open('wb') as stdout:
            subprocess.run(['dotnet',str(after),'compare',str(out/('tts-'+subject+'-'+reference+'.f32')),str(out/('tts-'+subject+'-e-after.f32'))],stdout=stdout,check=True)
print('Confirmation complete',flush=True)
