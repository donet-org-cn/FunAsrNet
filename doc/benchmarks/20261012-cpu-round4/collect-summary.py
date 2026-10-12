import hashlib
import json
import math
import statistics
from pathlib import Path

root = Path.cwd()
source = root/'artifacts/cpu-round4/final'
output = root/'doc/benchmarks/20261012-cpu-round4'
output.mkdir(parents=True,exist_ok=True)
previous = json.loads((root/'doc/benchmarks/20261012-cpu-native/summary.json').read_text(encoding='utf-8'))
manifest = json.loads((source/'build-manifest.json').read_text())
groups=[]
warmup=0
measured=0
for kind,subjects in [('tts',['huayan','aishell3']),('asr',['sip-short-pcma','sip-zh-pcmu','sip-repeated-pcma'])]:
    for subject in subjects:
        for variant,blocks in [('before',['a','f']),('after',['b','e']),('native',['c','d'])]:
            rows=[]
            for block in blocks:
                name=kind+'-'+subject+'-'+block+'-'+variant
                file=source/(name+'.jsonl')
                (output/file.name).write_bytes(file.read_bytes())
                records=[json.loads(line) for line in file.read_text(encoding='utf-8-sig').splitlines()]
                assert len(records)==(5 if subject=='sip-repeated-pcma' else 7)
                rows += [r for r in records if r['kind']==kind and not r['warmup']]
                warmup += sum(r.get('warmup',False) for r in records)
                if kind=='asr':
                    assert records[0]['promptCache'] is False
                    if variant=='native':
                        log=(source/(name+'.stderr')).read_text(encoding='utf-8')
                        log=log.replace(str(root/'artifacts/local-models'),'models').replace(str(root/'artifacts/native-gpu/llama-source'),'llama-source')
                        (output/(name+'.stderr')).write_bytes(('\n'.join(line.rstrip() for line in log.splitlines())+'\n').encode())
            if kind == 'tts':
                for file in sorted((source/'confirmation').glob('tts-'+subject+'-*-'+variant+'.jsonl')):
                    destination = output/'confirmation'/file.name
                    destination.parent.mkdir(exist_ok=True)
                    destination.write_bytes(file.read_bytes())
                    records = [json.loads(line) for line in file.read_text(encoding='utf-8-sig').splitlines()]
                    assert len(records) == 12
                    rows += [r for r in records if r['kind'] == kind and not r['warmup']]
                    warmup += sum(r.get('warmup', False) for r in records)
            seconds=sorted(r['seconds'] for r in rows)
            group=dict(kind=kind,subject=subject,variant=variant,count=len(rows),medianSeconds=statistics.median(seconds),p95Seconds=seconds[math.ceil(.95*len(seconds))-1],minimumSeconds=min(seconds),maximumSeconds=max(seconds),audioSeconds=rows[0]['audioSeconds'],peakRssBytes=max(r['peakRssBytes'] for r in rows))
            if kind=='asr':
                group['texts']=sorted(set(r['Text'] for r in rows))
                group['encoderMedianSeconds']=statistics.median(r.get('EncoderSeconds',r.get('encoderSeconds')) for r in rows)
                group['decoderMedianSeconds']=statistics.median(r.get('DecoderSeconds',r.get('decoderSeconds')) for r in rows)
            else:
                group['samples']=rows[0]['samples']
                if 'allocatedBytes' in rows[0]:
                    group['allocatedBytesMedian']=statistics.median(row['allocatedBytes'] for row in rows)
            groups.append(group)
            measured+=len(rows)
parity={}
for file in list(source.glob('parity-*.jsonl')) + list((source/'confirmation').glob('parity-*.jsonl')):
    (output/file.relative_to(source)).write_bytes(file.read_bytes())
    record=json.loads(file.read_text(encoding='utf-8-sig'))
    assert record['passed']
    parity[str(file.relative_to(source))]=record
assert measured==258 and warmup==42
checks={}
for subject in ['sip-short-pcma','sip-zh-pcmu','sip-repeated-pcma']:
    texts=[g['texts'] for g in groups if g['kind']=='asr' and g['subject']==subject]
    assert all(t==texts[0] for t in texts)
    checks[subject]=dict(exactMatch=True,texts=texts[0])
summary=dict(date='2026-10-12',platform='Windows x64 / Intel Core i7-12700',runtime='10.0.12',threads=4,affinity='0x55',beforeSourceCommit='ec19a4eed9ace38567cb7b69c18c9d17217e850e',afterSource='ec19a4e plus exact x64 quantization and grouped Q8 reductions, FP32 blocking, owned TTS buffers',buildManifest=manifest,funAsrCommit=previous['funAsrCommit'],llamaCommit=previous['llamaCommit'],nativeCpuBuild=dict(msvc='19.44.35207',avx2=True,avxVnni=True,fma=True,openmp=True,gpuLayers=0,cuda=False,vulkan=False),ortVersion='1.23.2',ortProvider='CPUExecutionProvider',modelSha256=previous['modelSha256'],inputSha256=previous['inputSha256'],gpuEnabled=False,asrPromptCache=False,maxTokens=128,order=['before-a','after-b','native-c','native-d','after-e','before-f'],measured=measured,warmup=warmup,percentileMethod='nearest rank',groups=groups,ttsParity=parity,asrTextChecks=checks,nativeLogPathRedaction='Local model/source prefixes replaced; diagnostic stderr whitespace normalized; numerical JSONL unchanged.',targetFullyMet=False)
(output/'summary.json').write_bytes((json.dumps(summary,ensure_ascii=False,indent=2)+'\n').encode())
for kind,subject in [(g['kind'],g['subject']) for g in groups if g['variant']=='after']:
    group={g['variant']:g for g in groups if g['kind']==kind and g['subject']==subject}
    before,after,native=[group[v]['medianSeconds'] for v in ['before','after','native']]
    print(kind,subject,'before',round(before,6),'after',round(after,6),'native',round(native,6),'ratio',round(after/native,3),'improved%',round((1-after/before)*100,2))
