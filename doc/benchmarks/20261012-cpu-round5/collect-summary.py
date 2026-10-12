import hashlib,json,math,statistics
from pathlib import Path
root=Path.cwd();source=root/'artifacts/cpu-round5/final';out=root/'doc/benchmarks/20261012-cpu-round5';out.mkdir(parents=True,exist_ok=True)
previous=json.loads((root/'doc/benchmarks/20261012-cpu-round4/summary.json').read_text(encoding='utf-8'))
groups=[];warmup=0;measured=0
subjects={'tts':['huayan','aishell3'],'asr':['sip-short-pcma','sip-zh-pcmu','sip-repeated-pcma','sip-official-short','sip-official-pcma']}
for kind,names in subjects.items():
    for name in names:
        for variant,blocks in [('before','af'),('after','be'),('native','cd')]:
            rows=[];loads=[]
            for block in blocks:
                file=source/(kind+'-'+name+'-'+block+'-'+variant+'.jsonl')
                (out/file.name).write_bytes(file.read_bytes())
                records=[json.loads(line) for line in file.read_text(encoding='utf-8-sig').splitlines()]
                assert len(records)==(12 if kind=='tts' else 5 if name=='sip-repeated-pcma' else 7),file
                rows += [r for r in records if r['kind']==kind and not r['warmup']]
                warmup += sum(r.get('warmup',False) for r in records)
                loads += [r['seconds'] for r in records if r['kind']=='load']
                if kind=='asr':
                    assert records[0]['promptCache'] is False
                    if variant=='native':
                        log=file.with_suffix('.stderr').read_text(encoding='utf-8')
                        log=log.replace(str(root/'artifacts/local-models'),'models').replace(str(root/'artifacts/native-gpu/llama-source'),'llama-source')
                        (out/file.with_suffix('.stderr').name).write_text('\n'.join(line.rstrip() for line in log.splitlines())+'\n',encoding='utf-8',newline='\n')
            times=sorted(r['seconds'] for r in rows)
            group=dict(kind=kind,subject=name,variant=variant,count=len(rows),medianSeconds=statistics.median(times),p95Seconds=times[math.ceil(.95*len(times))-1],minimumSeconds=min(times),maximumSeconds=max(times),audioSeconds=rows[0]['audioSeconds'],peakRssBytes=max(r['peakRssBytes'] for r in rows),loadSecondsMedian=statistics.median(loads))
            if kind=='asr':
                group['texts']=sorted(set(r['Text'] for r in rows))
                group['encoderMedianSeconds']=statistics.median(r.get('EncoderSeconds',r.get('encoderSeconds')) for r in rows)
                group['decoderMedianSeconds']=statistics.median(r.get('DecoderSeconds',r.get('decoderSeconds')) for r in rows)
            else:
                group['samples']=rows[0]['samples']
                if 'allocatedBytes' in rows[0]:group['allocatedBytesMedian']=statistics.median(r['allocatedBytes'] for r in rows)
            groups.append(group);measured+=len(rows)
assert measured==258 and warmup==42
checks={}
for name in subjects['asr']:
    texts=[g['texts'] for g in groups if g['kind']=='asr' and g['subject']==name]
    checks[name]={'exactMatch':all(t==texts[0] for t in texts),'textsByVariant':{g['variant']:g['texts'] for g in groups if g['kind']=='asr' and g['subject']==name}}
    assert checks[name]['exactMatch'],(name,texts)
parity={}
for file in source.glob('parity-*.jsonl'):
    (out/file.name).write_bytes(file.read_bytes());record=json.loads(file.read_text(encoding='utf-8-sig'));assert record['passed'];parity[file.stem]=record
inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'artifacts/native-gpu/inputs').iterdir() if p.stem in subjects['asr']+subjects['tts']}
summary=dict(date='2026-10-12',platform='Windows x64 / Intel Core i7-12700',runtime='10.0.12',threads=4,affinity='0x55',beforeSourceCommit='3a6105931c37fc2cbe027987930d3c388d7db730',afterSource='3a61059 plus shared Q8 projections, sequential FP32 microkernel with preserved K128 and exact F16 loading; CPU convolution fusion and layout operations',buildManifest=json.loads((source/'build-manifest.json').read_text()),funAsrCommit=previous['funAsrCommit'],llamaCommit=previous['llamaCommit'],nativeCpuBuild=previous['nativeCpuBuild'],ortVersion='1.23.2',ortProvider='CPUExecutionProvider',modelSha256=previous['modelSha256'],inputSha256=inputs,gpuEnabled=False,asrPromptCache=False,maxTokens=128,order=['before-a','after-b','native-c','native-d','after-e','before-f'],measured=measured,warmup=warmup,percentileMethod='nearest rank',groups=groups,ttsParity=parity,asrTextChecks=checks,nativeLogPathRedaction='Local model/source prefixes replaced; stderr trailing whitespace normalized; numerical JSONL unchanged.',extraInputScope='Official clean PCM and its PCMA version contain the same utterance as the existing short example; they add codec coverage, not independent linguistic coverage.',targetHalfNativeLatencyMet=False)
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for kind,names in subjects.items():
    for name in names:
        g={g['variant']:g for g in groups if g['kind']==kind and g['subject']==name};a,b,c=[g[v]['medianSeconds'] for v in ['before','after','native']]
        print(kind,name,'before',round(a,6),'after',round(b,6),'native',round(c,6),'ratio',round(b/c,3),'reduced%',round((1-b/a)*100,2))
