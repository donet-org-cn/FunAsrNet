"""Windows CPU comparison; run after publishing the JIT and AOT executables.

Keeps every sample, launches only one inference process at a time, and leaves
models and waveforms outside the public evidence. Paths are relative to repo.
"""
import argparse
import array
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stats(values):
    values = sorted(values)
    return dict(count=len(values), median=statistics.median(values),
                p95=values[math.ceil(.95 * len(values)) - 1], minimum=values[0], maximum=values[-1])


def wave_parity(reference, actual):
    left, right = array.array('f'), array.array('f')
    left.frombytes(reference.read_bytes())
    right.frombytes(actual.read_bytes())
    assert len(left) == len(right) and len(left) > 0
    assert all(math.isfinite(x) for x in left) and all(math.isfinite(x) for x in right)
    errors = [abs(a - b) for a, b in zip(left, right)]
    maximum = max(errors)
    rmse = math.sqrt(sum(x * x for x in errors) / len(errors))
    return dict(samples=len(left), maximumAbsoluteError=maximum, rmse=rmse,
                referenceSha256=digest(reference), actualSha256=digest(actual),
                passed=maximum <= .001 and rmse <= .0001)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=Path('artifacts/local-models'))
    parser.add_argument('--inputs', type=Path, default=Path('artifacts/native-gpu/inputs'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/aot-comparison/formal'))
    parser.add_argument('--jit', type=Path, default=Path('artifacts/aot-comparison/jit/FunAsrNet.AotBenchmarks.exe'))
    parser.add_argument('--aot', type=Path, default=Path('artifacts/aot-comparison/aot/FunAsrNet.AotBenchmarks.exe'))
    parser.add_argument('--current', type=Path, default=Path('tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll'))
    parser.add_argument('--ort', type=Path, default=Path('tools/FunAsrNet.NativeReference/bin/Release/net10.0/FunAsrNet.NativeReference.dll'))
    parser.add_argument('--cpp', type=Path, default=Path('artifacts/native-cpu/build-vnni/funasr-native-cpu-reference.exe'))
    options = parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('This protocol requires the Windows affinity/native reference build.')
    ctypes.windll.kernel32.SetErrorMode(0x8003)
    output = options.output
    output.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(('FUNASRNET_', 'COMPLUS_', 'DOTNET_'))}
    env.update(FUNASRNET_DISABLE_PROMPT_CACHE='1', OMP_NUM_THREADS='4')
    commands, all_runs, groups, parities = {}, [], [], {}
    order = ['current', 'jit', 'aot', 'native', 'native', 'aot', 'jit', 'current']
    model_paths = {
        'encoder': options.models / 'funasr-encoder-f16.gguf',
        'decoder': options.models / 'qwen3-0.6b-q8_0.gguf',
        'huayan': options.models / 'zh_CN-huayan-medium.onnx',
        'aishell3': options.models / 'aishell3/model.onnx',
    }
    binaries = {name: getattr(options, name) for name in ('jit', 'aot', 'current', 'ort', 'cpp')}
    binaries['jitCore'] = options.jit.parent / 'FunAsrNet.dll'
    binaries['currentCore'] = options.current.parent / 'FunAsrNet.dll'
    source_paths = [Path('Directory.Build.props')] + [
        path for directory in ('src/FunAsrNet', 'tools/FunAsrNet.AotBenchmarks')
        for path in Path(directory).rglob('*')
        if path.suffix in ('.cs', '.csproj', '.py') and not {'obj', 'bin'}.intersection(path.parts)]
    manifest = dict(sourceBaseCommit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    modelSha256={name: digest(path) for name, path in model_paths.items()},
                    inputSha256={path.name: digest(path) for path in options.inputs.iterdir() if path.suffix in ('.f32', '.i64')},
                    binarySha256={name: digest(path) for name, path in binaries.items()},
                    sourceSha256={path.as_posix(): digest(path) for path in source_paths},
                    aot=dict(publishAot=True, instructionSet='native', optimizationPreference='Speed'),
                    jit=dict(tieredCompilation=False, readyToRun=False, selfContained=False))
    save(output / 'manifest.json', manifest)

    def run(name, command):
        print('Starting ' + name, flush=True)
        command = [str(part) for part in command]
        commands[name] = command
        rows, observed = [], []
        start = time.perf_counter()
        with (output / (name + '.jsonl')).open('wb') as capture, (output / (name + '.stderr')).open('wb') as error:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=error)
            try:
                for line in process.stdout:
                    elapsed = time.perf_counter() - start
                    capture.write(line)
                    row = json.loads(line.decode('utf-8-sig'))
                    rows.append(row)
                    observed.append(dict(kind=row.get('kind'), iteration=row.get('iteration'), seconds=elapsed))
                if process.wait(timeout=1200):
                    raise RuntimeError(name + ' failed; see stderr')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        save(output / (name + '.observed.json'), observed)
        measured = [row['seconds'] for row in rows if row.get('warmup') is False]
        if measured:
            print(name + ' median=' + str(statistics.median(measured)), flush=True)
        return rows, observed

    # Both variants run once before process-start measurements; file cache is not flushed.
    for variant in ('jit', 'aot'):
        rows, _ = run('startup-prime-' + variant, [getattr(options, variant), 'cpu-info'])
        configuration = rows[0]
        assert configuration['engine'] == variant and configuration['runtime'] == '10.0.12'
        assert all(configuration[key] for key in ('avx2', 'fma', 'avxVnni'))
        assert configuration['dynamicCodeSupported'] == (variant == 'jit')
    startup = {'jit': [], 'aot': []}
    for index, variant in enumerate(['jit', 'aot', 'aot', 'jit'] * 5):
        _, observed = run(f'startup-{index:02d}-{variant}', [getattr(options, variant), 'cpu-info'])
        startup[variant].append(observed[0]['seconds'])

    for kind, subjects in [('tts', ['huayan', 'aishell3']), ('asr', ['sip-short-pcma', 'sip-zh-pcmu', 'sip-repeated-pcma'])]:
        for subject in subjects:
            by_variant = {variant: [] for variant in dict.fromkeys(order)}
            wave_paths = {}
            repeats = 3 if subject == 'sip-repeated-pcma' else 5
            for block, variant in enumerate(order):
                name = f'{kind}-{subject}-{block}-{variant}'
                wave = output / (name + '.f32')
                if variant == 'native':
                    command = ([options.cpp, model_paths['encoder'], model_paths['decoder'], options.inputs / (subject + '.f32'), repeats]
                               if kind == 'asr' else ['dotnet', options.ort, model_paths[subject], options.inputs / (subject + '.i64'), subject, repeats, wave, '--provider', 'cpu', '--threads', 4, '--affinity', '0x55'])
                else:
                    executable = ['dotnet', options.current] if variant == 'current' else [getattr(options, variant)]
                    command = executable + ([kind, options.models, options.inputs / (subject + '.f32'), repeats]
                                            if kind == 'asr' else [kind, options.models, options.inputs / (subject + '.i64'), subject, repeats, wave])
                    command += ['--threads', 4, '--affinity', '0x55']
                rows, observed = run(name, command)
                samples = [row for row in rows if row.get('kind') == kind]
                assert len(samples) == repeats + 1 and sum(row['warmup'] for row in samples) == 1
                if kind == 'asr':
                    assert next(row for row in rows if row['kind'] == 'load')['promptCache'] is False
                else:
                    wave_paths[name] = wave
                    hashes = {row['waveSha256'] for row in samples if 'waveSha256' in row}
                    assert len(hashes) <= 1
                by_variant[variant].append((rows, observed))
                all_runs.append(dict(name=name, kind=kind, subject=subject, variant=variant,
                                     formalCount=repeats, warmupCount=1))
            for variant, blocks in by_variant.items():
                samples = [row for rows, _ in blocks for row in rows if row.get('warmup') is False]
                first = [row['seconds'] for rows, _ in blocks for row in rows if row.get('warmup') is True]
                load = [row['seconds'] for rows, _ in blocks for row in rows if row['kind'] == 'load']
                ready = [next(row['seconds'] for row in observed if row['kind'] == 'load') for _, observed in blocks]
                first_result = [next(row['seconds'] for row in observed if row['kind'] == kind) for _, observed in blocks]
                group = dict(kind=kind, subject=subject, variant=variant, seconds=stats([row['seconds'] for row in samples]),
                             firstCallSeconds=stats(first), loadSeconds=stats(load),
                             processToLoadRecordSeconds=stats(ready), processToFirstResultSeconds=stats(first_result),
                             peakRssBytes=max(row['peakRssBytes'] for rows, _ in blocks for row in rows if row.get('kind') == kind))
                if kind == 'asr':
                    group['texts'] = sorted({row['Text'] for rows, _ in blocks for row in rows if row.get('kind') == kind})
                    group['encoderMedianSeconds'] = statistics.median(row.get('EncoderSeconds', row.get('encoderSeconds')) for row in samples)
                    group['decoderMedianSeconds'] = statistics.median(row.get('DecoderSeconds', row.get('decoderSeconds')) for row in samples)
                else:
                    group['samples'] = sorted({row['samples'] for row in samples})
                groups.append(group)
            if kind == 'tts':
                reference = next(path for name, path in wave_paths.items() if name.endswith('-native'))
                jit_reference = next(path for name, path in wave_paths.items() if name.endswith('-jit'))
                for name, path in wave_paths.items():
                    for tag, against in [('native', reference), ('jit', jit_reference)]:
                        parity = wave_parity(against, path)
                        assert parity['passed'], (name, tag, parity)
                        parities[name + '-vs-' + tag] = parity
            else:
                texts = [group['texts'] for group in groups if group['subject'] == subject]
                assert all(value == texts[0] and len(value) == 1 for value in texts)
    summary = dict(platform='Windows x64 / Intel Core i7-12700', runtime='10.0.12', threads=4, affinity='0x55',
                   order=order, percentileMethod='nearest rank', promptCache=False, maxTokens=128, gpu=False,
                   formalCount=sum(run['formalCount'] for run in all_runs), warmupCount=len(all_runs),
                   startupToConfigurationSeconds={variant: stats(values) for variant, values in startup.items()},
                   startupFileCache='not flushed; one priming process per variant excluded',
                   groups=groups, waveParity=parities, asrTextParity=True, runs=all_runs)
    save(output / 'summary.json', summary)
    save(output / 'commands.json', commands)
    for subject in ['sip-short-pcma', 'sip-zh-pcmu', 'sip-repeated-pcma', 'huayan', 'aishell3']:
        print(subject, {group['variant']: round(group['seconds']['median'], 6) for group in groups if group['subject'] == subject}, flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
