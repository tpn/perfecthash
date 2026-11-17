import ctypes, json, os, re, subprocess
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description='Repeat the existing Windows ARM64 benchmarks on CPU affinity mask 4.')
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
root = parser.parse_args().root.resolve()
out = root / 'build' / 'benchmarks-grace'
out.mkdir(exist_ok=True)
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
if not k.SetProcessAffinityMask(k.GetCurrentProcess(),4): raise ctypes.WinError(ctypes.get_last_error())
os.environ['PERFECT_HASH_ONLINE_MAX_CONCURRENCY']='1'
exe=root/'build/bin/Release/perfecthash_benchmarks.exe'
hashes=['MultiplyShiftR','MultiplyShiftRX','Mulshrolate1RX','Mulshrolate2RX','Mulshrolate3RX','Mulshrolate4RX']
cases=[('hologram-msr32',['--keys-file='+str(root/'keys/HologramWorld-31016.keys'),'--preset=hologramworld-msr']),('hologram-msr16',['--keys-file='+str(root/'keys/HologramWorld-31016.keys'),'--preset=hologramworld-msr','--allow-assigned16'])]
cases += [('random100k-'+h,['--keys=100000','--hash='+h]) for h in hashes]
cases += [('random1m-'+h,['--keys=1000000','--hash='+h]) for h in ['MultiplyShiftR','Mulshrolate4RX']]
results=[]
for name,args in cases:
    records=[]
    for repeat in range(3):
        cmd=[str(exe),*args,'--no-jit','--iterations=5','--loops='+('5' if name.startswith('random1m-') else '50'),'--max-solve-time=30']
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=180,cwd=root)
        (out/f'{name}-{repeat+1}.txt').write_text(p.stdout+p.stderr,encoding='utf-8',newline='\n')
        record={'repeat':repeat+1,'exit_code':p.returncode}
        if p.returncode != 0:
            if name == 'random1m-MultiplyShiftR' and p.returncode == 2:
                record['outcome'] = 'solve timeout; no timing'
            else:
                raise RuntimeError(f'{name} failed: {p.returncode}; see raw output')
        for metric,pattern in {'build_ms':r'Build time \(PerfectHash\):\s+([\d.]+) ms','ph_ns':r'Lookup avg \(PerfectHash\):[^\n]*\(([\d.]+) ns/op\)','unordered_ns':r'Lookup avg \(unordered_map\):[^\n]*\(([\d.]+) ns/op\)','map_ns':r'Lookup avg \(map\):[^\n]*\(([\d.]+) ns/op\)'}.items():
            m=re.search(pattern,p.stdout)
            if m: record[metric]=float(m[1])
        records.append(record)
        print(name,record,flush=True)
    results.append({'name':name,'command':cmd,'runs':records})
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8',newline='\n')
# README benchmark executables return their cycle measurement as the exit code.
generated=[]
for h in hashes:
    dirs={'MultiplyShiftR':'codegen','MultiplyShiftRX':'codegen-multiplyshiftrx'}
    folder=dirs.get(h,'codegen-'+h.lower())
    directory=root/'build/tests/cli-out'/folder/'_build/Release'
    index=next(directory.glob('BenchmarkIndex_*.exe'))
    full=next(directory.glob('BenchmarkFull_*.exe'))
    test=next(directory.glob('Test_*.exe'))
    subprocess.run([str(test)],check=True,capture_output=True,timeout=30)
    row={'hash':h,'index_command':str(index),'full_command':str(full)}
    for kind,binary in [('index',index),('full',full)]:
        row[kind+'_cycles']=[]
        for repeat in range(7):
            cycles=subprocess.run([str(binary)],capture_output=True,timeout=60).returncode & 0xffffffff
            if cycles == 0xffffffff:
                raise RuntimeError(f'Cycle timing unavailable: {binary}, run {repeat+1}')
            if cycles == 0 or cycles >= 0x80000000:
                raise RuntimeError(f'Invalid cycle status {cycles}: {binary}, run {repeat+1}')
            row[kind+'_cycles'].append(cycles)
    if any(value >= 0x80000000 for kind in ('index', 'full') for value in row[kind+'_cycles']):
        raise RuntimeError(f'Unexpected benchmark exit status: {row}')
    generated.append(row)
    print('generated',row,flush=True)
(out/'generated.json').write_text(json.dumps(generated,indent=2),encoding='utf-8',newline='\n')
