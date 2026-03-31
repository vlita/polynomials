import pickle

base = 'unfiltered_method_dfs'
methods = [
    'B2PLYP-D3', 'B3LYP-D3', 'CCSD', 'CCSD(T)',
    'FNO-CCSD', 'FNO-CCSD(T)', 'HF', 'M05-2X',
    'MP2', 'PBE-D3', 'wB97X-D', 'wB97X-V',
]

print('%-20s %10s %10s %10s %14s %8s' % (
    'Method', 'DiskDFJK', 'In-core', 'Turn-off', 'Unclassified', 'Missing'))
print('-' * 74)

totals = [0, 0, 0, 0, 0]

for m in methods:
    with open('%s/%s.pkl' % (base, m), 'rb') as f:
        df = pickle.load(f)
    d, ic, to, nh, miss = 0, 0, 0, 0, 0
    for i in range(len(df)):
        output = df.iloc[i]['psi4 output']
        if not isinstance(output, str):
            miss += 1
            continue
        idx = output.find('==> Integral Setup <==')
        if idx == -1:
            nh += 1
            continue
        snippet = output[idx:idx + 600]
        if 'DiskDFJK' in snippet:
            d += 1
        elif 'Using in-core AOs' in snippet:
            ic += 1
        elif 'Turning off in-core' in snippet:
            to += 1
        else:
            nh += 1
    print('%-20s %10d %10d %10d %14d %8d' % (m, d, ic, to, nh, miss))
    for j, v in enumerate([d, ic, to, nh, miss]):
        totals[j] += v

print('-' * 74)
print('%-20s %10d %10d %10d %14d %8d' % (('TOTAL',) + tuple(totals)))
