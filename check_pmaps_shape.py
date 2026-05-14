import os
import sys
from pathlib import Path

import numpy as np
import tables as tb

raw = sys.argv[1] if len(sys.argv) > 1 else "$ICDIR/database/test_data/Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_PMP.h5"
path = os.path.expandvars(raw)
print('PATH', path)
print('EXISTS', Path(path).exists())
if not Path(path).exists():
    raise SystemExit

with tb.open_file(path) as h5:
    s2 = h5.root.PMAPS.S2.read()
    s2si = h5.root.PMAPS.S2Si.read()

bad = []
keys = np.unique(np.stack((s2['event'], s2['peak']), axis=1), axis=0)
for evt, pk in keys:
    evt = int(evt)
    pk = int(pk)

    s2_mask = (s2['event'] == evt) & (s2['peak'] == pk)
    si_mask = (s2si['event'] == evt) & (s2si['peak'] == pk)

    lt = int(np.count_nonzero(s2_mask))
    n_si = int(np.count_nonzero(si_mask))
    if n_si == 0:
        continue

    nids = int(np.unique(s2si['nsipm'][si_mask]).size)
    if nids == 0:
        continue

    if n_si % nids != 0:
        bad.append((evt, pk, lt, n_si, nids, n_si / nids))
        continue

    ls = n_si // nids
    if ls != lt:
        bad.append((evt, pk, lt, n_si, nids, ls))

print('S2 rows', len(s2), 'S2Si rows', len(s2si))
print('BAD COUNT', len(bad))
for row in bad[:20]:
    print('BAD', row)
