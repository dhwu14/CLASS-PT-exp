#!/usr/bin/env python3
"""DW: validate H/Hconf normalization through the linear Kaiser multipoles.

This regression test is intended for qcdm_rsd_H_Z3_audit.ini.  It compares
P0/P2/P4 from the exact-time RSD output against the analytic Kaiser result
using CLASS's own background growth rate f(z).  The C runtime additionally
checks Hconf=aH, Z2 symmetry/transverse limits, and collapsed Z3 symmetry and
transverse limits before any loop integration.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def load(path: Path) -> np.ndarray:
    arr=np.loadtxt(path)
    if arr.ndim==1:
        arr=arr[None,:]
    if not np.isfinite(arr).all():
        raise RuntimeError(f"non-finite data in {path}")
    return arr


def interp(x: np.ndarray,y: np.ndarray,xp: np.ndarray) -> np.ndarray:
    return np.interp(xp,x,y)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='output/qcdm_rsd_H_Z3_audit')
    ap.add_argument('--z',type=float,default=0.5)
    ap.add_argument('--kmin',type=float,default=1.0e-4)
    ap.add_argument('--kmax',type=float,default=2.0e-2)
    ap.add_argument('--rtol',type=float,default=3.0e-3)
    args=ap.parse_args()

    root=Path(args.root)
    mult=load(Path(str(root)+'_qcdm_pt_multipoles.dat'))
    pk=load(Path(str(root)+'_pk.dat'))
    bg=load(Path(str(root)+'_background.dat'))

    # CLASS background output columns: z=0, H=3, D=31, f=32 (zero based).
    order=np.argsort(bg[:,0])
    fclass=float(np.interp(args.z,bg[order,0],bg[order,32]))

    mask=(mult[:,0]>=args.kmin)&(mult[:,0]<=args.kmax)
    if not np.any(mask):
        raise RuntimeError('no multipole rows in requested k interval')
    k=mult[mask,0]
    # Columns 4,5,6 (zero based) are P0_tree,P2_tree,P4_tree.
    p0,p2,p4=mult[mask,4],mult[mask,5],mult[mask,6]
    pl=interp(pk[:,0],pk[:,1],k)
    b=1.0
    expected=np.vstack([
        (b*b+2.0*b*fclass/3.0+fclass*fclass/5.0)*pl,
        (4.0*b*fclass/3.0+4.0*fclass*fclass/7.0)*pl,
        (8.0*fclass*fclass/35.0)*pl,
    ])
    measured=np.vstack([p0,p2,p4])
    rel=np.abs(measured-expected)/np.maximum(np.abs(expected),1.0e-30)
    max_by_ell=rel.max(axis=1)
    f_from_p4=np.sqrt(np.maximum(35.0*p4/(8.0*pl),0.0))
    f_rel=np.max(np.abs(f_from_p4-fclass)/max(abs(fclass),1.0e-30))

    print(f'DW RSD audit z={args.z:g}: CLASS f={fclass:.12g}')
    print('max relative Kaiser residuals: '
          f'P0={max_by_ell[0]:.6e}, P2={max_by_ell[1]:.6e}, '
          f'P4={max_by_ell[2]:.6e}')
    print(f'max relative f(P4) residual={f_rel:.6e}')
    passed=bool(np.all(max_by_ell<=args.rtol) and f_rel<=args.rtol)
    print('PASS' if passed else 'FAIL')
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
