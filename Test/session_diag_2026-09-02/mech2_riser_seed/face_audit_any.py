import numpy as np, collections, sys
NAME = {0:"CAST",1:"RISER",2:"CHILL",4:"CORE",6:"SPRUE",7:"RUNNER",8:"FILTER",9:"GATE",-1:"MOLD/empty"}
for p in sys.argv[1:]:
    t = np.load(p); nx,ny,nz = t.shape
    print(f"\n##### {p}  shape={t.shape}  metal={int((t>=0).sum())}")
    faces = [("x=0",t[0]),(f"x={nx-1}",t[nx-1]),("y=0",t[:,0]),(f"y={ny-1}",t[:,ny-1]),
             ("z=0",t[:,:,0]),(f"z={nz-1}",t[:,:,nz-1])]
    for nmf, sl in faces:
        m = sl>=0
        c = collections.Counter(NAME.get(int(v),str(v)) for v in sl[m].ravel())
        print(f"  {nmf:8s}: {int(m.sum()):6d} part cells {dict(c) if m.any() else ''}")
    for ax,(lo,hi,ln,hn) in {"x":(t[0],t[nx-1],"x=0",f"x={nx-1}"),
                             "y":(t[:,0],t[:,ny-1],"y=0",f"y={ny-1}"),
                             "z":(t[:,:,0],t[:,:,nz-1],"z=0",f"z={nz-1}")}.items():
        both=(lo>=0)&(hi>=0)
        if both.sum():
            c=collections.Counter((NAME.get(int(lo[i,j])),NAME.get(int(hi[i,j]))) for i,j in zip(*np.nonzero(both)))
            print(f"  WRAP-{ax}: {int(both.sum())} metal<->metal pairs {dict(c)}")
        else:
            print(f"  WRAP-{ax}: 0 metal<->metal pairs (safe: opposite face is mold/empty)")
