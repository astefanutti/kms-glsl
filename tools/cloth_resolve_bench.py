"""Resolver bench: load a wad fixture, release the grab (restore inv-mass), run
post-release frames with the resolver, print the drain curve + wall time.
Usage: cloth_resolve_bench.py wad_1500.npz [frames=90]
(fixtures: .npz with pos/prev/vel/inv/pinned/sphere/xings, e.g. from a
CLOTH_RECORD replay snapshot; see the pdt-collision branch notes)"""
import sys, os, time
# Resolves the repo's examples/ relative to this file (tools/); CLOTH_DIR overrides
# for side-by-side A/B builds. Prints the module actually imported -- the
# import-shadowing trap has bitten repeatedly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EX = os.environ.get("CLOTH_DIR", os.path.join(_ROOT, "examples"))
sys.path.insert(0, _ROOT); sys.path.insert(0, _EX)
import numpy as np, warp as wp
import cloth_drag_fuzz as F
import cloth as C
print("cloth module:", C.__file__, flush=True)

fx = sys.argv[1]; nfr = int(sys.argv[2]) if len(sys.argv) > 2 else 90
w = np.load(fx)
sc = w["sphere"]
cl = F.build(400, sphere_center=(float(sc[0]), float(sc[1]), float(sc[2])), warmup=1)
C.sphere.radius = float(sc[3])
inv = w["inv"].copy(); inv[w["pinned"]] = inv[inv > 0].max()   # release: unpin members
wp.copy(cl.pos, wp.array(w["pos"], dtype=wp.vec3))
wp.copy(cl.prevPos, wp.array(w["prev"], dtype=wp.vec3))
wp.copy(cl.vel, wp.array(w["vel"], dtype=wp.vec3))
wp.copy(cl.invMass, wp.array(inv, dtype=float))
cl.hostInvMass.numpy()[:] = inv
cl.hostPos.numpy()[:] = w["pos"]
n0, _ = F.global_xings(cl)
print(f"fixture {os.path.basename(fx)}: start XINGS={n0} (saved {int(w['xings'])})", flush=True)
t_total = 0.0; curve = []
for f in range(nfr):
    t0 = time.perf_counter()
    cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
    wp.synchronize()
    dt = time.perf_counter() - t0; t_total += dt
    if f % 5 == 4 or f < 5:
        n, _ = F.global_xings(cl)
        curve.append((f, n, dt * 1e3))
        print(f"f{f:3d} XINGS={n:5d} frame_ms={dt*1e3:7.1f}", flush=True)
        if n == 0:
            break
n, _ = F.global_xings(cl)
print(f"END XINGS={n} frames={f+1} total_s={t_total:.1f} max_frame_ms={max(c[2] for c in curve):.0f}", flush=True)
