"""Replay a recorded session and trace per-frame drag AUTHORITY: pointer-target
vs committed anchor lag, the brake/yield/advance scales, and exact crossings."""
import sys, os, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "examples"))
sys.path.insert(0, _ROOT)
import numpy as np, warp as wp
import cloth_drag_fuzz as F
import cloth as C
from cloth import AnchorFlag

path = sys.argv[1]; nx = int(sys.argv[2]) if len(sys.argv) > 2 else 400
frames = [json.loads(l) for l in open(path) if l.strip()]
sph0 = frames[0]["sphere"]
cl = F.build(nx, sphere_center=(sph0[0], sph0[1], sph0[2]))
C.sphere.radius = sph0[3]
live = {}
prev_anchor_pos = None
cum_cmd = 0.0; cum_act = 0.0
for fi, fr in enumerate(frames):
    sp = fr["sphere"]
    C.sphere.center = wp.vec3(sp[0], sp[1], sp[2]); C.sphere.radius = sp[3]
    C.sphere.dc = wp.vec3(sp[4], sp[5], sp[6]); C.sphere.dr = sp[7]
    F.RAYMAP.clear(); seen = set()
    for a in fr["anchors"]:
        key = (round(a["screen"][0], 2), round(a["screen"][1], 2))
        F.RAYMAP[key] = (wp.vec3f(*a["origin"]), wp.vec3f(*a["dir"]))
        seen.add(a["id"])
        if a["id"] not in live:
            p = cl.drag_anchor(a["screen"][0], a["screen"][1])
            if p is None: print(f"f{fi} pick MISS"); continue
            live[a["id"]] = p; prev_anchor_pos = None
        live[a["id"]].screen = wp.vec2(float(a["screen"][0]), float(a["screen"][1]))
    for rid in [r for r in live if r not in seen]:
        live[rid].flags &= ~AnchorFlag.ACTIVE; del live[rid]
    cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
    C.sphere.center = C.sphere.center + C.sphere.dc; C.sphere.radius = C.sphere.radius + C.sphere.dr
    C.sphere.dc = wp.vec3(); C.sphere.dr = 0.0
    if live and fi % 4 == 0:
        p = list(live.values())[0]
        a = fr["anchors"][0]
        o = np.array(a["origin"]); d = np.array(a["dir"])
        ptr = o + d * p.depth                      # raw pointer target at current depth
        anc = np.asarray(p.prev_target)
        lag = float(np.linalg.norm(ptr - anc))
        step = 0.0
        if prev_anchor_pos is not None:
            step = float(np.linalg.norm(anc - prev_anchor_pos))
        prev_anchor_pos = anc.copy()
        n_near = getattr(cl, "crossNearHost", {}).get(p.id, 0)
        hold = getattr(p, "brake_hold", 0.0)
        yld = cl.grabPressureHost.get(p.id)
        mm = 0.0
        if yld: mm = yld[1] / float(yld[2] * C.numSubsteps) / C.d_offset
        nx_, _ = F.global_xings(cl)
        if nx_ > int(os.environ.get("TRACE_ABORT_XINGS", "800")):
            print(f"ABORT f{fi}: XINGS={nx_} runaway wad -- config FAILED", flush=True)
            sys.exit(3)
        max_step = 0.45 * C.d_offset * C.numSubsteps
        print(f"f{fi:3d} lag={lag:.3f} step/frame={step:.4f} (max {max_step:.4f}) "
              f"brake_hold={hold:.1f} n_near={n_near} yield_mag={mm:.2f}d XINGS={nx_}", flush=True)
for _ in range(30):
    cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
n, _ = F.global_xings(cl)
print(f"FINAL XINGS={n}")
