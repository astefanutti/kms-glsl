#!/usr/bin/env python
"""Human-drag fuzzer for the cloth simulation, headless.

Emulates the APP's full grab control path -- screen-space rays through the real
drag_anchor / update_anchors code (picking, depth rules, conforming grip,
per-substep anchor sweep) -- with seeded, human-like drag strokes, and detects
penetration failures automatically. This exists because collision-level repros
that drive anchor positions directly kept testing clean while users still saw
penetration: the bugs lived in the control layer this file exercises.

Usage
  # fuzz N sessions (each = several strokes on a fresh drape), report failures
  .venv/bin/python examples/cloth_drag_fuzz.py --sessions 12 --nx 200
  # replay a recorded failure deterministically
  .venv/bin/python examples/cloth_drag_fuzz.py --replay /tmp/fuzz_fail_1234.json
  # ROD rig (multi-fold contact drags; COLLIDER_KIND=1 z-cylinder at (0,1.5)
  # R=0.35 -- main() re-execs with the env var set, since it is an import-time
  # wp.constant; explicit COLLIDER_KIND=1 up front works too):
  .venv/bin/python examples/cloth_drag_fuzz.py --scenario rod_cross --seed 500
  .venv/bin/python examples/cloth_drag_fuzz.py --scenario rod_slide --seed 500
  .venv/bin/python examples/cloth_drag_fuzz.py --scenario rod_pile_push --seed 500
  # no-collider control (LOCKED-anchor clothesline, any collider kind):
  .venv/bin/python examples/cloth_drag_fuzz.py --scenario clothesline --seed 500

A session FAILS if, after a stroke is released and the cloth settles,
self-collision separation violations persist (entanglement), or fabric remains
inside the sphere. Transient squeeze during an active pinch is reported but is
not a failure (real pinches press fabric together).
"""

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)  # _HERE must end up FIRST: a cloth.py in the parent
                           # dir (side-by-side A/B layouts) would otherwise
                           # shadow the one next to this harness -- the
                           # import-shadowing trap that has bitten 4 times now.

import numpy as np
import warp as wp

import cloth as C

W, H = 1920.0, 1080.0
FOV = 40.0  # gluPerspective fovy used by Camera.init
CAM_POS = np.array([0.0, 1.0, 5.0])
CAM_FWD = np.array([0.0, 0.0, -1.0])
CAM_UP = np.array([0.0, 1.0, 0.0])
CAM_RIGHT = np.cross(CAM_FWD, CAM_UP)
TAN = math.tan(math.radians(FOV) / 2.0)
ASPECT = W / H


def set_camera(pos, fwd=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0)):
    """Repoint the emulated camera (screen rays + projection + the cloth
    module's camera.pos). The rod's along-axis scenarios need a SIDE view:
    from the default z-camera, dragging along the rod axis is a drag in
    DEPTH, which the app's control scheme has no authority over (grab depth
    is fixed outside the ground/collider rules) -- measured as stall ~0.8
    with zero contact. From the side, along-rod is screen-horizontal."""
    global CAM_POS, CAM_FWD, CAM_UP, CAM_RIGHT
    CAM_POS = np.array(pos, dtype=float)
    CAM_FWD = np.array(fwd, dtype=float)
    CAM_FWD = CAM_FWD / np.linalg.norm(CAM_FWD)
    CAM_UP = np.array(up, dtype=float)
    CAM_RIGHT = np.cross(CAM_FWD, CAM_UP)
    if getattr(C, "camera", None) is not None:
        C.camera = SimpleNamespace(pos=CAM_POS)


RAYMAP = {}   # (round(sx,2), round(sy,2)) -> (origin, dir) from a recorded app frame


def emu_ray_from_screen(screen_x, screen_y):
    hit = RAYMAP.get((round(screen_x, 2), round(screen_y, 2)))
    if hit is not None:
        return hit
    return _emu_ray_synthetic(screen_x, screen_y)


def _emu_ray_synthetic(screen_x, screen_y):
    """GL-free replica of cloth.ray_from_screen for the default camera pose."""
    gl_y = H - screen_y - 1.0
    ndc_x = 2.0 * (screen_x + 0.5) / W - 1.0
    ndc_y = 2.0 * (gl_y + 0.5) / H - 1.0
    d = CAM_FWD + ndc_x * TAN * ASPECT * CAM_RIGHT + ndc_y * TAN * CAM_UP
    d = d / np.linalg.norm(d)
    return (wp.vec3f(CAM_POS[0], CAM_POS[1], CAM_POS[2]),
            wp.vec3f(d[0], d[1], d[2]))


def project(p):
    """World point -> app screen coords (top-down y). None if behind camera."""
    v = np.asarray(p) - CAM_POS
    zc = np.dot(v, CAM_FWD)
    if zc <= 1e-6:
        return None
    sx = (np.dot(v, CAM_RIGHT) / (zc * TAN * ASPECT) + 1.0) / 2.0 * W
    sy_gl = (np.dot(v, CAM_UP) / (zc * TAN) + 1.0) / 2.0 * H
    return sx, H - sy_gl - 1.0


class Metrics:
    def __init__(self, cl):
        self.cl = cl
        self._md = wp.zeros(1, dtype=float)
        self._nv = wp.zeros(1, dtype=wp.int32)
        # hinge table for the flipped-face (visual inversion) metric
        T = cl.hostTriIds
        em = {}
        for f in range(len(T)):
            a, b, c = int(T[f, 0]), int(T[f, 1]), int(T[f, 2])
            for u, v in ((a, b), (b, c), (c, a)):
                k = (u, v) if u < v else (v, u)
                em.setdefault(k, []).append(f)
        self.H = np.array([fs for fs in em.values() if len(fs) == 2], dtype=np.int32)
        self.T = T

    def sample(self):
        cl = self.cl
        self._md.fill_(1.0e30)
        self._nv.zero_()
        mesh = wp.Mesh(cl.pos, cl.triIds.flatten(), bvh_constructor="lbvh")
        wp.launch(C.Cloth.count_self_contacts, dim=cl.numParticles,
                  inputs=[mesh.id, cl.pos, cl.gridRC, 0.5 * C.d_offset,
                          self._md, self._nv])
        wp.synchronize()
        P = cl.pos.numpy()
        a, b, c = P[self.T[:, 0]], P[self.T[:, 1]], P[self.T[:, 2]]
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        flips = int((np.einsum('ij,ij->i', n[self.H[:, 0]], n[self.H[:, 1]]) < -0.7).sum())
        # fabric-vs-collider (edge metric). COLLIDER_KIND 1 (infinite z-cylinder):
        # the same segment-vs-center closest point computed in the xy (radial)
        # metric -- multiply by the mask, which is all-ones (exact no-op) for
        # the sphere so the kind-0 numbers are unchanged.
        sc = np.array([C.sphere.center[0] + C.sphere.dc[0],
                       C.sphere.center[1] + C.sphere.dc[1],
                       C.sphere.center[2] + C.sphere.dc[2]])
        r = C.sphere.radius + C.sphere.dr + C.thickness
        M = np.array([1.0, 1.0, 0.0]) if getattr(C, "colliderKind", 0) == 1 \
            else np.array([1.0, 1.0, 1.0])
        E = cl.edgeIds.numpy()
        pa, pb = P[E[:, 0]], P[E[:, 1]]
        ab = (pb - pa) * M
        t = np.clip(np.einsum('ij,ij->i', (sc - pa) * M, ab)
                    / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12), 0, 1)
        de = np.linalg.norm((pa - sc) * M + t[:, None] * ab, axis=1)
        spen = float(max(0.0, (r - de[de < r]).max()) if (de < r).any() else 0.0)
        return (float(self._md.numpy()[0]), int(self._nv.numpy()[0]), flips, spen)


@wp.kernel
def _seg_xing_k(mesh_id: wp.uint64, edges: wp.array2d(dtype=wp.int32),
                tri_ids: wp.array2d(dtype=wp.int32),
                pos: wp.array(dtype=wp.vec3),
                out_count: wp.array(dtype=wp.int32),
                out_pts: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    va = edges[i, 0]
    vb = edges[i, 1]
    pa = pos[va]
    pb = pos[vb]
    d = pb - pa
    l = wp.length(d)
    if l < 1.0e-9:
        return
    dn = d / l
    start = pa
    remaining = l
    for _ in range(4):
        q = wp.mesh_query_ray(mesh_id, start, dn, remaining)
        if not q.result:
            return
        f = q.face
        i0 = tri_ids[f, 0]
        i1 = tri_ids[f, 1]
        i2 = tri_ids[f, 2]
        if (i0 != va and i0 != vb and i1 != va and i1 != vb
                and i2 != va and i2 != vb):
            k = wp.atomic_add(out_count, 0, 1)
            if k < 4096:
                out_pts[k] = start + dn * q.t
        adv = q.t + 1.0e-5
        start = start + dn * adv
        remaining = remaining - adv
        if remaining <= 0.0:
            return


def global_xings(cl, P=None):
    """EXACT global self-intersection count: every cloth edge vs every
    non-adjacent face. The ground truth for 'did fabric pass through fabric'
    -- position/gap heuristics cannot distinguish crossing from wrap."""
    if P is None:
        P = cl.pos.numpy()
    mesh = wp.Mesh(wp.array(P, dtype=wp.vec3), cl.triIds.flatten(),
                   bvh_constructor="lbvh")
    cnt = wp.zeros(1, dtype=wp.int32)
    pts = wp.zeros(4096, dtype=wp.vec3)
    wp.launch(_seg_xing_k, dim=cl.edgeIds.shape[0],
              inputs=[mesh.id, cl.edgeIds, cl.triIds,
                      wp.array(P, dtype=wp.vec3), cnt, pts])
    wp.synchronize()
    n = int(cnt.numpy()[0])
    return n, pts.numpy()[:min(n, 4096)]


def replay_app_session(path, nx):
    """Replay a CLOTH_RECORD app session (JSONL from the live app) through
    the identical control code, probing exact crossings as it goes."""
    from cloth import AnchorFlag
    frames = [json.loads(l) for l in open(path) if l.strip()]
    print(f"[app-replay] {len(frames)} frames, "
          f"{sum(1 for f in frames if f['anchors'])} with active anchors", flush=True)
    sph0 = frames[0]["sphere"]
    cl = build(nx, sphere_center=(sph0[0], sph0[1], sph0[2]))
    C.sphere.radius = sph0[3]
    met = Metrics(cl)
    live = {}     # recorded anchor id -> Particle
    worst = [0, 0]
    for fi, fr in enumerate(frames):
        sp = fr["sphere"]
        C.sphere.center = wp.vec3(sp[0], sp[1], sp[2])
        C.sphere.radius = sp[3]
        C.sphere.dc = wp.vec3(sp[4], sp[5], sp[6])
        C.sphere.dr = sp[7]
        RAYMAP.clear()
        seen = set()
        for a in fr["anchors"]:
            key = (round(a["screen"][0], 2), round(a["screen"][1], 2))
            RAYMAP[key] = (wp.vec3f(*a["origin"]), wp.vec3f(*a["dir"]))
            seen.add(a["id"])
            if a["id"] not in live:
                p = cl.drag_anchor(a["screen"][0], a["screen"][1])
                if p is None:
                    print(f"[app-replay] f{fi}: pick MISS at {a['screen']}", flush=True)
                    continue
                live[a["id"]] = p
            live[a["id"]].screen = wp.vec2(float(a["screen"][0]), float(a["screen"][1]))
        for rid in [r for r in live if r not in seen]:
            live[rid].flags &= ~AnchorFlag.ACTIVE
            del live[rid]
        cl.update_anchors()
        cl.simulate(steps=C.numSubsteps)
        # commit sphere pose like the app's post_render
        C.sphere.center = C.sphere.center + C.sphere.dc
        C.sphere.radius = C.sphere.radius + C.sphere.dr
        C.sphere.dc = wp.vec3(); C.sphere.dr = 0.0
        if fi % 5 == 4:
            n, pts = global_xings(cl)
            g, v, fl, spn = met.sample()
            worst[0] = max(worst[0], n); worst[1] = max(worst[1], v)
            # Runaway guard: past this wad size the post-release host-side
            # resolver (cluster vote/veto over the crossing set) can spin for
            # tens of minutes at 400^2 while holding the shared GPU. Such a
            # rep has already failed acceptance -- abort and report it.
            abort_n = int(os.environ.get("TRACE_ABORT_XINGS", "800"))
            if n > abort_n:
                print(f"[app-replay] ABORT f{fi}: XINGS={n} > {abort_n} "
                      f"(runaway wad; rep counted as FAILED)", flush=True)
                print(f"[app-replay] FINAL XINGS={n} viol={v} flips={fl} | "
                      f"worst during: XINGS={worst[0]} viol={worst[1]} | ABORTED",
                      flush=True)
                return
            anc = [f"{q.prev_target.round(2).tolist()}" for q in live.values()
                   if q.prev_target is not None]
            print(f"[app-replay] f{fi:4d} XINGS={n} viol={v} flips={fl} "
                  f"spen={spn:.4f} anchors={anc}", flush=True)
            if n:
                P = cl.pos.numpy()
                for q in live.values():
                    if q.prev_target is not None:
                        near = int((np.linalg.norm(
                            pts - np.asarray(q.prev_target), axis=1) < 0.2).sum())
                        print(f"[app-replay]   near_anchor={near}/{n}", flush=True)
    for _ in range(30):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
    n, _ = global_xings(cl)
    g, v, fl, spn = met.sample()
    print(f"[app-replay] FINAL XINGS={n} viol={v} flips={fl} | "
          f"worst during: XINGS={worst[0]} viol={worst[1]}", flush=True)


def ease(t):
    return t * t * (3.0 - 2.0 * t)  # smoothstep: human-ish accel/decel


def gen_stroke(rng, press_xy, cloth_bbox):
    """A stroke = press point + per-frame screen path (smooth waypoints + flicks)."""
    (x0, y0), path = press_xy, []
    x, y = x0, y0
    n_way = rng.integers(1, 4)
    for _ in range(n_way):
        # waypoints within the cloth's screen bbox inflated 25%
        bx0, by0, bx1, by1 = cloth_bbox
        mx, my = (bx1 - bx0) * 0.25, (by1 - by0) * 0.25
        wx = rng.uniform(bx0 - mx, bx1 + mx)
        wy = rng.uniform(by0 - my, by1 + my)
        frames = int(rng.integers(8, 40))
        flick = rng.random() < 0.3
        for i in range(frames):
            t = ease((i + 1) / frames)
            px = x + (wx - x) * t
            py = y + (wy - y) * t
            if flick and i < 3:  # a fast jerk at stroke start
                px += rng.uniform(-40, 40)
                py += rng.uniform(-40, 40)
            path.append((float(px), float(py)))
        x, y = wx, wy
    return {"press": [float(x0), float(y0)], "path": path}


def gen_sphere_stroke(rng):
    """A sphere gesture: eased world-space waypoints for the center (the 4-finger
    translate analog; Sphere.translate applies the app's own per-frame clamp),
    with occasional resize segments (5-finger analog, radius kept in the app's
    0.25..1.25 gesture range)."""
    frames = []
    n_way = int(rng.integers(1, 4))
    for _ in range(n_way):
        target = np.array([rng.uniform(-1.2, 1.2),
                           rng.uniform(0.55, 2.8),
                           rng.uniform(-0.8, 0.8)])
        nfr = int(rng.integers(10, 45))
        resize = rng.random() < 0.25
        dr_total = rng.uniform(-0.5, 0.5) if resize else 0.0
        frames.append({"target": target.tolist(), "frames": nfr, "dr": dr_total})
    return {"kind": "sphere", "segments": frames}


def run_sphere_stroke(cl, stroke, on_frame):
    """Drive the sphere along a stroke through the app's own translate/commit path."""
    for seg in stroke["segments"]:
        target = np.array(seg["target"])
        nfr = seg["frames"]
        for i in range(nfr):
            cur = np.array([C.sphere.center[0], C.sphere.center[1], C.sphere.center[2]])
            t = ease((i + 1) / nfr)
            want = cur + (target - cur) * min(1.0, t * 1.5)
            step = want - cur
            C.sphere.translate(wp.vec3f(step[0], step[1], step[2]))  # app clamp applies
            dr = seg["dr"] / nfr
            if dr:
                new_r = min(1.25, max(0.25, C.sphere.radius + C.sphere.dr + dr))
                C.sphere.resize(new_r - C.sphere.radius - C.sphere.dr)
            on_frame()
            # commit like Sphere.post_render
            C.sphere.center = C.sphere.center + C.sphere.dc
            C.sphere.radius = C.sphere.radius + C.sphere.dr
            C.sphere.dc = wp.vec3()
            C.sphere.dr = 0.0


def run_session(cl, met, session, verbose=False):
    """Run one recorded/generated session through the REAL control code."""
    from cloth import AnchorFlag
    worst_transient = (1e30, 0, 0, 0.0)
    streaks = {"viol": 0, "spen": 0, "flip": 0}
    max_streaks = {"viol": 0, "spen": 0, "flip": 0}

    trace = session.get("_trace")

    def classify_sites(anchors_pos):
        gaps = wp.zeros(cl.numParticles, dtype=float)
        mesh = wp.Mesh(cl.pos, cl.triIds.flatten(), bvh_constructor="lbvh")
        wp.launch(C.Cloth.self_contact_gaps, dim=cl.numParticles,
                  inputs=[mesh.id, cl.pos, cl.gridRC], outputs=[gaps])
        wp.synchronize()
        g = gaps.numpy(); P = cl.pos.numpy()
        vids = np.nonzero(g < 0.5 * C.d_offset)[0]
        if not len(vids):
            return {}
        vp = P[vids]
        sc = np.array([C.sphere.center[0] + C.sphere.dc[0],
                       C.sphere.center[1] + C.sphere.dc[1],
                       C.sphere.center[2] + C.sphere.dc[2]])
        r = C.sphere.radius + C.sphere.dr
        out = dict(n=len(vids),
                   floor=float((vp[:, 1] < 0.06).mean()),
                   shell=float((np.abs(np.linalg.norm(vp - sc, axis=1) - r) < 0.06).mean()))
        if anchors_pos is not None and len(anchors_pos):
            d = np.linalg.norm(vp[:, None, :] - anchors_pos[None, :, :], axis=2).min(axis=1)
            out["near_grab"] = float((d < 0.15).mean())
        out["air"] = max(0.0, 1.0 - out["floor"] - out["shell"])
        # centroid of the worst cluster for spatial pinpointing
        wid = vids[np.argmin(g[vids])]
        out["worst_at"] = [round(float(x), 3) for x in P[wid]]
        return out

    def observe(si, fi):
        nonlocal worst_transient
        s = met.sample()
        worst_transient = min(worst_transient[0], s[0]), max(worst_transient[1], s[1]), \
            max(worst_transient[2], s[2]), max(worst_transient[3], s[3])
        # noticeability streaks (samples are every 3 frames = 0.1 s): sustained
        # deep trouble reads as wrong penetration; single-sample blips are the
        # accepted transient pinch response
        for key, bad in (("viol", s[1] > 10), ("spen", s[3] > 0.002), ("flip", s[2] > 15)):
            streaks[key] = streaks[key] + 1 if bad else 0
            max_streaks[key] = max(max_streaks[key], streaks[key])
        if trace is not None:
            rec = dict(stroke=si, frame=fi, gap=s[0], viol=s[1], flips=s[2], spen=s[3])
            if s[1] > 10 or s[3] > 0.002:
                ap = np.array([[t[0], t[1], t[2]] for t in
                               (a.prev_target for a in cl.anchors
                                if a.flags & AnchorFlag.ACTIVE) if t is not None])
                rec["sites"] = classify_sites(ap)
            trace.append(rec)
        if verbose:
            print(f"  s{si} f{fi:3d}: gap={s[0]:.5f} viol={s[1]} flips={s[2]} sphere={s[3]:.4f}",
                  flush=True)

    for si, stroke in enumerate(session["strokes"]):
        if stroke.get("kind") == "sphere":
            fc = [0]
            def on_frame():
                cl.update_anchors()
                cl.simulate(steps=C.numSubsteps)
                if fc[0] % 3 == 0:
                    observe(si, fc[0])
                fc[0] += 1
            run_sphere_stroke(cl, stroke, on_frame)
            continue
        sx, sy = stroke["press"]
        p = cl.drag_anchor(sx, sy)
        if p is None:
            continue
        sphere_seg = stroke.get("sphere_drift")  # mixed family: sphere moves while dragging
        for fi, (mx, my) in enumerate(stroke["path"]):
            if sphere_seg:
                C.sphere.translate(wp.vec3f(*sphere_seg))
            p.screen = wp.vec2(float(mx), float(my))
            cl.update_anchors()
            cl.simulate(steps=C.numSubsteps)
            if sphere_seg:
                C.sphere.center = C.sphere.center + C.sphere.dc
                C.sphere.dc = wp.vec3()
            if fi % 3 == 0:
                observe(si, fi)
        p.flags &= ~AnchorFlag.ACTIVE
        cl.update_anchors()
        cl.simulate(steps=C.numSubsteps)
    # settle and take the persistence verdict
    tail = []
    for f in range(15):
        cl.update_anchors()
        cl.simulate(steps=C.numSubsteps)
        if f >= 9:
            tail.append(met.sample())
    # classify persistent-violation sites: floor pile / sphere shell / free air
    gaps = wp.zeros(cl.numParticles, dtype=float)
    mesh = wp.Mesh(cl.pos, cl.triIds.flatten(), bvh_constructor="lbvh")
    wp.launch(C.Cloth.self_contact_gaps, dim=cl.numParticles,
              inputs=[mesh.id, cl.pos, cl.gridRC], outputs=[gaps])
    wp.synchronize()
    g = gaps.numpy(); P = cl.pos.numpy()
    vids = np.nonzero(g < 0.5 * C.d_offset)[0]
    site = ""
    if len(vids):
        vp = P[vids]
        sc = np.array([C.sphere.center[0], C.sphere.center[1], C.sphere.center[2]])
        floor = (vp[:, 1] < 0.06).mean()
        shell = (np.abs(np.linalg.norm(vp - sc, axis=1) - C.sphere.radius) < 0.06).mean()
        site = f" sites: floor={floor:.0%} shell={shell:.0%} air={max(0.0, 1 - floor - shell):.0%}"
    gap = min(t[0] for t in tail)
    viol = min(t[1] for t in tail)     # min over tail: persistent if never recovers
    flips = min(t[2] for t in tail)
    spen = max(t[3] for t in tail)
    failed = viol > 5 or gap < 0.3 * C.d_offset or spen > 0.005 or flips > 10
    noticeable = max_streaks["viol"] >= 3 or max_streaks["spen"] >= 3 or max_streaks["flip"] >= 4
    return failed, dict(site=site, noticeable=noticeable, streaks=dict(max_streaks),
                        post_gap=gap, post_viol=viol, post_flips=flips, post_sphere=spen,
                        worst_transient=dict(gap=worst_transient[0], viol=worst_transient[1],
                                             flips=worst_transient[2], sphere=worst_transient[3]))


def build(nx, sphere_center=(0.0, 1.5, 0.0), warmup=45, radius=0.5, y_offset=2.2):
    set_camera((0.0, 1.0, 5.0))               # scenarios run back-to-back: reset
    C.sphere = C.Sphere(center=wp.vec3(*sphere_center), radius=radius)
    C.camera = SimpleNamespace(pos=CAM_POS)   # update_anchors reads camera.pos[1]
    C.ray_from_screen = emu_ray_from_screen   # replace the GL-based ray builder
    cl = C.Cloth(y_offset=y_offset, num_x=nx, num_y=nx, spacing=0.015)
    cl.init_headless()
    for _ in range(warmup):
        cl.simulate(steps=C.numSubsteps)
    return cl


def cloth_screen_bbox(cl):
    P = cl.pos.numpy()[::37]
    pts = [project(p) for p in P]
    pts = [p for p in pts if p is not None]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Scenario harness: reproducible high-level human use cases (vs random fuzz).
# Each scenario returns a metrics dict; --scenario NAME --seed N runs one.
# ---------------------------------------------------------------------------

def _pick_near(cl, world_hint, require_visible=True):
    """Particle id nearest a world-space hint (optionally must project on-screen)."""
    P = cl.pos.numpy()
    order = np.argsort(np.linalg.norm(P - np.asarray(world_hint), axis=1))
    for pid in order[:4000]:
        spt = project(P[pid])
        if spt is None:
            continue
        if not require_visible or (0 < spt[0] < W and 0 < spt[1] < H):
            return int(pid), spt
    return int(order[0]), project(P[order[0]])


def drag_to(cl, met, world_press, world_target, frames, observe=None, hold=0,
            settle=12):
    """Grab nearest-to-hint vertex, drag its screen point toward the projection
    of an eased world waypoint path, release, settle. Returns (effectiveness,
    grabbed particle id): effectiveness = actual world travel of the grabbed
    vertex / commanded travel — the user-visible 'did my drag do anything'."""
    from cloth import AnchorFlag
    pid, spt = _pick_near(cl, world_press)
    p = cl.drag_anchor(float(spt[0]), float(spt[1]))
    if p is None:
        return None, None
    start = cl.pos.numpy()[p.id].copy()
    tgt = np.asarray(world_target, dtype=np.float64)
    last = spt
    k = 0
    for i in range(frames + hold):
        t = ease(min(1.0, (i + 1) / frames))
        w = start * (1 - t) + tgt * t
        sp = project(w)
        if sp is not None:
            last = sp
        p.screen = wp.vec2(float(last[0]), float(last[1]))
        cl.update_anchors()
        cl.simulate(steps=C.numSubsteps)
        if observe and k % 3 == 0:
            observe()
        k += 1
    end = cl.pos.numpy()[p.id].copy()
    eff = float(np.linalg.norm(end - start) / max(np.linalg.norm(tgt - start), 1e-9))
    p.flags &= ~AnchorFlag.ACTIVE
    for _ in range(settle):
        cl.update_anchors()
        cl.simulate(steps=C.numSubsteps)
        if observe:
            observe()
    return eff, p.id


class ScenarioLog:
    """Streaming worst-case + streak tracker over met.sample() tuples."""
    def __init__(self, met):
        self.met = met
        self.worst = dict(gap=1e30, viol=0, flips=0, spen=0.0)
        self.streak = self.max_streak = 0
        self.samples = []

    def __call__(self):
        g, v, f, sp = self.met.sample()
        self.samples.append((g, v, f, sp))
        self.worst["gap"] = min(self.worst["gap"], g)
        self.worst["viol"] = max(self.worst["viol"], v)
        self.worst["flips"] = max(self.worst["flips"], f)
        self.worst["spen"] = max(self.worst["spen"], sp)
        bad = v > 10 or sp > 0.002
        self.streak = self.streak + 1 if bad else 0
        self.max_streak = max(self.max_streak, self.streak)


def scenario_fold_slide(nx, seed):
    """Fold the drape over the sphere 1-3 times, then slide the folded stack
    across the shell. The user-reported penetration case."""
    rng = np.random.default_rng(seed)
    cl = build(nx)
    met = Metrics(cl)
    log = ScenarioLog(met)
    sc = np.array([0.0, 1.5, 0.0])
    r = 0.5
    n_folds = int(rng.integers(1, 4))
    for _ in range(n_folds):
        a = rng.uniform(0, 2 * np.pi)
        press = sc + [(r + 0.05) * np.sin(a), rng.uniform(-0.35, -0.15),
                      (r + 0.05) * np.cos(a)]
        target = sc + [-0.35 * np.sin(a), r + rng.uniform(0.1, 0.25),
                       -0.35 * np.cos(a)]
        drag_to(cl, met, press, target, int(rng.integers(28, 45)), observe=log)
    # slide: grab the top of the folded stack, sweep across the shell
    P = cl.pos.numpy()
    top = P[np.argmax(P[:, 1])]
    a2 = rng.uniform(0, 2 * np.pi)
    exit_dir = np.array([np.sin(a2), 0.0, np.cos(a2)])
    arc = []
    for th in np.linspace(0.15, rng.uniform(1.1, 1.5), 3):
        arc.append(sc + exit_dir * (r + 0.06) * np.sin(th)
                   + np.array([0, (r + 0.06) * np.cos(th), 0]))
    prev = top
    for wtgt in arc:
        drag_to(cl, met, prev, wtgt, int(rng.integers(14, 24)), observe=log,
                settle=2)
        prev = wtgt
    for _ in range(20):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps); log()
    tail = log.samples[-4:]
    return dict(scenario="fold_slide", seed=seed, n_folds=n_folds,
                worst=log.worst, bad_streak=log.max_streak,
                post_viol=min(t[1] for t in tail), post_gap=min(t[0] for t in tail),
                post_spen=max(t[3] for t in tail))


def scenario_collapse(nx, seed):
    """Lift a flap of the drape above the sphere, release, and measure whether
    the folded fabric collapses/slumps naturally or hangs rigidly in the air."""
    rng = np.random.default_rng(seed)
    cl = build(nx)
    met = Metrics(cl)
    log = ScenarioLog(met)
    sc = np.array([0.0, 1.5, 0.0]); r = 0.5
    a = rng.uniform(0, 2 * np.pi)
    press = sc + [(r + 0.05) * np.sin(a), rng.uniform(-0.4, -0.2),
                  (r + 0.05) * np.cos(a)]
    target = sc + [rng.uniform(-0.15, 0.15), r + rng.uniform(0.45, 0.7),
                   rng.uniform(-0.15, 0.15)]
    drag_to(cl, met, press, target, 30, observe=log, hold=10, settle=0)
    # measure the collapse over 90 free frames
    heights, prev_P = [], None
    quiet_at = None
    for f in range(90):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
        if f % 3 == 0:
            log()
            P = cl.pos.numpy()
            heights.append((float(P[:, 1].mean()),
                            float(np.percentile(P[:, 1], 90))))
            if prev_P is not None and quiet_at is None:
                if float(np.abs(P - prev_P).max()) < 2e-3:
                    quiet_at = f
            prev_P = P.copy()
    P = cl.pos.numpy()
    d_axis = np.linalg.norm(P[:, [0, 2]], axis=1)
    aerial = float(((P[:, 1] > sc[1] + 0.12) & (d_axis > r + 6 * C.d_offset)).mean())
    tail = log.samples[-4:]
    return dict(scenario="collapse", seed=seed,
                mean_y=round(heights[-1][0], 4), p90_y=round(heights[-1][1], 4),
                aerial_frac=round(aerial, 4),
                quiet_at=quiet_at if quiet_at is not None else 90,
                worst=log.worst, post_viol=min(t[1] for t in tail))


def scenario_flatten(nx, seed):
    """Crumple the cloth into a floor pile, then flatten it back out with
    corner drags — the user's 'make it flat again' workflow. Key metric:
    drag effectiveness (did the corner actually travel where commanded)."""
    rng = np.random.default_rng(seed)
    cl = build(nx, sphere_center=(0.0, 8.0, 0.0), warmup=70)   # sphere parked away
    met = Metrics(cl)
    log = ScenarioLog(met)
    # crumple: lift the center high, drop; sweep an edge across the pile
    P = cl.pos.numpy()
    center = P[np.argmin(np.linalg.norm(P[:, [0, 2]], axis=1))]
    drag_to(cl, met, center, [rng.uniform(-0.2, 0.2), 1.5, rng.uniform(-0.2, 0.2)],
            22, observe=log, settle=25)
    P = cl.pos.numpy()
    edge = P[np.argmax(P[:, 0])]
    drag_to(cl, met, edge, [rng.uniform(-1.0, -0.6), 0.35, rng.uniform(-0.4, 0.4)],
            22, observe=log, settle=25)
    # flatten: four corner drags toward the flat footprint
    half = 0.5 * (cl.num_x - 1) * cl.spacing if hasattr(cl, "num_x") else 1.45
    effs, picked = [], 0
    P = cl.pos.numpy()
    for sx, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        score = sx * P[:, 0] + sz * P[:, 2]
        press = P[int(np.argmax(score))]
        target = [sx * half * 0.95, 0.03, sz * half * 0.95]
        eff, _ = drag_to(cl, met, press, target, 40, observe=log, settle=12)
        if eff is not None:
            effs.append(eff)
            picked += 1
        P = cl.pos.numpy()
    for _ in range(25):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps); log()
    P = cl.pos.numpy()
    flat_frac = float((P[:, 1] < 5 * C.d_offset).mean())
    span = (P[:, 0].max() - P[:, 0].min()) * (P[:, 2].max() - P[:, 2].min())
    cover = float(span / (2 * half) ** 2)
    tail = log.samples[-4:]
    return dict(scenario="flatten", seed=seed, picked=picked,
                eff=[round(e, 3) for e in effs],
                eff_mean=round(float(np.mean(effs)) if effs else 0.0, 3),
                flat_frac=round(flat_frac, 3), coverage=round(cover, 3),
                worst=log.worst, post_viol=min(t[1] for t in tail))


def _grab_at(cl, world_hint, tries=6):
    """drag_anchor with retries around a world hint (piles often fail picks)."""
    P = cl.pos.numpy()
    order = np.argsort(np.linalg.norm(P - np.asarray(world_hint), axis=1))
    seen = 0
    for pid in order:
        spt = project(P[pid])
        if spt is None or not (0 < spt[0] < W and 0 < spt[1] < H):
            continue
        p = cl.drag_anchor(float(spt[0]), float(spt[1]))
        if p is not None:
            return p, spt
        seen += 1
        if seen >= tries:
            break
    return None, None


def scenario_rigidity(nx, seed):
    """Pile-mobility probe for the user-reported rigidity ('folded/crumpled
    cloth acts like a board'). The contact scaffold only engages under FORCED
    motion (settled piles rest at ~1.36*d_offset, outside the repulsion engage
    zone; drag -> compression -> jamming), so both probes force the pile and
    measure how far the motion TRANSMITS:

    LIFT -- grab the pile top, raise it 0.7 m, hold. A soft pile yields a
    local tent; a scaffolded pile comes up as a slab.
      lift_moved -- fraction of particles displaced > 0.05 during the lift
      lift_high  -- fraction lifted above y = 0.10 at full raise
      tent_r     -- rms xz distance of moved particles from the grab (m):
                    small = local tent, large = slab
    TOW -- after release+settle, grab the pile's +x edge and tow it 0.9 m
    across the floor. A soft pile pays out a tongue of fabric; a rigid one
    slides as a block.
      tow_moved  -- fraction displaced > 0.05 during the tow
      tow_com    -- xz center-of-mass displacement of the whole cloth (m)
    Plus pile_h1/pile_slump (passive settle before the probes) and the usual
    worst/post violation counters as regression guards.
    """
    rng = np.random.default_rng(seed)
    cl = build(nx, sphere_center=(0.0, 8.0, 0.0), warmup=70)
    met = Metrics(cl)
    log = ScenarioLog(met)
    # crumple into a floor pile (same recipe as scenario_flatten)
    P = cl.pos.numpy()
    center = P[np.argmin(np.linalg.norm(P[:, [0, 2]], axis=1))]
    drag_to(cl, met, center, [rng.uniform(-0.2, 0.2), 1.5, rng.uniform(-0.2, 0.2)],
            22, observe=log, settle=25)
    P = cl.pos.numpy()
    edge = P[np.argmax(P[:, 0])]
    drag_to(cl, met, edge, [rng.uniform(-1.0, -0.6), 0.35, rng.uniform(-0.4, 0.4)],
            22, observe=log, settle=25)
    # short passive settle (also gives the slump-under-nothing number)
    h0 = float(np.percentile(cl.pos.numpy()[:, 1], 90))
    for f in range(40):
        cl.update_anchors()
        cl.simulate(steps=C.numSubsteps)
        if f % 5 == 0:
            log()
    P = cl.pos.numpy()
    h1 = float(np.percentile(P[:, 1], 90))

    from cloth import AnchorFlag
    out = dict(scenario="rigidity", seed=seed,
               pile_h0=round(h0, 4), pile_h1=round(h1, 4),
               pile_slump=round(h0 - h1, 4))

    # ---- LIFT: raise the pile top 0.7 m, hold ----
    top = P[np.argmax(P[:, 1])]
    p, spt = _grab_at(cl, top + [0.0, -0.01, 0.0])
    if p is None:
        out.update(lift_moved=None)
    else:
        P0 = cl.pos.numpy().copy()
        start_w = P0[p.id].copy()
        tgt = start_w + [0.0, 0.7, 0.0]
        last = spt
        for i in range(25 + 8):
            t = ease(min(1.0, (i + 1) / 25))
            w = start_w * (1 - t) + tgt * t
            sp = project(w)
            if sp is not None:
                last = sp
            p.screen = wp.vec2(float(last[0]), float(last[1]))
            cl.update_anchors()
            cl.simulate(steps=C.numSubsteps)
            if i % 3 == 0:
                log()
        P = cl.pos.numpy()
        dP = np.linalg.norm(P - P0, axis=1)
        moved = dP > 0.05
        ax, az = P[p.id, 0], P[p.id, 2]
        rxz = np.sqrt((P[moved, 0] - ax) ** 2 + (P[moved, 2] - az) ** 2)
        out.update(lift_moved=round(float(moved.mean()), 4),
                   lift_high=round(float((P[:, 1] > 0.10).mean()), 4),
                   tent_r=round(float(np.sqrt((rxz ** 2).mean())) if moved.any() else 0.0, 4),
                   lift_eff=round(float(np.linalg.norm(P[p.id] - start_w) / 0.7), 3))
        p.flags &= ~AnchorFlag.ACTIVE
        for _ in range(15):
            cl.update_anchors()
            cl.simulate(steps=C.numSubsteps)

    # ---- TOW: drag the +x pile edge 0.9 m across the floor ----
    P = cl.pos.numpy()
    eidx = int(np.argmax(P[:, 0]))
    p, spt = _grab_at(cl, P[eidx])
    if p is None:
        out.update(tow_moved=None)
    else:
        P0 = cl.pos.numpy().copy()
        com0 = P0[:, [0, 2]].mean(axis=0)
        start_w = P0[p.id].copy()
        tgt = start_w + [0.9, 0.0, 0.0]
        tgt[1] = 0.03
        last = spt
        for i in range(30):
            t = ease((i + 1) / 30)
            w = start_w * (1 - t) + tgt * t
            sp = project(w)
            if sp is not None:
                last = sp
            p.screen = wp.vec2(float(last[0]), float(last[1]))
            cl.update_anchors()
            cl.simulate(steps=C.numSubsteps)
            if i % 3 == 0:
                log()
        P = cl.pos.numpy()
        dP = np.linalg.norm(P - P0, axis=1)
        com1 = P[:, [0, 2]].mean(axis=0)
        out.update(tow_moved=round(float((dP > 0.05).mean()), 4),
                   tow_com=round(float(np.linalg.norm(com1 - com0)), 4),
                   tow_eff=round(float(np.linalg.norm(P[p.id] - start_w) / 0.9), 3))
        p.flags &= ~AnchorFlag.ACTIVE

    for _ in range(12):
        cl.update_anchors()
        cl.simulate(steps=C.numSubsteps)
        log()
    tail = log.samples[-4:]
    out.update(worst=log.worst, bad_streak=log.max_streak,
               post_viol=min(t[1] for t in tail),
               post_gap=round(min(t[0] for t in tail), 5),
               post_spen=max(t[3] for t in tail))
    return out


def scenario_cross_drag(nx, seed):
    """Drag the right hanging flank across the sphere's VISIBLE face into the
    left flank (both camera-side, so the depth rules stay smooth and the
    contact actually happens on screen, like the user's gesture). Metrics:
    grip shakiness = total variation of the anchor's velocity relative to its
    speed (smooth steady drag ~0.2-0.5; limit-cycling >> 1), stall fraction vs
    the commanded arc, and push transmission into the contacted flank."""
    from cloth import AnchorFlag
    rng = np.random.default_rng(seed)
    cl = build(nx)
    met = Metrics(cl)
    log = ScenarioLog(met)
    sc = np.array([0.0, 1.5, 0.0]); r = 0.5
    a0 = np.radians(rng.uniform(45, 70))    # right flank, front quadrant
    b0 = -np.radians(rng.uniform(45, 70))   # left flank, front quadrant
    ay = rng.uniform(1.0, 1.3)
    press = np.array([(r + 0.05) * np.sin(a0), ay, (r + 0.05) * np.cos(a0)])
    pid, spt = _pick_near(cl, press)
    p = cl.drag_anchor(float(spt[0]), float(spt[1]))
    if p is None:
        return dict(scenario="cross_drag", seed=seed, error="no pick")
    P0 = cl.pos.numpy()
    tgt_pt = np.array([(r + 0.05) * np.sin(b0), ay, (r + 0.05) * np.cos(b0)])
    band = np.nonzero(np.linalg.norm(P0 - tgt_pt, axis=1) < 0.18)[0]
    band_start = P0[band].mean(axis=0) if len(band) else None
    push_dir = np.array([np.sin(b0), 0.0, np.cos(b0)])  # outward at flank B
    start_pos = P0[p.id].copy()
    n_frames = 80
    traj, depths, frame_ms = [], [], []
    last = spt
    overshoot = np.radians(20)              # push past B so contact is sustained
    for i in range(n_frames):
        th = a0 + (i + 1) / n_frames * (b0 - overshoot - a0)
        w = np.array([(r + 0.08) * np.sin(th), ay, (r + 0.08) * np.cos(th)])
        sp = project(w)
        if sp is not None:
            last = sp
        p.screen = wp.vec2(float(last[0]), float(last[1]))
        t0 = time.perf_counter()
        cl.update_anchors()
        cl.simulate(steps=C.numSubsteps)
        wp.synchronize()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        traj.append(cl.pos.numpy()[p.id].copy())
        depths.append(float(p.depth))
        if i % 3 == 0:
            log()
    T = np.array(traj)
    V = np.diff(T, axis=0)
    speed = np.linalg.norm(V, axis=1)
    tv = float(np.linalg.norm(np.diff(V, axis=0), axis=1).sum())
    tv_ratio = tv / max(float(speed.sum()), 1e-9)
    arc_cmd = abs(b0 - overshoot - a0) * (r + 0.08)
    dragged = float(np.linalg.norm(T[-1] - start_pos))
    path_len = float(speed.sum())
    depth_tv = float(np.abs(np.diff(np.array(depths))).sum())
    push_r, push_az, push_dy = 0.0, 0.0, 0.0
    if band_start is not None:
        b_end = cl.pos.numpy()[band].mean(axis=0)
        bd = b_end - band_start
        push_r = float(np.dot(bd, push_dir))           # radial: + = away from shell
        # A hanging flank that gets shoved swings AROUND the sphere: measure
        # the azimuthal sweep of the band (positive = onward, away from the
        # arriving sheet -- the drag runs from +azimuth to -azimuth). The
        # anchor itself rides the visible face (depth rule), so anchor-frame
        # axes cannot see this; the sweep angle can.
        th0 = math.atan2(band_start[0], band_start[2])
        th1 = math.atan2(b_end[0], b_end[2])
        push_az = float(np.degrees(th0 - th1))
        push_dy = float(bd[1])
    p.flags &= ~AnchorFlag.ACTIVE
    for _ in range(20):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps); log()
    tail = log.samples[-4:]
    json.dump(dict(traj=T.tolist(), depths=depths),
              open(f"/tmp/crossdrag_{seed}.json", "w"))
    return dict(scenario="cross_drag", seed=seed,
                tv_ratio=round(tv_ratio, 3), depth_tv=round(depth_tv, 3),
                dragged=round(dragged, 3), path_len=round(path_len, 3),
                arc_cmd=round(arc_cmd, 3),
                stall=round(1.0 - min(1.0, dragged / arc_cmd), 3),
                push_az=round(push_az, 1), push_r=round(push_r, 4),
                push_dy=round(push_dy, 4),
                ms_first=round(float(np.mean(frame_ms[:20])), 1),
                ms_contact=round(float(np.mean(frame_ms[40:])), 1),
                ms_max=round(float(max(frame_ms)), 1),
                worst=log.worst, bad_streak=log.max_streak,
                post_viol=min(t[1] for t in tail), post_gap=min(t[0] for t in tail))


# ---------------------------------------------------------------------------
# ROD test rig: multi-fold contact drags against an INFINITE-CYLINDER collider
# (COLLIDER_KIND=1: z-axis rod through (0, 1.5), R=0.35 -- see cloth.py). A
# horizontal rod makes parallel folds trivially reproducible: drape over it ->
# two hanging halves; drag along/across it -> accordion folds with sustained
# line contact. Invocation: the rod scenarios need the cylinder compiled in
# (COLLIDER_KIND is an import-time wp.constant), so main() RE-EXECS the process
# with COLLIDER_KIND=1 when needed; on a cloth module without the cylinder
# (pristine) they still run, degrading to a 0.35 sphere at the same center
# (noted in the output as kind=0). scenario_clothesline is the no-collider
# CONTROL: the same fold-squeeze geometry with zero collider involvement.
# ---------------------------------------------------------------------------

ROD_R = 0.35
ROD_CY = 1.5


def _damped_settle(cl, damped=120, free=15, every=6):
    """Warmup for line-supported drapes (rod / clothesline): the two hanging
    halves form a barely-damped pendulum that swings for hundreds of frames
    (measured: amplitude still ~0.9 m after 90 free frames), so kill the swing
    energy by zeroing velocities every few frames, then run a short free tail.
    Scaffolding for reaching a static drape only -- the scenarios' drags run
    fully free."""
    for f in range(damped + free):
        cl.simulate(steps=C.numSubsteps)
        if f < damped and f % every == every - 1:
            cl.vel.zero_()


def _rod_build(nx):
    """Cloth draped over the rod (z-cylinder at (0, 1.5), R=0.35), both halves
    hanging and settled. On a pristine cloth module this is a 0.35 sphere."""
    cl = build(nx, radius=ROD_R, warmup=0)
    _damped_settle(cl)
    return cl


def _lock_far_edge(cl):
    """End-stop for the along-rod scenarios: LOCK the cloth's far edge column
    (z = +1.5) at its settled drape position. Without it the whole drape
    slides along the frictionless rod like curtain rings on a rail (measured:
    dragged_z 0.94, strip compression 1.04 -- zero bunching); with the end
    fixed, pushed fabric MUST pleat into accordion folds between the grip and
    the stop, which is the geometry the rod rig exists to produce. Harness
    scaffolding, same mechanism as the clothesline's locked row."""
    from cloth import AnchorFlag, Particle
    n_side = int(round(np.sqrt(cl.numParticles)))
    im = cl.hostInvMass.numpy()
    for xi in range(n_side):
        pid = xi * n_side + (n_side - 1)
        pr = Particle(id=pid, screen=wp.vec2(0.0, 0.0),
                      mass=float(im[pid]), depth=0.0)
        pr.flags = AnchorFlag.LOCKED
        cl.anchors.append(pr)
        im[pid] = 0.0


def _azimuth_xy(pos3, cx, cy):
    """Azimuth of a point around the z-axis line through (cx, cy): 0 = straight
    down, positive toward +x."""
    return math.atan2(pos3[0] - cx, -(pos3[1] - cy))


def _instrumented_drag(cl, met, log, p, waypoints, spt):
    """Drive an active grab along projected world waypoints, recording anchor
    trajectory, per-frame wall ms and depth (the cross_drag metric core)."""
    traj, depths, frame_ms = [], [], []
    last = spt
    for w in waypoints:
        sp = project(w)
        if sp is not None:
            last = sp
        p.screen = wp.vec2(float(last[0]), float(last[1]))
        t0 = time.perf_counter()
        cl.update_anchors()
        cl.simulate(steps=C.numSubsteps)
        wp.synchronize()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        traj.append(cl.pos.numpy()[p.id].copy())
        depths.append(float(p.depth))
        if (len(traj) - 1) % 3 == 0:
            log()
    return np.array(traj), depths, frame_ms


def _drag_stats(traj, frame_ms, start_pos, cmd):
    """cross_drag's authority/shakiness/frame-time metric set."""
    V = np.diff(traj, axis=0)
    speed = np.linalg.norm(V, axis=1)
    tv = float(np.linalg.norm(np.diff(V, axis=0), axis=1).sum())
    dragged = float(np.linalg.norm(traj[-1] - start_pos))
    n = len(frame_ms)
    return dict(
        tv_ratio=round(tv / max(float(speed.sum()), 1e-9), 3),
        dragged=round(dragged, 3),
        path_len=round(float(speed.sum()), 3),
        arc_cmd=round(float(cmd), 3),
        stall=round(1.0 - min(1.0, dragged / max(cmd, 1e-9)), 3),
        ms_first=round(float(np.mean(frame_ms[:max(1, n // 4)])), 1),
        ms_contact=round(float(np.mean(frame_ms[n // 2:])), 1),
        ms_max=round(float(max(frame_ms)), 1))


def scenario_rod_cross(nx, seed):
    """The user's exact failing gesture on the rod: cloth draped over the
    horizontal z-rod (both halves hanging), grab the LEFT hanging half below
    the rod and drag it horizontally ACROSS into the right half -- sustained
    symmetric fold-vs-fold squeeze, with the collider only holding the drape."""
    from cloth import AnchorFlag
    rng = np.random.default_rng(seed)
    kind = getattr(C, "colliderKind", 0)
    cl = _rod_build(nx)
    met = Metrics(cl)
    log = ScenarioLog(met)
    # Arc the drag UNDER the rod (the xy analog of cross_drag's sweep across
    # the sphere's visible face): a straight horizontal drag lets the other
    # half escape by feeding OVER the frictionless rod like a rope on a pulley
    # (measured: band swings away, zero contact violations). Sweeping the
    # grabbed sheet around the rod's underside and UP the far flank pinches
    # the other hanging half between the arriving sheet and the rod -- the
    # sustained fold-vs-fold squeeze of the user's gesture.
    z0 = float(rng.uniform(0.2, 0.8))
    rr = ROD_R + 0.07
    # phi0 low on the flank: sight lines to the upper flank (|phi| ~ 70-90 deg)
    # pass through the rod's xy silhouette from the y=1 camera -- genuinely
    # occluded fabric, drag_anchor rightly refuses it (verified: the ray enters
    # the cylinder at z~2-3, in front of the fabric)
    phi0 = -np.radians(float(rng.uniform(50, 60)))   # left flank, visible zone
    phi1 = np.radians(float(rng.uniform(100, 115)))  # up the right flank
    press = [rr * math.sin(phi0), ROD_CY - rr * math.cos(phi0), z0]
    pid, spt = _pick_near(cl, press)
    p = cl.drag_anchor(float(spt[0]), float(spt[1]))
    if p is None:
        return dict(scenario="rod_cross", seed=seed, kind=kind, error="no pick")
    P0 = cl.pos.numpy()
    band = np.nonzero(np.linalg.norm(
        P0 - np.array([ROD_R + 0.02, 1.3, z0]), axis=1) < 0.18)[0]
    band_start = P0[band].mean(axis=0) if len(band) else None
    start_pos = P0[p.id].copy()
    n_frames = 80
    ws = []
    for i in range(n_frames):
        phi = phi0 + (i + 1) / n_frames * (phi1 - phi0)
        ws.append(np.array([rr * math.sin(phi), ROD_CY - rr * math.cos(phi), z0]))
    traj, depths, frame_ms = _instrumented_drag(cl, met, log, p, ws, spt)
    out = _drag_stats(traj, frame_ms, start_pos, rr * (phi1 - phi0))
    push_az, push_x, push_dy = 0.0, 0.0, 0.0
    if band_start is not None:
        b_end = cl.pos.numpy()[band].mean(axis=0)
        bd = b_end - band_start
        # the contacted half's swing around the rod axis: + = pushed onward (+x)
        push_az = float(np.degrees(_azimuth_xy(b_end, 0.0, ROD_CY)
                                   - _azimuth_xy(band_start, 0.0, ROD_CY)))
        push_x = float(bd[0])
        push_dy = float(bd[1])
    p.flags &= ~AnchorFlag.ACTIVE
    for _ in range(20):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps); log()
    tail = log.samples[-4:]
    out.update(scenario="rod_cross", seed=seed, kind=kind,
               depth_tv=round(float(np.abs(np.diff(np.array(depths))).sum()), 3),
               push_az=round(push_az, 1), push_x=round(push_x, 4),
               push_dy=round(push_dy, 4),
               worst=log.worst, bad_streak=log.max_streak,
               post_viol=min(t[1] for t in tail), post_gap=min(t[0] for t in tail))
    return out


def scenario_rod_slide(nx, seed):
    """Drag a grab at the rod's top ridge ALONG the rod axis (+z) so the drape
    ahead of the grip bunches into accordion folds. Reports drag authority vs
    commanded travel, fold proxies (contact violations while pressed + the
    z-compression of the fabric strip ahead of the grip), and frame times."""
    from cloth import AnchorFlag
    rng = np.random.default_rng(seed)
    kind = getattr(C, "colliderKind", 0)
    cl = _rod_build(nx)
    met = Metrics(cl)
    log = ScenarioLog(met)
    # SIDE view + flank grab. Two control facts force this staging: (1) the
    # rod occludes its own top ridge from the default low camera (the pick ray
    # enters the cylinder before ridge fabric -- drag_anchor rightly refuses);
    # (2) from the default camera the rod axis IS the view axis, so an
    # along-rod drag is a drag in DEPTH, which the grab control scheme has no
    # authority over (fixed grab depth; measured stall ~0.8 with zero
    # contact). From the side, along-rod is a plain horizontal drag -- the
    # curtain-on-a-rail gesture: pleats bunch ahead of the grip.
    set_camera((5.0, 1.2, 0.0), fwd=(-1.0, 0.0, 0.0))
    _lock_far_edge(cl)
    yg = float(rng.uniform(1.1, 1.3))
    z0 = float(rng.uniform(-0.45, -0.1))
    travel = float(rng.uniform(1.15, 1.35))  # deep into the accordion: the
    # last ~0.4 m squeezes the formed pleats against the end-stop
    pid, spt = _pick_near(cl, [ROD_R + 0.015, yg, z0])   # near flank (x > 0)
    p = cl.drag_anchor(float(spt[0]), float(spt[1]))
    if p is None:
        return dict(scenario="rod_slide", seed=seed, kind=kind, error="no pick")
    P0 = cl.pos.numpy()
    strip = np.nonzero((P0[:, 0] > 0.2) & (P0[:, 1] > 0.6)
                       & (P0[:, 2] > z0 + 0.15) & (P0[:, 2] < 1.45))[0]
    ext0 = float(P0[strip, 2].max() - P0[strip, 2].min()) if len(strip) else 0.0
    start_pos = P0[p.id].copy()
    n_frames = 70
    ws = [np.array([ROD_R + 0.07, yg, z0 + (i + 1) / n_frames * travel])
          for i in range(n_frames)]
    traj, depths, frame_ms = _instrumented_drag(cl, met, log, p, ws, spt)
    out = _drag_stats(traj, frame_ms, start_pos, travel)
    P1 = cl.pos.numpy()
    ext1 = float(P1[strip, 2].max() - P1[strip, 2].min()) if len(strip) else 0.0
    dz = float(P1[p.id][2] - start_pos[2])
    mid = len(log.samples) // 2
    viol_contact = int(np.median([s[1] for s in log.samples[mid:]])) \
        if len(log.samples) > mid else 0
    p.flags &= ~AnchorFlag.ACTIVE
    for _ in range(20):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps); log()
    tail = log.samples[-4:]
    out.update(scenario="rod_slide", seed=seed, kind=kind,
               dragged_z=round(dz, 3),
               strip_compress=round(ext1 / ext0, 3) if ext0 else None,
               viol_contact=viol_contact,
               worst=log.worst, bad_streak=log.max_streak,
               post_viol=min(t[1] for t in tail), post_gap=min(t[0] for t in tail))
    return out


def scenario_rod_pile_push(nx, seed):
    """Pre-fold the drape into 2-3 accordion folds resting against each other
    (successive short +z drags at the rod top), then PUSH the whole stack
    further along the rod -- 'pushing multiple folds is impossible'. Authority
    of the final push (stall/dragged) is the headline metric; pile_dz tracks
    how far the fold mass itself moved."""
    from cloth import AnchorFlag
    rng = np.random.default_rng(seed)
    kind = getattr(C, "colliderKind", 0)
    cl = _rod_build(nx)
    met = Metrics(cl)
    log = ScenarioLog(met)
    # Side view + near-flank grabs (same control staging as scenario_rod_slide):
    # successive short +z pushes on the hanging half pleat it like a curtain
    # on a rail -- folds resting against each other on one side.
    set_camera((5.0, 1.2, 0.0), fwd=(-1.0, 0.0, 0.0))
    _lock_far_edge(cl)
    yg = float(rng.uniform(1.1, 1.25))
    xg = ROD_R + 0.015
    z_grab = float(rng.uniform(-0.55, -0.35))
    n_pre = int(rng.integers(2, 4))
    pre_effs = []
    for _k in range(n_pre):
        # fabric keeps feeding forward, so grabbing at the same world z each
        # time pushes fresh material into the fold zone ahead
        eff, _ = drag_to(cl, met, [xg, yg, z_grab],
                         [xg, yg, z_grab + 0.45], int(rng.integers(18, 24)),
                         observe=log, settle=8)
        if eff is not None:
            pre_effs.append(round(eff, 3))
    pid, spt = _pick_near(cl, [xg, yg, z_grab + 0.15])
    p = cl.drag_anchor(float(spt[0]), float(spt[1]))
    if p is None:
        return dict(scenario="rod_pile_push", seed=seed, kind=kind,
                    error="no pick", pre_eff=pre_effs)
    P0 = cl.pos.numpy()
    start_pos = P0[p.id].copy()
    pile = np.nonzero((P0[:, 0] > 0.2) & (P0[:, 1] > 0.6)
                      & (P0[:, 2] > start_pos[2] + 0.05))[0]
    push = 0.9
    n_frames = 60
    ws = [np.array([ROD_R + 0.07, yg, start_pos[2] + (i + 1) / n_frames * push])
          for i in range(n_frames)]
    traj, depths, frame_ms = _instrumented_drag(cl, met, log, p, ws, spt)
    out = _drag_stats(traj, frame_ms, start_pos, push)
    pile_dz = float((cl.pos.numpy()[pile, 2] - P0[pile, 2]).mean()) \
        if len(pile) else 0.0
    p.flags &= ~AnchorFlag.ACTIVE
    for _ in range(20):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps); log()
    tail = log.samples[-4:]
    out.update(scenario="rod_pile_push", seed=seed, kind=kind,
               n_pre=n_pre, pre_eff=pre_effs, pile_dz=round(pile_dz, 3),
               worst=log.worst, bad_streak=log.max_streak,
               post_viol=min(t[1] for t in tail), post_gap=min(t[0] for t in tail))
    return out


def scenario_clothesline(nx, seed):
    """No-collider CONTROL for rod_cross: lock the cloth's center row (a line
    along z, real LOCKED-flag anchors) at y=1.6 so both halves hang in
    face-to-face contact, then drag one half sideways into the other -- the
    identical fold-squeeze geometry with ZERO collider involvement. Runs on any
    COLLIDER_KIND (the sphere is parked at (0, 8, 0))."""
    from cloth import AnchorFlag, Particle
    rng = np.random.default_rng(seed)
    line_y = 1.6
    cl = build(nx, sphere_center=(0.0, 8.0, 0.0), warmup=0, y_offset=line_y)
    n_side = int(round(np.sqrt(cl.numParticles)))
    xi_c = (n_side - 1) // 2
    im = cl.hostInvMass.numpy()
    for pid in range(xi_c * n_side, xi_c * n_side + n_side):
        pr = Particle(id=pid, screen=wp.vec2(0.0, 0.0),
                      mass=float(im[pid]), depth=0.0)
        pr.flags = AnchorFlag.LOCKED
        cl.anchors.append(pr)
        im[pid] = 0.0
    _damped_settle(cl)   # same barely-damped pendulum as the rod drape
    met = Metrics(cl)
    log = ScenarioLog(met)
    ay = float(rng.uniform(0.9, 1.2))
    z0 = float(rng.uniform(0.2, 0.8))
    # pick explicitly from the LEFT half: the halves hang nearly coincident in
    # screen space, so a blind hint-pick could land on either sheet
    P = cl.pos.numpy()
    rows = np.arange(cl.numParticles) // n_side
    left = np.nonzero(rows < xi_c - 4)[0]
    hint = np.array([-0.03, ay, z0])
    pid = int(left[np.argmin(np.linalg.norm(P[left] - hint, axis=1))])
    spt = project(P[pid])
    p = cl.drag_anchor(float(spt[0]), float(spt[1]))
    if p is None:
        return dict(scenario="clothesline", seed=seed, error="no pick")
    side = -1.0 if (p.id // n_side) < xi_c else 1.0  # sheet actually grabbed
    dirx = -side                                     # drag INTO the other half
    P0 = cl.pos.numpy()
    other = np.nonzero(rows > xi_c + 4)[0] if side < 0 \
        else np.nonzero(rows < xi_c - 4)[0]
    band = other[np.linalg.norm(P0[other] - np.array([0.0, ay, z0]), axis=1) < 0.18]
    band_start = P0[band].mean(axis=0) if len(band) else None
    start_pos = P0[p.id].copy()
    cmd = 0.55
    n_frames = 60
    ws = [np.array([start_pos[0] + (i + 1) / n_frames * cmd * dirx, ay, z0])
          for i in range(n_frames)]
    traj, depths, frame_ms = _instrumented_drag(cl, met, log, p, ws, spt)
    out = _drag_stats(traj, frame_ms, start_pos, cmd)
    push_az = 0.0
    if band_start is not None:
        b_end = cl.pos.numpy()[band].mean(axis=0)
        # swing of the pushed half around the LINE; + = pushed away from the drag
        push_az = float(np.degrees(_azimuth_xy(b_end, 0.0, line_y)
                                   - _azimuth_xy(band_start, 0.0, line_y)) * dirx)
    p.flags &= ~AnchorFlag.ACTIVE
    for _ in range(20):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps); log()
    tail = log.samples[-4:]
    out.update(scenario="clothesline", seed=seed, side=int(side),
               push_az=round(push_az, 1),
               worst=log.worst, bad_streak=log.max_streak,
               post_viol=min(t[1] for t in tail), post_gap=min(t[0] for t in tail))
    return out


def scenario_hang_press(nx, seed):
    """Cloth hung as a vertical curtain by its four corners; grab the left
    half, fold it onto the right half, and KEEP pressing for a sustained
    window. The taut pinned backing cannot yield or escape, so any guaranteed
    anchor advance must either rubber-band (correct) or force the dragged
    layer THROUGH the backing (the user-reported penetration). Metric:
    left-half vertices that end up BEHIND the backing plane in the press
    region, during the press and after release."""
    from cloth import AnchorFlag
    rng = np.random.default_rng(seed)
    cl = build(nx, sphere_center=(0.0, 8.0, 0.0), warmup=1)
    met = Metrics(cl)
    log = ScenarioLog(met)
    n = cl.num_x if hasattr(cl, "num_x") else int(round(math.sqrt(cl.numParticles)))
    half = 0.5 * (n - 1) * cl.spacing
    # re-pose the grid as a vertical curtain (x = u, y = v, z = 0), zero motion
    u = np.arange(n) * cl.spacing - half
    X, Y = np.meshgrid(u, u)                      # row-major: id = v*n + u
    P = np.stack([X.ravel(), Y.ravel() + 0.25 + 2 * half - (Y.ravel() + half), 
                  np.zeros(n * n)], axis=1)      # y in [0.25, 0.25+2*half] flipped
    P[:, 1] = 0.25 + (np.tile(np.arange(n), n) * 0.0)  # placeholder, fixed below
    V = np.repeat(np.arange(n), n) * cl.spacing
    P = np.stack([np.tile(u, n), 0.25 + V, np.zeros(n * n)], axis=1)
    wp.copy(cl.pos, wp.array(P, dtype=wp.vec3))
    wp.copy(cl.prevPos, wp.array(P, dtype=wp.vec3)) if hasattr(cl, "prevPos") else None
    if hasattr(cl, "vel"):
        cl.vel.zero_()
    # pin the four corners (direct inv-mass, like the rod rig's end-stop)
    inv = cl.invMass.numpy()
    corners = [0, n - 1, n * (n - 1), n * n - 1]
    for c in corners:
        inv[c] = 0.0
    wp.copy(cl.invMass, wp.array(inv, dtype=float))
    hm = cl.hostInvMass.numpy(); hm[corners] = 0.0
    for _ in range(30):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
    # grab mid-height on the left quarter, fold right, then sustained press
    P = cl.pos.numpy()
    grab_hint = np.array([-half * 0.9, 0.25 + half, 0.06])
    pid, spt = _pick_near(cl, grab_hint)
    p = cl.drag_anchor(float(spt[0]), float(spt[1]))
    if p is None:
        return dict(scenario="hang_press", seed=seed, error="no pick")
    left_ids = np.nonzero(np.tile(u, n) < -0.15 * half)[0]
    right_ids = np.nonzero(np.tile(u, n) > 0.15 * half)[0]
    wrap_max = 0
    snap = []
    start = P[p.id].copy()
    # waypoint path: swing out toward camera, then across to the right half,
    # then KEEP COMMANDING rightward/into the curtain (sustained press)
    path = []
    for t in np.linspace(0, 1, 25):                     # fold out+across
        path.append([start[0] + (half * 0.8 - start[0]) * t,
                     start[1], 0.10 + 0.10 * math.sin(math.pi * t)])
    for i in range(75):                                 # sustained press
        path.append([half * 0.8 + i * 0.004, start[1], 0.02])
    pen_press, pen_max = 0, 0
    frame_ms = []
    for i, wpt in enumerate(path):
        sp = project(wpt)
        if sp is not None:
            p.screen = wp.vec2(float(sp[0]), float(sp[1]))
        t0 = time.perf_counter()
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
        wp.synchronize()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        if i % 3 == 0:
            log()
            PP = cl.pos.numpy()
            anc = PP[p.id]
            lp = PP[left_ids]
            rp = PP[right_ids]
            # true pass-through: left-half fabric BEHIND the backing in the
            # press band around the anchor, with right-half fabric verifiably
            # in FRONT of it there (excludes wrap-around the free edges)
            band = (np.abs(lp[:, 0] - anc[0]) < 0.30) & \
                   (np.abs(lp[:, 1] - anc[1]) < 0.30)
            rband = (np.abs(rp[:, 0] - anc[0]) < 0.35) & \
                    (np.abs(rp[:, 1] - anc[1]) < 0.35)
            if rband.any():
                back_z = float(np.median(rp[rband, 2]))
                pen_press = int((band & (lp[:, 2] < back_z - 3.0 * C.thickness)).sum())
            else:
                pen_press = 0
            wrap = int(((lp[:, 2] < -3.0 * C.thickness) & ~band).sum())
            pen_max = max(pen_max, pen_press)
            wrap_max = max(wrap_max, wrap)
            if i == 60:
                zs = np.sort(PP[np.abs(PP[:, 0] - anc[0]) + np.abs(PP[:, 1] - anc[1]) < 0.08][:, 2])
                snap = [round(float(z), 4) for z in zs[:: max(1, len(zs) // 8)]]
    p.flags &= ~AnchorFlag.ACTIVE
    for _ in range(25):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
    log()
    PP = cl.pos.numpy()
    anc = PP[p.id]
    lp = PP[left_ids]; rp = PP[right_ids]
    band = (np.abs(lp[:, 0] - anc[0]) < 0.30) & (np.abs(lp[:, 1] - anc[1]) < 0.30)
    rband = (np.abs(rp[:, 0] - anc[0]) < 0.35) & (np.abs(rp[:, 1] - anc[1]) < 0.35)
    pen_post = 0
    if rband.any():
        back_z = float(np.median(rp[rband, 2]))
        pen_post = int((band & (lp[:, 2] < back_z - 3.0 * C.thickness)).sum())
    tail = log.samples[-3:]
    ovf = int(cl.selfCollisionOverflow.numpy()[0])
    return dict(scenario="hang_press", seed=seed, overflow=ovf,
                pen_max=pen_max, pen_end=pen_press, pen_post=pen_post,
                wrap_max=wrap_max, z_profile=snap,
                ms_press=round(float(np.mean(frame_ms[30:])), 1),
                worst=log.worst,
                post_viol=min(t[1] for t in tail), post_gap=min(t[0] for t in tail))


def scenario_stretch_drag(nx, seed):
    """The user's extreme case: grab the hanging hem and keep dragging far past
    the cloth's span, so the fabric goes taut and can no longer slide with the
    grip -- then keep pulling. Exact crossings counted throughout."""
    from cloth import AnchorFlag
    rng = np.random.default_rng(seed)
    cl = build(nx)
    met = Metrics(cl)
    sc = np.array([0.0, 1.5, 0.0]); r = 0.5
    a = rng.uniform(-0.6, 0.6)
    press = np.array([(r + 0.05) * np.sin(a), rng.uniform(0.35, 0.6),
                      (r + 0.05) * np.cos(a)])          # low on the hanging hem
    pid, spt = _pick_near(cl, press)
    p = cl.drag_anchor(float(spt[0]), float(spt[1]))
    if p is None:
        return dict(scenario="stretch_drag", seed=seed, error="no pick")
    start = cl.pos.numpy()[p.id].copy()
    # drag horizontally far past the span (drape hangs ~1.5 m of fabric; the
    # target is 2.4 m out), then KEEP commanding for a sustained taut window
    tdir = np.array([np.sin(a + rng.uniform(1.2, 1.9)), 0.0,
                     np.cos(a + rng.uniform(1.2, 1.9))])
    xmax, xworst, vworst = 0, 0, 0
    last = spt
    frame_ms = []
    for i in range(95):
        w = start + tdir * min(2.4, (i + 1) * 0.045) + np.array([0, 0.25, 0]) \
            * min(1.0, i / 30.0)
        sp = project(w)
        if sp is not None:
            last = sp
        p.screen = wp.vec2(float(last[0]), float(last[1]))
        t0 = time.perf_counter()
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
        wp.synchronize()
        frame_ms.append((time.perf_counter() - t0) * 1e3)
        if i % 5 == 4:
            n, pts = global_xings(cl)
            xmax = max(xmax, n)
            g, v, fl, spn = met.sample()
            vworst = max(vworst, v)
    p.flags &= ~AnchorFlag.ACTIVE
    for _ in range(25):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
    n_post, _ = global_xings(cl)
    g, v, fl, spn = met.sample()
    anc_end = cl.pos.numpy()[p.id] if p.id < cl.numParticles else start
    return dict(scenario="stretch_drag", seed=seed,
                xings_max=xmax, xings_post=n_post,
                viol_worst=vworst, post_viol=v, post_flips=fl,
                dragged=round(float(np.linalg.norm(anc_end - start)), 3),
                ms_press=round(float(np.mean(frame_ms[45:])), 1))


def scenario_gentle_slide(nx, seed):
    """Bulb-artifact quantifier v2: fold the drape over the sphere once so the
    grip presses LAYERED fabric (where the bulb is most visible), grab ON the
    2-layer stack and slide it slowly down the shell. A grip-guard volume
    shows up as a radial RIDGE around the anchor: fabric in an annulus
    1.2-2.5*fingerRadius displaced outward relative to a 3.0-4.5*fingerRadius
    CONTROL annulus of the same stack -- the differential cancels the stack's
    own thickness, which the v1 absolute-clearance metric could not (v1
    discriminated weakly on single-layer fabric: ball-on bulb_mean 0.041 vs
    ball-off 0.035). ridge_mean/ridge_p95 are the primary fields; the v1
    absolute fields and exact crossings are kept (a guard that trades bulb
    for crossings loses)."""
    from cloth import AnchorFlag
    rng = np.random.default_rng(seed)
    cl = build(nx)
    met = Metrics(cl)
    sc = np.array([0.0, 1.5, 0.0]); r = 0.5
    # fold a flap over the apex (fold_slide's first-drag geometry, ranges
    # tightened so the fold lands reliably on top)
    a = rng.uniform(0, 2 * np.pi)
    press = sc + [(r + 0.05) * np.sin(a), rng.uniform(-0.30, -0.20),
                  (r + 0.05) * np.cos(a)]
    target = sc + [-0.35 * np.sin(a), r + rng.uniform(0.12, 0.20),
                   -0.35 * np.cos(a)]
    drag_to(cl, met, press, target, 32, settle=14)
    # grab ON the folded stack near its highest point
    P = cl.pos.numpy()
    top = P[np.argmax(P[:, 1])].copy()
    pid, spt = _pick_near(cl, top)
    p = cl.drag_anchor(float(spt[0]), float(spt[1]))
    if p is None:
        return dict(scenario="gentle_slide", seed=seed, error="no pick")
    fr = C.fingerRadius
    P = cl.pos.numpy()
    anc0 = P[p.id]
    # layering proxy at the grip (a 2-layer grab has ~2x the single-layer count)
    n_under = int((np.linalg.norm(P - anc0, axis=1) < 1.2 * fr).sum())

    def sample_bulb(P, anc, members):
        """(n_hug, ridge_mean, ridge_p95): n_hug = COUNT of free vertices
        within 1.0*fingerRadius of the anchor -- the fabric that hugs the
        grip. A guard ball of radius >= fingerRadius EMPTIES this ball (the
        carved void that feeds the visual bulb); a guard smaller than the
        patch leaves the backing layer in place; no guard leaves everything.
        (v3 lesson: a p10-of-distance measure was swamped by the same-sheet
        skirt at patch-rim distance, identical across configs; the raw count
        of surviving fabric is the direct, integer-robust signature.)
        ridge = clearance of a tight annulus around the grip minus a control
        annulus of the same stack (the outward bulge of ejected fabric)."""
        d_anc = np.linalg.norm(P - anc, axis=1)
        clr = np.linalg.norm(P - sc, axis=1) - r
        n_hug = int(((~members) & (d_anc < 1.0 * fr)).sum())
        near = clr < 0.15                      # on/near-shell fabric only
        ann = (d_anc > 1.0 * fr) & (d_anc < 1.7 * fr) & near
        ctl = (d_anc > 2.5 * fr) & (d_anc < 4.0 * fr) & near
        rm, rp = 0.0, 0.0
        if ann.any() and ctl.any():
            base = float(clr[ctl].mean())
            rm = float(clr[ann].mean()) - base
            rp = float(np.percentile(clr[ann], 95)) - base
        return n_hug, rm, rp

    # HOLD phase: the guard volume's signature with no plow-front confound
    # (v2 lesson: sampling only during the slide buried the bulb under the
    # bunching every config produces ahead of a moving grip).
    hug_hold, ridge_hold, ridgep_hold = 10 ** 9, 0.0, 0.0
    for i in range(12):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
        if i % 3 == 2:
            P = cl.pos.numpy()
            members = cl.invMass.numpy() == 0.0
            nh, rm, rp = sample_bulb(P, P[p.id], members)
            hug_hold = min(hug_hold, nh)
            ridge_hold = max(ridge_hold, rm)
            ridgep_hold = max(ridgep_hold, rp)
    # slide path: down the great circle through the grab point (fall back to
    # the fold direction when grabbed at the exact apex)
    u = anc0 - sc
    exz = np.array([u[0], 0.0, u[2]])
    nexz = np.linalg.norm(exz)
    exz = exz / nexz if nexz > 1e-6 else np.array([-np.sin(a), 0.0, -np.cos(a)])
    th_g = math.acos(np.clip(u[1] / max(np.linalg.norm(u), 1e-9), -1.0, 1.0))
    hug_slide, ridge_slide, xmax = 10 ** 9, 0.0, 0
    last = spt
    for i in range(50):
        th = th_g + (i + 1) / 50.0 * 0.9       # slow ~0.45 m slide down the shell
        w = sc + exz * (r + 0.06) * np.sin(th) \
            + np.array([0.0, (r + 0.06) * np.cos(th), 0.0])
        sp = project(w)
        if sp is not None:
            last = sp
        p.screen = wp.vec2(float(last[0]), float(last[1]))
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
        if i % 5 == 4:
            P = cl.pos.numpy()
            members = cl.invMass.numpy() == 0.0
            nh, rm, rp = sample_bulb(P, P[p.id], members)
            hug_slide = min(hug_slide, nh)
            ridge_slide = max(ridge_slide, rm)
            n, _ = global_xings(cl)
            xmax = max(xmax, n)
    p.flags &= ~AnchorFlag.ACTIVE
    for _ in range(20):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
    n_post, _ = global_xings(cl)
    g, v, fl, spn = met.sample()
    return dict(scenario="gentle_slide", seed=seed,
                hug_hold=hug_hold, ridge_hold=round(ridge_hold, 4),
                ridgep_hold=round(ridgep_hold, 4),
                hug_slide=hug_slide,
                ridge_slide=round(ridge_slide, 4), n_under=n_under,
                xings_max=xmax, xings_post=n_post, post_viol=v)


def scenario_valley_plow(nx, seed):
    """The user's reproduction: drag horizontally from the left flank through a
    valley so wrinkles pile up ABOVE the anchor, keep plowing right — far
    enough and the dragged sheet penetrates through the accumulated wrinkles.
    Long constant-speed horizontal plow at sub-equator height; exact crossings
    with near-anchor attribution sampled throughout."""
    from cloth import AnchorFlag
    rng = np.random.default_rng(seed)
    cl = build(nx)
    met = Metrics(cl)
    sc = np.array([0.0, 1.5, 0.0]); r = 0.5
    # grab low on the LEFT flank front quadrant (below the equator: wrinkles
    # the plow raises bunch above the grip)
    a0 = np.radians(rng.uniform(-75, -50))
    gy = 1.5 - rng.uniform(0.05, 0.20)
    rr = math.sqrt(max(r * r - (gy - 1.5) ** 2, 0.02))
    press = np.array([(rr + 0.03) * np.sin(a0), gy, (rr + 0.03) * np.cos(a0)])
    pid, spt = _pick_near(cl, press)
    p = cl.drag_anchor(float(spt[0]), float(spt[1]))
    if p is None:
        return dict(scenario="valley_plow", seed=seed, error="no pick")
    n_frames = 130
    a1 = np.radians(rng.uniform(55, 80))     # far right — "a bit far enough"
    xmax, xend, near_max = 0, 0, 0
    viol_max = 0
    xseries = []
    last = spt
    for i in range(n_frames):
        th = a0 + (i + 1) / n_frames * (a1 - a0)
        w = np.array([(rr + 0.05) * np.sin(th), gy, (rr + 0.05) * np.cos(th)])
        sp = project(w)
        if sp is not None:
            last = sp
        p.screen = wp.vec2(float(last[0]), float(last[1]))
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
        if i % 5 == 4:
            n, pts = global_xings(cl)
            xmax = max(xmax, n)
            anc = cl.pos.numpy()[p.id]
            near = int((np.linalg.norm(pts - anc, axis=1) < 0.2).sum()) if n else 0
            near_max = max(near_max, near)
            g, v, fl, spn = met.sample()
            viol_max = max(viol_max, v)
            xseries.append(n)
    xend = xseries[-1] if xseries else 0
    p.flags &= ~AnchorFlag.ACTIVE
    for _ in range(25):
        cl.update_anchors(); cl.simulate(steps=C.numSubsteps)
    n_post, _ = global_xings(cl)
    g, v, fl, spn = met.sample()
    return dict(scenario="valley_plow", seed=seed,
                xings_max=xmax, xings_near_anchor=near_max, xings_end=xend,
                xings_post=n_post, viol_max=viol_max, post_viol=v,
                xseries=xseries)


SCENARIOS = {"cross_drag": scenario_cross_drag, "hang_press": scenario_hang_press,
             "valley_plow": scenario_valley_plow,
             "gentle_slide": scenario_gentle_slide,
             "stretch_drag": scenario_stretch_drag,
             "fold_slide": scenario_fold_slide, "collapse": scenario_collapse,
             "flatten": scenario_flatten, "rigidity": scenario_rigidity,
             "rod_cross": scenario_rod_cross, "rod_slide": scenario_rod_slide,
             "rod_pile_push": scenario_rod_pile_push,
             "clothesline": scenario_clothesline}

# rod scenarios need the cylinder collider compiled in (import-time knob)
ROD_KIND_SCENARIOS = {"rod_cross", "rod_slide", "rod_pile_push"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000, help="first seed of the batch")
    ap.add_argument("--strokes", type=int, default=3, help="strokes per session")
    ap.add_argument("--family", choices=["drag", "sphere", "mixed", "all"], default="drag",
                    help="gesture family: cloth drags, sphere moves/resizes, both at once, "
                         "or round-robin across the three")
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None, help="run a single seeded session")
    ap.add_argument("--replay", type=str, default=None)
    ap.add_argument("--trace", type=str, default=None,
                    help="with --replay: write per-sample site-classified records to this JSON")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default=None,
                    help="run a named use-case scenario instead of random fuzz")
    ap.add_argument("--replay-app", type=str, default=None,
                    help="replay a CLOTH_RECORD app session (JSONL) and probe "
                         "exact crossings; --nx must match the app's cloth size")
    args = ap.parse_args()

    if args.replay_app:
        replay_app_session(args.replay_app, args.nx)
        return

    if args.scenario in ROD_KIND_SCENARIOS and getattr(C, "colliderKind", 0) != 1:
        # COLLIDER_KIND is an import-time wp.constant: re-exec once with the
        # cylinder compiled in. A cloth module without the knob (pristine)
        # ignores the env var; after one re-exec we proceed against the sphere
        # fallback rather than looping.
        if os.environ.get("_ROD_KIND_REEXEC") != "1":
            env = dict(os.environ, COLLIDER_KIND="1", _ROD_KIND_REEXEC="1")
            os.execve(sys.executable, [sys.executable] + sys.argv, env)
        print("[scenario] NOTE: cloth module has no cylinder collider "
              "(COLLIDER_KIND ignored); rod scenario runs against the SPHERE "
              "fallback, radius 0.35 at (0, 1.5, 0).", flush=True)

    if args.scenario:
        seeds = [args.seed] if args.seed is not None else \
            list(range(args.seed0, args.seed0 + args.sessions))
        for sd in seeds:
            t0 = time.perf_counter()
            r = SCENARIOS[args.scenario](args.nx, sd)
            dt = time.perf_counter() - t0
            print(f"[scenario] {json.dumps(r)} ({dt:.0f}s)", flush=True)
        return

    if args.replay:
        session = json.load(open(args.replay))
        cl = build(session.get("nx", args.nx))
        met = Metrics(cl)
        if args.trace:
            session["_trace"] = []
        failed, r = run_session(cl, met, session, verbose=True)
        print(f"[replay] failed={failed} {r}", flush=True)
        if args.trace:
            json.dump(session["_trace"], open(args.trace, "w"), indent=1)
            print(f"[replay] trace -> {args.trace}", flush=True)
        return

    seeds = [args.seed] if args.seed is not None else list(range(args.seed0, args.seed0 + args.sessions))
    fails = 0
    post_viols, trans_viols = [], []
    per_family = {}
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        family = args.family if args.family != "all" else ("drag", "sphere", "mixed")[k % 3]
        cl = build(args.nx)
        met = Metrics(cl)
        bbox = cloth_screen_bbox(cl)
        strokes = []
        P = cl.pos.numpy()
        for si in range(args.strokes):
            if family == "sphere" or (family == "mixed" and si % 2 == 1 and rng.random() < 0.5):
                strokes.append(gen_sphere_stroke(rng))
                continue
            for _try in range(6):
                pid = int(rng.integers(0, cl.numParticles))
                spt = project(P[pid])
                if spt and 0 < spt[0] < W and 0 < spt[1] < H:
                    break
            st = gen_stroke(rng, spt, bbox)
            if family == "mixed" and rng.random() < 0.7:
                # human drags cloth while nudging the sphere with the other hand
                v = rng.normal(size=3) * [0.02, 0.012, 0.02]
                st["sphere_drift"] = [float(x) for x in v]
            strokes.append(st)
        session = {"seed": seed, "nx": args.nx, "family": family, "strokes": strokes}
        t0 = time.perf_counter()
        failed, r = run_session(cl, met, session, verbose=args.verbose)
        dt = time.perf_counter() - t0
        post_viols.append(r['post_viol']); trans_viols.append(r['worst_transient']['viol'])
        fam = per_family.setdefault(family, dict(n=0, fails=0, notice=0, post=0))
        fam["n"] += 1; fam["fails"] += int(failed); fam["notice"] += int(r["noticeable"])
        fam["post"] += r["post_viol"]
        tag = "FAIL" if failed else ("NOTC" if r["noticeable"] else "ok  ")
        sk = r["streaks"]
        print(f"[{tag}] seed={seed} {family:6s} ({dt:.0f}s) post: gap={r['post_gap']:.5f} "
              f"viol={r['post_viol']} flips={r['post_flips']} sphere={r['post_sphere']:.4f}{r['site']} | "
              f"streaks v={sk['viol']} s={sk['spen']} f={sk['flip']} | worst transient: "
              f"viol={r['worst_transient']['viol']} flips={r['worst_transient']['flips']} "
              f"sphere={r['worst_transient']['sphere']:.4f}", flush=True)
        if failed or r["noticeable"]:
            fails += int(failed)
            path = f"/tmp/fuzz_fail_{seed}.json"
            json.dump(session, open(path, "w"))
            print(f"        replay: {path}", flush=True)
    print(f"[fuzz] {fails}/{len(seeds)} sessions failed | aggregate post_viol sum={sum(post_viols)} "
          f"mean={np.mean(post_viols):.1f} | transient viol mean={np.mean(trans_viols):.0f}", flush=True)
    for family, f in sorted(per_family.items()):
        print(f"[fuzz]   {family:6s}: {f['fails']}/{f['n']} fail, {f['notice']}/{f['n']} noticeable, "
              f"post_viol sum={f['post']}", flush=True)


if __name__ == "__main__":
    main()
