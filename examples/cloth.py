#!/usr/bin/env -S uv run --script

# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "shaderbang",
#     "pyopengl",
#     "warp-lang",
# ]
#
# [tool.uv.sources]
# shaderbang = { git = "https://github.com/astefanutti/shaderbang.git", branch = "main" }
# ///

"""
Cloth Simulation
================

GPU-accelerated cloth simulation using NVIDIA Warp with XPBD physics,
adapted to run with Shaderbang.

Inspired by Matthias Mueller's Ten Minute Physics:
https://matthias-research.github.io/pages/tenMinutePhysics/
https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/16-GPUCloth.py

Keyboard Controls
-----------------
    P               Pause / resume the simulation
    Space / Right   Advance one simulation step (works while paused)
    S               Cycle step granularity:
                      Frame step -> Sub-step -> Contact step -> Frame step
    R               Reset the cloth to its initial state
    C               Toggle particle self-collision
    W               Toggle wireframe rendering
    F               Toggle back-face culling (shows front/back in different colors)
    Ctrl+A          Select all anchors
    Delete / Bksp   Remove selected anchors

Mouse Controls
--------------
    Left drag on cloth      Grab and drag a cloth particle
    Left drag elsewhere     Orbit the camera
    Right drag              Track (pan) the camera
    Scroll wheel            Dolly (zoom) the camera
    Ctrl + release          Lock the dragged particle as a persistent anchor
    Click (no drag)         Toggle anchor selection

Touchscreen Controls
--------------------
    1 finger on cloth       Grab and drag a cloth particle
    1 finger elsewhere      Orbit the camera (trackball)
    2-3 fingers             Track, dolly, and rotate the camera
    4 fingers               Translate the sphere
    5+ fingers              Rotate and resize the sphere

Trackpad Controls
-----------------
    2 fingers               Orbit, dolly, and rotate the camera
    3 fingers               Track, dolly, and rotate the camera
    4 fingers               Translate the sphere
    5+ fingers              Rotate and resize the sphere
"""


import argparse
import ctypes
import glob
import json
import os
import math
import stat
import sys
import signal
import threading
import time

import numpy as np
import warp as wp

from dataclasses import dataclass, field
from enum import auto, Flag
from typing import Callable, Generic, Optional, Self, TypeVar

from contextlib import ExitStack
from pathlib import Path
from signal import pthread_sigmask, pthread_kill, sigwait
from threading import main_thread, Thread

from libevdev import Device, EV_ABS, EV_KEY, EV_REL, INPUT_PROP_DIRECT, INPUT_PROP_POINTER

import shaderbang.input
from shaderbang.inotify import INotify, IN_CREATE, IN_ATTRIB
from shaderbang.input import Input, TouchSlot
from shaderbang.gesture import homothety_and_rotation
from shaderbang import lib as sb, options

from OpenGL import setPlatform
setPlatform("egl")

from OpenGL.GL import *
from OpenGL.GLU import *


parser = argparse.ArgumentParser(description="Run cloth simulation")
parser.add_argument("-D", "--device", metavar="DEVICE", type=Path,
                    help="the DRM device")
parser.add_argument("-C", "--connector", metavar="CONNECTOR", type=int,
                    help="the DRM connector")
parser.add_argument("--mode", metavar="MODE", type=str,
                    help="the name of the video mode, e.g., 1920x1080")
parser.add_argument("--refresh", metavar="FREQ", type=int,
                    help="the vertical refresh rate in Hz")
parser.add_argument("--async-page-flip", action=argparse.BooleanOptionalAction,
                    help="use async page flipping")
parser.add_argument("--atomic-drm-mode", action=argparse.BooleanOptionalAction,
                    help="use atomic mode setting")
parser.add_argument("--triple-buffer", action=argparse.BooleanOptionalAction,
                    help="use triple buffering (vblank-synced page flips, without blocking on them)")
parser.add_argument("-n", "--frames", metavar="N", type=int,
                    help="run for N frames and exit")

gravity = wp.vec3(0.0, -9.80665, 0.0)

thickness = 0.001
particleRadius = 0.0045
fingerRadius = 0.08     # world radius of a touch/click grab: all cloth particles
                        # within this distance of the picked point are pinned and
                        # dragged together (see Particle.group / drag_anchor)
maxGrabs = 8            # simultaneous grab anchors swept per substep in-graph
maxGrabMembers = 4096   # total pinned patch particles across all grabs
maxVelocity = 1e2   # m/s cap on cloth particle velocity (spike guard)

# Stability / broadphase-safety bounds (see the collision audit).
# maxDisplacement: hard cap on a free particle's per-substep trial displacement
#   |pos - prev_pos|, applied AFTER the XPBD solve and BEFORE self-collision. A
#   legitimate substep travels <= maxVelocity*dt (~0.056) plus small constraint
#   corrections, so 0.2 never fires on a contract-valid substep but bounds the
#   swept query AABB when an instability spike scatters positions.
# maxQueryExtent: if a self-collision query AABB edge exceeds this, skip the query.
#   A governed box tops out around edge_len + 2*maxDisplacement + 2*d_offset ~= 0.44,
#   so 1.0 is a pure no-op in normal operation and only fires on NaN/blown-up boxes,
#   turning a would-be multi-second (O(edges x tris)) frame into a bounded one.
maxDisplacement = 0.2
maxQueryExtent = 1.0

# Planar Divide-and-Truncate (PDT) collision parameters
d_offset = 2.0 * particleRadius  # cloth self-collision separation (matches old 2*particleRadius)
gamma_r = 0.9                    # conservative truncation safety ratio (Newton uses 0.85-0.95)

# Per-substep cap on the accumulated C<0 feasibility-recovery push magnitude. The
# push is an atomic_add over every overlapping pair (order-non-deterministic float
# add) applied AFTER the displacement governor, and it feeds straight back into
# velocity (push/dt). Uncapped, a dense-overlap substep can jump a particle
# arbitrarily far and pump energy, occasionally diverging the sim ("sometimes gets
# stuck"). Capping to a fraction of d_offset bounds the per-substep separation
# (overlaps still relax over several of the 60 substeps) and the injected velocity.
pushClamp = 0.5 * d_offset

# --- Stack-aware grab shell clamp (sliding a folded stack on the sphere) ---
# The anchor-out-of-sphere clamp keeps a grab's target at bare shell contact
# distance (radius + thickness + particleRadius), which is only right when the
# grabbed fabric sits DIRECTLY on the shell. Grabbing the TOP of a folded
# stack lying on the sphere and dragging it across (the user's "penetration
# while sliding with multiple folds") pressed the pinned patch to bare-shell
# distance with 1-3 free layers trapped beneath -- squeezed between two hard
# constraints into transient violation bursts (60-250) and sphere penetration.
# ANCHOR_STACK lifts the clamp each frame by the measured free-stack height in
# the patch's shadow column (90th percentile shell distance + d_offset,
# capped), so the drag slides the top layer OVER the trapped fabric.
anchorStack = int(os.environ.get("ANCHOR_STACK", "1") not in ("0", "", "false"))
anchorStackCap = float(os.environ.get("ANCHOR_STACK_CAP", str(12.0 * d_offset)))
# Per-frame cap on the STACK-driven outward motion of the anchor/member clamp
# (the bare-shell recovery stays geometric -- it must ride an advancing
# shell). A stack detection can appear within one frame (fold flops under the
# patch); an unlimited lift then yanks the pinned patch outward by up to the
# full cap in one frame, plowing it into whatever rests on top (measured as
# occasional 50-90-violation air-pinch bursts + flip spikes with the
# unlimited lift). 2*d_offset/frame still clears a forming stack in 2-6
# frames, faster than the squeeze can lock. The DECAY is slower still: the
# clamp is a min bound the commanded target presses against, so a fast decay
# lets the measurement flicker (percentile of a churning candidate set) drive
# a rise-lag/instant-drop sawtooth around the stack top -- measured as
# sustained 6-9-sample viol 10-40 squeeze streaks. Both are per-frame limits
# on the SMOOTHED LIFT STATE stored on the anchor, not on the target motion.
anchorLiftStep = float(os.environ.get("ANCHOR_LIFT_STEP", str(2.0 * d_offset)))
anchorLiftDecay = float(os.environ.get("ANCHOR_LIFT_DECAY", str(0.5 * d_offset)))
# Stage the FULL member sphere push-out into the frame's pinned offsets (the
# stored conform blend stays half-folded): without it, members lag the shell
# by several frames while the anchor approaches and park INSIDE the sphere.
grabConformFull = int(os.environ.get("GRAB_CONFORM_FULL", "1") not in ("0", "", "false"))

# --- Load-yielding grip ---
# A pinned grab patch is otherwise an infinitely strong actuator: it plows
# through layered fabric regardless of resistance, and sustained forcing beyond
# the pushClamp-capped recovery bandwidth is the dominant entanglement driver
# (once sheets cross, the C<0 push acts on the wrong side and locks the knot).
# The narrowphase C<0 branches record the separation share an inv_mass==0
# (grabbed) vertex WOULD have taken into its push[] slot (sim-inert: apply_
# truncation skips pinned vertices) as a PRESSURE SIGNAL. simulate() sums it
# per grab over the frame's substeps; update_anchors then yields the grip
# under load, one frame later:
#   STALL  -- scale the advance toward the pointer by
#             1 / (1 + grabYieldStallK * mean_member_pressure / d_offset):
#             the grip lags under load and catches up when the load clears
#             (a finger cannot force cloth through cloth).
#   RETREAT - back the anchor off along the net pressure direction by
#             grabYieldGain * (frame pressure sum / members), capped at the
#             per-frame advance limit so the grip goes mushy, never detaches.
grabYieldStallK = float(os.environ.get("GRAB_YIELD_STALL_K", "8.0"))
grabYieldGain = float(os.environ.get("GRAB_YIELD_GAIN", "1.0"))
# STALL FLOOR -- minimum drag authority. The raw stall 1/(1+K*m/d_offset) has
# no lower bound: a settled fold squeezed between the grab and the sphere
# holds a STEADY mean member pressure (~0.5-0.8*d_offset measured on
# cross-flank drags), so the scale parks at ~0.15 indefinitely and the drag
# feels dead (the user's "dragging one side into the other is impossible").
# A real finger keeps moving -- the pile bunches and gives way. Guarantee a
# minimum advance fraction: scale = max(floor, 1/(1+K*m)). Safe because the
# per-frame speed clamp (0.45*d_offset/substep) already caps even full-speed
# plowing just under the fabric's per-substep yield capacity (pushClamp
# 0.5*d_offset); the stall's entanglement protection is about damping
# SUSTAINED forcing, and the floor value below was fuzz-validated (36-session
# all-family sweep + historic knot-seed replays not worse than the unfloored
# baseline).
# Floor raised 0.3 -> 0.4 once the fingertip collider took over guarding the
# actual crossing site (the stall's protective role shrank; the user read the
# friction-loaded stall as "the drag does not get enough force").
grabYieldMinScale = float(os.environ.get("GRAB_YIELD_MIN_SCALE", "0.4"))
# PRESSURE-BUDGET STALL -- stall on the TRANSIENT part of the pressure.
# m_eff = max(0, m_now - beta*ema(m)): a steady parked-contact pressure (the
# freeze regime) is progressively discounted, so a steady plow converges to
# authority 1/(1+K*(1-beta)*m/d_offset) (>= the floor), while a sharp NEW
# spike (a knot forming) still brakes at full strength because the EMA lags
# it. beta=0 disables (raw pressure, historic behavior).
grabYieldSustainBeta = float(os.environ.get("GRAB_YIELD_SUSTAIN_BETA", "0.0"))
# EMA rate per frame for the sustained-pressure tracker (only used when
# beta > 0). Slower rate = longer "burst grace" before a sustained load is
# discounted.
grabYieldSustainEma = float(os.environ.get("GRAB_YIELD_SUSTAIN_EMA", "0.25"))
# Shape of the floored stall: saturating (floor + (1-floor)/(1+K*m), default --
# the scan winner "S30": authority 0.57-0.63 during fold contact with zero
# post-violations) vs a hard clip (max(scale, floor)).
grabYieldSat = os.environ.get("GRAB_YIELD_SAT", "1") not in ("0", "", "false")
# PRE-CONTACT pressure (compile-time kernel constant): also record, in the
# c >= 0 truncation branches, how far a pinned vertex's substep displacement
# would OVERSHOOT the shared separating plane (the share truncation cannot
# deliver to it). This fires while the pair still has a positive gap -- before
# any violation exists -- so the grip yields before sheets cross rather than
# reacting to an already-locked knot.
GRAB_YIELD_PRECONTACT = wp.constant(
    1 if os.environ.get("GRAB_YIELD_PRECONTACT", "1") not in ("0", "", "false") else 0)
# DIRECTION-AWARE yield: pressure MAGNITUDE alone cannot tell extraction
# (pulling a grabbed corner OUT of a pile squeezes the members and reads as
# load, but moves WITH the direction the contacts push them -- relieving)
# from plowing (advancing AGAINST that push, forcing fabric through fabric).
# With this flag on, the advance component ALONG the net member-push
# direction (pvec) bypasses the stall (see update_anchors for the exact
# split). DEFAULT OFF after adversarial validation: pvec's sign is only
# trustworthy BEFORE sheets cross -- once a crossing exists the c<0 recovery
# push acts on the wrong side (see the entanglement-study notes), pvec flips
# INTO the drag direction, and the bypass feeds the locked knot at full
# speed (fuzz seed 3009: 0/4 baseline fails -> 4/4 with the bypass, clean
# again with it off). The magnitude-blind stall is load-bearing exactly
# because it also brakes post-crossing forcing. Kept behind the env knob
# for app-side A/B.
grabYieldDirectional = os.environ.get("GRAB_YIELD_DIRECTIONAL", "0") not in ("0", "", "false")
# EMA factor for the retreat term (1.0 = raw/no smoothing). See update_anchors.
grabYieldRetreatEma = float(os.environ.get("GRAB_YIELD_RETREAT_EMA", "0.3"))
grabYieldDirCoherence = float(os.environ.get("GRAB_YIELD_DIR_COHERENCE", "0.25"))
# Fingertip-tolerant picking: the pixel ray is infinitesimal, so a press on
# the visible EDGE of fabric (the corner of a floor pile -- exactly what a
# user grabs to flatten it) can graze past every triangle by a couple of
# millimeters and return no anchor ("the drag is inoperant"). When the ray
# hits nothing (or only claimed vertices), fall back to the frontmost
# unclaimed particle within fingerRadius of the ray -- what the finger pad
# would actually touch.
grabPickTolerant = os.environ.get("GRAB_PICK_TOLERANT", "1") not in ("0", "", "false")
# Session recorder: CLOTH_RECORD=/path.jsonl makes update_anchors append one
# JSON line per frame with every active anchor's control inputs (screen point,
# the exact pointer ray, committed target, depth) plus the sphere pose --
# everything needed to REPLAY a live app session bit-faithfully in the
# headless harness (cloth_drag_fuzz.py --replay-app FILE) and probe it with
# the exact-crossing instruments. Near-zero overhead when unset.
# Fingertip collider: the grab patch is pinned (inv_mass 0) and therefore a
# HOLE in the collision response -- no projection pass can move it, so fabric
# pinched between the patch and any backing (shell, fold, taut sheet) has one
# escape route: THROUGH the patch. Proven by exact crossing probes on recorded
# sessions: crossings during cross-flank drags concentrate 100% within a patch
# radius of the anchor. Treat each active grab as a small kinematic sphere:
# free vertices are projected out of the fingertip ball every substep
# (velocity-bounded like the collider passes), so fabric flows AROUND the grip
# the way it flows around the ball -- members are inv_mass 0 and unaffected.
FINGER_COLLIDER = wp.constant(
    1 if os.environ.get("FINGER_COLLIDER", "1") not in ("0", "", "false") else 0)
# Default 0.7 (was 1.15) since GRAB_EVADE landed: at 0.7 the ball (0.042 m) is
# SMALLER than the patch itself (fingerRadius 0.06), so it no longer sticks
# out past the patch outline shoving fabric into the visible "bulb" ring --
# it survives as an onset-recovery volume under the patch skirt (fabric
# already inside the grip at grab time, and locked clusters the pair-exact
# evasion cannot undo), while the evasion handles the pre-contact geometry at
# the patch fringe exactly. Measured (12-rep replay-app pools, worst-during
# exact crossings): ball 1.15 alone median 12.5 / max 183; evade + 0.7 ball
# median 12 / max 46 -- the grip-closure burst class (79, 183) disappeared.
# Layered-fold gentle slides: the 1.15 ball's own eviction pumped crossings
# (median 7.5, max 105 exact crossings; 3*projClamp bounded eviction can
# cross a sheet closer than the step); 0.7+evade measured median 0 / max 27,
# with the grip-annulus bulge ridge down 38% (0.043 -> 0.0265).
fingerColliderR = float(os.environ.get("FINGER_COLLIDER_R", "0.7"))  # x fingerRadius
# Eviction speed inside the ball, x projClamp. The grip CLOSES on fabric that
# is already deep inside the volume; at 1x it takes ~20 substeps to clear and
# crossings form in the onset window. Inside the ball everything is being
# co-evicted, so a faster bound is low-risk there.
fingerColliderPush = float(os.environ.get("FINGER_COLLIDER_PUSH", "3.0"))
# Swept CCD for the fingertip ball (always a true sphere, independent of
# COLLIDER_KIND): at the cloth's stretch limit a taut sheet cannot comply with
# the bounded eviction, and the ball dragged into it crosses WITHIN a substep
# -- the user's "keep dragging until it stretches, then it penetrates". The
# analytic time-of-impact catches that regardless of drag speed.
FINGER_CCD = wp.constant(
    1 if os.environ.get("FINGER_CCD", "1") not in ("0", "", "false") else 0)
# CCD radius, x fingerRadius, independent of the volumetric radius above: the
# CCD term only fires on substep trajectories that ENTER the ball from
# outside (fabric already inside is untouched), so it contributes no
# steady-state bulb and can afford more coverage than the eviction volume.
# Kept at the original 1.15 when the volumetric ball shrank to 0.7: the taut
# stretch-limit sheet is exactly the CCD case (it cannot comply with bounded
# eviction, so the grip crosses it within a substep), and shrinking the CCD
# with the ball re-opened it -- stretch_drag worst crossings 27 (ball 1.15)
# -> 101 (everything at 0.7) -> 26 with CCD back at 1.15, post residue 18 -> 0.
fingerCcdR = float(os.environ.get("FINGER_CCD_R", "1.15"))
# Anticipatory evasion push: the narrowphase pre-contact branches (see
# GRAB_YIELD_PRECONTACT, which must be on for this to fire) already compute,
# PER PAIR, exactly how far a pinned member's substep displacement overshoots
# the shared separating plane -- the share truncation cannot deliver to it.
# Today that overshoot is only RECORDED as grip pressure; with this flag the
# FREE side of the pair also receives it as an immediate evasion displacement
# along the separation direction (accumulated into push[], so it rides
# apply_truncation's pushClamp bound and floor/shell invariants): the patch
# pushes fabric out of its own path through the exact pair geometry, the same
# reassignment philosophy as the c<0 lmbd=0 rule. Unlike the fingertip ball
# there is no guard volume, so no "bulb" -- fabric is displaced only by the
# amount the member actually invades, only while it approaches (the overshoot
# is gated on approach speed by construction: resting contacts have ~zero
# displacement and produce ~zero overshoot).
# Consumer choice (push[] vs deltas): push is reset by clamp_displacement,
# written only by the narrowphase, and applied EXACTLY ONCE in
# apply_truncation immediately after the truncation -- the evasion lands in
# the same substep the overshoot occurs, before the collider passes final-say
# and before update_velocity bakes it into vel. deltas written here would sit
# until the first add_deltas INSIDE the collider-edge-pass loop (after
# apply_truncation and collider_project), arriving late and mixed into the
# edge-collider Jacobi step. push also brings the right clamps for free:
# pushClamp (0.5*d_offset) bounds the TOTAL of evasion + c<0 recovery per
# vertex per substep (the recovery-bandwidth bound the entanglement design is
# built on -- and >= the grab speed clamp 0.45*d_offset/substep, so evasion
# can keep pace with the fastest patch), and the floor/shell invariants keep
# an evasion against a backing from being teleported through the ground or
# into the sphere (the inward component is cancelled -> the push turns into
# the lateral escape a plow should produce).
# No double count: a pair is EITHER pre-contact (c >= 0, evasion) or
# overlapping (c < 0, recovery push) in a given substep. Across the two EE
# discovery directions, each thread writes evasion only to its OWN free
# endpoints from the CANDIDATE side's pinned overshoot (the mirrored thread,
# if any, owns the other side; a fully pinned own edge early-outs and its
# free counterpart delivers) -- the same own-endpoint-only convention that
# keeps the c<0 push single-counted.
GRAB_EVADE = wp.constant(
    1 if os.environ.get("GRAB_EVADE", "1") not in ("0", "", "false") else 0)
grabEvadeGain = float(os.environ.get("GRAB_EVADE_GAIN", "1.0"))
# --- Far free-free crossing guard (the valley-plow "burst" class) ---
# Per-substep attribution of the plow-front bursts (free-free pairs at grid
# ring > 3 crossing in a single frame) showed every burst pair WAS in the
# detection caches and ALREADY inside d_offset at the frozen reference
# (c < 0) -- and the c<0 recovery branch skips the truncation plane entirely,
# so nothing constrains motion across the REMAINING gap d: the crossing is
# then completed by whichever bounded mover fires next (measured mix: the
# fingertip eviction step 3*projClamp > gap, the net c<0 push composition in
# a 3+ layer squeeze where opposing-neighbor pushes close the middle pair,
# the collider passes, or a repulsion/pinch delta the skipped plane never
# vetoed). Two cooperating pieces, both env-gated here:
#  * BARRIER PLANE: in the c<0 branches, additionally truncate every free
#    vertex's displacement against the remaining-gap split plane
#    (cp + lmbd*d*n): the pair may stay pressed but the frozen surfaces
#    cannot pass each other via truncated displacement this substep. The
#    recovery push still rides on top exactly as before.
#  * CROSSING BUDGET (push_limit): every narrowphase pair also atomic_min's
#    a per-vertex budget kappa*d (kappa < 0.5, so the two sides' budgets sum
#    below the remaining gap). apply_truncation clamps the recovery push to
#    it (a net push composed across pairs can no longer close the tightest
#    pair) and hands the LEFTOVER to fingertip_project, whose volumetric
#    eviction step is capped by it (an eviction can no longer step across a
#    sheet closer than 3*projClamp). Vertices with no near pair keep an
#    unbounded budget -- onset grip clearance is unaffected. The fingertip
#    CCD branch is deliberately NOT capped (taut-sheet anti-tunnel).
FAR_GUARD = wp.constant(
    1 if os.environ.get("FAR_GUARD", "1") not in ("0", "", "false") else 0)
# Whether the crossing budget also CAPS the c<0 recovery push (it always
# decrements for the fingertip cap). See the apply_truncation note.
FAR_BUDGET_PUSH = wp.constant(
    1 if os.environ.get("FAR_BUDGET_PUSH", "1") not in ("0", "", "false") else 0)
farGuardKappa = float(os.environ.get("FAR_GUARD_KAPPA", "0.45"))
# Barrier floor (x d_offset): the barrier plane is SIDE-BLIND -- for a pair
# that has ALREADY crossed (via a mover outside PDT jurisdiction), the frozen
# "gap" d is the wrong-side depth and the barrier would truncate the RETURN
# motion, locking the crossing in place (measured: a growing, post-release-
# persistent ring-2/3 crossing band in one plow rep -- a failure class the
# baseline never shows; the same side-blindness critique as ring_floor).
# Crossed pairs sit at |gap| ~ 0 while burst creations complete from
# 0.4-1.0 x d_offset (attribution: median 0.72, only ~10% below 0.25), so a
# small floor keeps the prevention and lets crossed pairs slide back out.
farBarrierFloor = float(os.environ.get("FAR_BARRIER_FLOOR", "0.25"))
# --- Crossed-flag guard + crossing brake (400^2 plow-density fix) ---
# At the user's real 400x400 the plow front packs ~2x the layer density of the
# 200x200 tuning scenes: pairs get pressed BELOW the side-blind barrier floor
# above before they cross, and once sub-floor NOTHING constrains them. Exact
# per-substep mover attribution on the recorded 400^2 session (replay-app
# f138-206 with intra-substep phase snapshots, one locked rep, ~26k creation
# events): the post-PDT collider/strain/ring-floor block completes 56% of all
# crossing creations (col/ff/far alone 46%), pre-narrowphase displacement
# (solve + anchor sweep) ~18%, the recovery-push/evade composition riding over
# the planes ~17%, repulsion ~8%, fingertip eviction ~1% -- so the nx=200
# levers (floored push budget, eviction cap) do not even see the dominant
# 400^2 movers. Two cooperating pieces, both grab-gated (grab-free scenes are
# bit-identical, and the sphere/crush machinery is untouched):
#  * FLAG_GUARD: once per frame while a grab is active, run the resolver's
#    EXACT crossing sweep (detect_crossings; no host readback -- a device
#    scatter marks the involved vertices in crossedFlag). The c<0 barrier
#    floor becomes side-AWARE: a pair with NO flagged vertex is genuinely
#    uncrossed, so it gets the barrier all the way down to FAR_BARRIER_EPS
#    (sub-floor gaps protected); a flagged pair keeps the legacy blind floor
#    so return motion stays free (the lock hazard the 0.25 floor existed
#    for). Flags lag one frame: a pair crossing mid-frame is barrier-held for
#    the remaining substeps, then flagged and freed.
#  * GRAB_BRAKE: exact-topology negative feedback on the FORCING. The same
#    per-frame sweep hands update_anchors each grab's near-anchor crossing
#    count; while crossings exist at the plow front, the anchor advance is
#    scaled down by 1/(1 + K*n) (own floor, below the yield stall's authority
#    floor -- an actual crossing is fabric ripping through fabric, braking
#    then is correct, unambiguous, and self-releasing: the count decays as
#    the recovery machinery clears the front, and the brake vanishes with
#    it). Baseline 400^2 dynamics motivate this: crossings self-cleared to 0
#    MID-DRAG whenever formation paused, and every locked ending grew out of
#    a sustained-plow phase where formation outran the pushClamp-bounded
#    recovery -- the mover, not the guards, sets the failure rate at this
#    density.
# MEASURED NEGATIVE at 400^2 (removed; kept as a warning): an end-of-substep
# truncation-only plane re-pass (re-applying the frozen planes to the NET
# displacement after the collider/strain block, to veto the post-PDT movers
# that complete 74% of the crossings) makes things ~3x WORSE (worst 514-518
# vs baseline 106-694 median ~150, flips 20-40 -> 90-116, locked endings):
# planar truncation cancels the WHOLE displacement vector of a grazing
# contact (t ~ 0 at resting gaps), so the re-pass fights the strain-limiter
# and collider convergence work every substep and the pile's geometry
# quality collapses. Do not re-add a net-displacement plane re-pass.
FLAG_GUARD = wp.constant(
    1 if os.environ.get("FLAG_GUARD", "1") not in ("0", "", "false") else 0)
flagGuardEnable = int(os.environ.get("FLAG_GUARD", "1") not in ("0", "", "false"))
farBarrierEps = float(os.environ.get("FAR_BARRIER_EPS", "0.02"))
grabBrakeK = float(os.environ.get("GRAB_BRAKE_K", "1.0"))
grabBrakeFloor = float(os.environ.get("GRAB_BRAKE_FLOOR", "0.05"))
grabBrakeR = float(os.environ.get("GRAB_BRAKE_R", "0.3"))
# Brake hysteresis: the raw count releases the brake the instant a wad
# resolves, and the anchor then rams the still-compressed pile at full
# authority -- measured as a resolve/ram limit cycle with GROWING re-bursts
# (worst 70-115 -> 134-159 when in-drag resolution cleared wads instantly).
# Peak-hold with exponential decay: authority returns over ~15-20 frames
# after the front clears, giving the pile time to decompress.
grabBrakeDecay = float(os.environ.get("GRAB_BRAKE_DECAY", "0.85"))
# Onset pre-arm: initial brake hold at grab creation (see drag_anchor).
# 12 means first-frame authority ~1/(1+K*12*0.85) ~ 0.09, back above 0.5 by
# ~13 frames, fully free by ~20 -- unless the sweep starts reporting crossings,
# in which case the count takes over. Set 0 to disable.
grabBrakeOnset = float(os.environ.get("GRAB_BRAKE_ONSET", "12"))
# In-drag resolution (UNCROSS_DRAG): the quiescent-only resolver leaves every
# crossing formed during a 60-frame drag to accumulate until release -- at
# 400^2 the release-time wad is then 100s of pairs and the cluster vote locks
# ~half the time. The historical reason for quiescent-only (30 substeps of
# plow re-cross whatever one host pass uncrosses, churning the pile) is
# neutralized by the crossing brake: while crossings exist near the grab the
# anchor is throttled to ~grabBrakeFloor, so the local state is quasi-
# quiescent and resolution sticks. Gated on the brake's own signal (last
# sweep found crossings, a grab is active), every UNCROSS_DRAG_EVERY frames.
uncrossDrag = int(os.environ.get("UNCROSS_DRAG", "1") not in ("0", "", "false"))
uncrossDragEvery = int(os.environ.get("UNCROSS_DRAG_EVERY", "2"))
# Staged-patch self-crossing veto (see _patch_self_crossed): valley-plow
# attribution showed the dominant in-drag crossing bursts are the patch RIM
# crossing itself/its skirt (memE1-2 x memF1-3, all ring<=2) -- the per-frame
# conform push-outs scramble the member offsets into a self-crossed shape
# that no runtime mechanism can repair (PDT ring-culls it, members are
# pinned). Veto the staged shape instead.
grabVeto = os.environ.get("GRAB_VETO", "1") not in ("0", "", "false")
clothRecordPath = os.environ.get("CLOTH_RECORD", "")
_clothRecordFile = open(clothRecordPath, "a") if clothRecordPath else None
# Sliding grab: when the committed anchor target chronically lags the
# commanded pointer-ray target (yield stall against a snagged patch, or any
# obstruction), the grip slips over the fabric like a fingertip: release the
# current members, re-grab a fingertip patch one step toward the pointer, and
# continue the stroke seamlessly (see Cloth._slide_grab).
# Default OFF: in app testing the re-grab produced visible artifacts (the
# grip hopping to neighbor vertices reads as a snap); the stack-aware shell
# clamp turned out to be the fix that mattered for folded slides. Set
# GRAB_SLIDE=1 to re-enable for A/B.
grabSlide = os.environ.get("GRAB_SLIDE", "0") not in ("0", "", "false")
grabSlideLag = float(os.environ.get("GRAB_SLIDE_LAG_R", "2.0")) * fingerRadius
grabSlideFrames = int(os.environ.get("GRAB_SLIDE_FRAMES", "4"))
grabSlideStep = float(os.environ.get("GRAB_SLIDE_STEP_R", "1.0")) * fingerRadius
# Ground-following drag target: with the camera above the ground the depth
# rule only ever DECREASED the anchor depth, so dragging fabric that lies ON
# the floor AWAY from the camera left the target hovering at the grab depth
# (far-side floor drags traveled ~1/3 of the commanded distance). While the
# grabbed fabric is at floor level, let the target follow the RECEDING ground
# intersection too (see update_anchors).
grabGroundFollow = os.environ.get("GRAB_GROUND_FOLLOW", "1") not in ("0", "", "false")
grabGroundBand = 5.0 * d_offset  # "at floor level" tolerance for the rule above
# The same bound applied to every HARD projection (sphere pass-2 ejection, the
# edge-vs-sphere passes, the ground snap): an unbounded projection is a teleport
# that can carry fabric THROUGH a neighboring sheet in one substep -- measured as
# entangled knots forming exactly on the sphere's contact shell when dragged
# fabric squeezes resting fabric into the sphere (the collider ejected it back
# out through the dragged sheet; PDT never saw the motion). Velocity-bounded
# projections keep every subsystem's motion visible to the PDT planes; collider
# penetration under squeeze becomes a small bounded transient instead.
projClamp = 0.5 * d_offset

# --- Pile-crush floor invariant + pinch-corridor extrusion ---
# The ground gets exactly ONE velocity-bounded (projClamp) correction per
# substep (in collider_project), while its opponents during a sphere-onto-pile
# crush -- multi-pair repulsion sums, the PDT recovery push, strain/ring-floor
# deltas, the 11 edge-collider passes -- are each separately capped and all
# APPLIED THROUGH add_deltas / apply_truncation with no floor awareness. Under
# an overfull pinch (a ~1.0-radius sphere parked at its floor clamp over a
# 5-layer pile) the bottom layers lose that fight and are expelled BELOW the
# floor (measured: ~1100 particles at y down to -0.044, persisting through the
# whole settle), crossing every layer on the way -- the permanent-entanglement
# highway. The floor is therefore a hard feasibility bound for INTERNAL
# displacement writers: add_deltas and apply_truncation refuse to move a
# particle from above the particle floor (y = thickness) to below it, and
# refuse to push an already-below particle deeper. Only integrate (gravity)
# may dip below, and the ground pass recovers it as before.
#
# Preventing the floor crossing still leaves the pinch wedge overfull (honest
# squeeze violations; every contact normal there is near-vertical, so nothing
# transports fabric sideways). Three cooperating pieces add the missing
# tangential escape, all gated on REAL local stacking pressure (this substep's
# cached vtCount; a lone sheet scores ~0, 2+ pressed layers ~12+):
#  * pinch_extrude -- while the wedge is CLOSING (shell descending, growing,
#    or parked at its floor clamp), free particles in a radial band around
#    the lower shell surface get a bounded horizontal step away from the
#    sphere axis, out of the wedge. Runs BEFORE the truncation narrowphase
#    (via deltas, like the repulsion) so the PDT planes veto any step that
#    would cross another sheet -- fabric stops at fold walls instead of being
#    teleported through them.
#  * collider_project pass 2 / collider_project_edges -- a mostly-DOWNWARD
#    radial ejection of shallowly-penetrating fabric (the band the descending
#    shell "eats" each substep) is redirected to the lateral shell exit at
#    the particle's own height: parallel to the pancaked sheets below, so the
#    ejection cannot cross them. Deep penetrations keep the radial route so
#    sphere non-penetration always converges.
#  * apply_truncation -- the c<0 recovery push may not drive a particle
#    deeper INTO the shell (the wad-vs-collider stalemate that left fabric
#    parked inside the sphere).
extrudeGain = float(os.environ.get("EXTRUDE_GAIN", "0.75"))      # x projClamp per substep
# Gain dose-response on the pile-crush replay (seed 3014, median post_viol of
# 3 runs / persistent sphere-pen fails): 0.5 -> 103/0, 0.75 -> 53/0,
# 1.0 -> ~26/3-of-6. Higher gain evacuates the wedge faster but at 1.0 the
# aggressive lateral transport starts leaving knots PINNED INTO the shell
# (post sphere_pen 0.008-0.018 via the knot's own distance constraints);
# 0.75 is the strongest spen-clean setting. Lowering EXTRUDE_PRESSURE to 8
# was strictly worse (median 236 + a spen fail): extruding 1-2-layer regions
# churns the pile instead of evacuating it.
extrudePressure = int(os.environ.get("EXTRUDE_PRESSURE", "12"))  # min vtCount (~2+ layers pressed;
                                                                 # a lone sheet scores ~0)
extrudeBand = 8.0 * d_offset  # radial engage band above the lower shell: the overfull
                              # stack forms while the wedge is still closing, so
                              # evacuation must start BEFORE the final pinch

# Pre-crossing contact repulsion (the solver-integrated contact-pressure piece
# the PDT paper pairs truncation with; this codebase had truncation + capped
# recovery only). A unilateral spring displacement applied to the CURRENT
# positions of the cached candidate pairs (vtBuf/eeBuf, read-only) right after
# the candidate buffers are built and BEFORE the truncation narrowphase, so the
# PDT planes still veto any repulsion overshoot within the same substep. With
# no contact force, a stack transmits no pressure: a plowed pile cannot move
# its far layers out of the way, so the middle sheets cross and the c<0 push
# then acts on the wrong side and locks the knot. The repulsion engages only
# below repulsionEngage * d_offset (dead zone: the settled pile rests at
# ~d_offset and must not be inflated) and pushes pairs back toward the engage
# distance (continuous at the engage boundary -- no force jump), step-capped
# per pair so a feasible-start pair can never be pushed across a third layer
# in one substep. Pinned targets are skipped (they receive pressure only via
# the existing c<0 push signal). Env overrides allow parameter sweeps without
# editing baked constants (values are inlined at warp codegen).
ringFloorEnable = int(os.environ.get("RING_FLOOR", "1") not in ("0", "", "false"))

# Cloth-cloth contact friction v2 (position-level Coulomb on the cached
# repulsion pairs). Root cause it addresses (proven by an exact edge-triangle
# crossing probe): the "dragged side penetrates through the other side" the
# user sees is NOT a collision failure -- crossings are ZERO throughout those
# gestures -- it is frictionless SLIP-AROUND: the dragged flank cannot grip
# the flank it presses, slides over/around it, and reappears on the far side
# (visually identical to penetration). v2 fixes v1's two rejected flaws:
#  * cost: v1 re-solved closest-point-on-triangle against prev positions
#    (doubled the hot loop); v2 reuses the CURRENT contact's barycentric
#    weights on prev positions -- a few FMAs.
#  * fold-locking: v1 engaged across the whole repulsion zone, so RESTING
#    stacks glued into a truss; v2 grips only PRESSED contacts
#    (gap < frictionEngage*d_offset, well inside the repulsion dead zone) --
#    resting piles stay slippery.
frictionMu = float(os.environ.get("FRICTION_MU", "0.4"))
frictionEngage = float(os.environ.get("FRICTION_ENGAGE", "0.5"))  # x d_offset
# Lower bound of the friction band: pairs pressed DEEPER than this are in
# violation-recovery (possibly crossed -- unsigned gap cannot tell), and
# friction there damps exactly the relative sliding the c<0 recovery push
# needs to UNCROSS them. Post-friction fuzz showed a fatter sphere-family
# residue tail (571-609-class knots in half the reps vs ~0 before); gripping
# only the healthy band returns recovery to frictionless.
frictionFloor = float(os.environ.get("FRICTION_FLOOR", "0.25"))    # x d_offset
repulsionK = float(os.environ.get("REPULSION_K", "1.0"))          # spring gain
repulsionEngage = float(os.environ.get("REPULSION_ENGAGE", "0.8"))  # dead zone, x d_offset
repulsionIters = int(os.environ.get("REPULSION_ITERS", "2"))      # Jacobi passes/substep
repulsionEE = int(os.environ.get("REPULSION_EE", "1"))            # include edge-edge pairs
repulsionCap = 0.5 * d_offset  # per-pair step cap (pushClamp pattern)
# Approach-gated repulsion (REPULSION_APPROACH=1): the full spring fires only
# while a pair's gap is CLOSING this substep (measured on the frozen reference
# positions vs current -- the crossing-risk / plow-pressure regime the
# repulsion earned its entanglement win in); quasi-static or separating pairs
# get repulsionSepGain x the correction instead. Rationale: as a static
# unilateral spring on every cached pair, the repulsion also acts as
# SCAFFOLDING -- a settled fold stack or floor pile becomes a truss of
# near-rigid struts (measured: flatten-scenario drag effectiveness 0.36 with
# the spring vs 0.84 without; folded lobes hang in the air instead of
# slumping). Gating on approach keeps the pressure-transmission behavior
# (approaching layers still repel at full gain) while letting settled stacks
# relax and slide tangentially (tangential slide keeps the gap ~constant, so
# it no longer fights the spring every substep).
REPULSION_APPROACH = wp.constant(
    1 if os.environ.get("REPULSION_APPROACH", "0") not in ("0", "", "false") else 0)
repulsionSepGain = float(os.environ.get("REPULSION_SEP_GAIN", "0.25"))  # non-approach gain
repulsionApproachEps = float(os.environ.get("REPULSION_APPROACH_EPS", "0.02")) * d_offset

# --- Analytic collider kind (compile-time) ---
# COLLIDER_KIND=0 (default): the sphere collider, exactly as before.
# COLLIDER_KIND=1: an INFINITE CYLINDER along the world-z axis through
# (center.x, center.y) with the same radius state -- the "rod" test collider
# (a horizontal rod makes parallel accordion folds trivially reproducible).
# The analytic distance changes from |p - c| - R to |(p.xy) - (c.xy)| - R with
# the normal confined to the xy-plane; z is free. Implemented by routing every
# collider-relative vector through Cloth.radial() (identity for the sphere, xy
# projection for the rod). COLLIDER_KIND is a wp.constant, so the kind-0
# codegen and behavior are unchanged; read from the env at import time like
# the other compile-time knobs (relaunch to switch kind).
# Kind-1 limitations (deliberate, sphere-crush-specific machinery):
#   * pinch_extrude and the lateral pile-crush redirects in collider_project /
#     collider_project_edges are compiled out (they encode sphere geometry;
#     the rod scenarios never park the collider on a floor pile).
#   * Sphere.render still draws a GL sphere (headless harnesses don't render).
COLLIDER_KIND = wp.constant(int(os.environ.get("COLLIDER_KIND", "0")))
colliderKind = int(os.environ.get("COLLIDER_KIND", "0"))  # host-side mirror

# SOLVER-side ring cull radius (grid Chebyshev distance at or below which a
# vertex pair is skipped by the self-collision broadphase). Historically 2
# ("so bending isn't frozen") -- but the 2-ring is a CROSSING BLIND ZONE: the
# persistent post-drag violation knots dissect as exact edge-triangle
# intersections whose partners sit at Chebyshev distance 2-3 (a fold pinched
# to cell scale), where no PDT plane, no repulsion and no side-aware guard
# exists (ring_floor enforces DISTANCE only, so once crossed it stabilizes
# the WRONG side). At 1, ring-2 pairs get the full swept truncation +
# repulsion: a crease is cushioned at d_offset (= the fabric-thickness
# semantics used everywhere else; rest ring-2 distance is 2*spacing = 3.3x
# d_offset, so nothing engages until a fold is nearly razor-sharp), and
# cell-scale fold-throughs can no longer form. Bending is NOT frozen: ring-1
# (the actual hinge neighbors) stays culled. Metric kernels keep the 2-ring.
SOLVER_RING = wp.constant(int(os.environ.get("SOLVER_RING", "1")))

# --- Rim pair solver (grab-patch ring-1 kink) ---
# Valley-plow attribution (exact edge-triangle crossings, classed by member
# involvement and TRUE grid ring): the dominant persistent in-drag crossing
# class is the patch RIM folding through its own free SKIRT at SINGLE-CELL
# scale -- pinned member edges through mixed member/free faces and skirt
# free-free pairs, all at Chebyshev ring 1, traveling WITH the patch for tens
# of frames. Ring-1 pairs are excluded from PDT, repulsion, evasion and the
# uncross resolver BY DESIGN (adjacent vertices legitimately touch; ring-1
# planes at full d_offset would freeze bending), and members are pinned, so
# NO runtime mechanism owns the kink: a skirt vertex that snags while its
# pinned neighbor advances stretches the cell until it passes THROUGH the
# adjacent cell's fabric.
# Fix: an explicit small pair list (vertex-face and edge-edge pairs at ring
# EXACTLY 1, restricted to the grab patch's members + their ring-1 skirt,
# faces/edges within ring 2 of the members) built host-side at grab time and
# solved by two tiny dedicated kernels that mirror the standard PDT
# DIVIDE/TRUNCATE -- exempt from the ring cull because the patch context
# makes these pairs adversarial, unlike ordinary mesh neighbors -- with a
# REDUCED separation rimDOffset = RIM_D_FRAC * d_offset. Rest ring-1 gaps on
# this mesh are >= spacing*sqrt(2)/2 ~ 0.0106 >> 0.0045, and a legitimate
# tight rim fold rests around d_offset, so the reduced offset engages only on
# sub-half-fabric-thickness razor kinks and cannot stiffen the visible drape.
# Pinned members are never truncated and never pushed (their pressure
# bookkeeping is untouched -- the yield/stall feel is unchanged); their
# undeliverable separation shares and plane overshoots are reassigned to the
# free side of the pair, the same reassignment philosophy as GRAB_EVADE and
# the c<0 lmbd=0 rule. All rim pushes ride apply_truncation's pushClamp bound
# and floor/shell invariants.
RIM_SOLVER = wp.constant(
    1 if os.environ.get("RIM_SOLVER", "1") not in ("0", "", "false") else 0)
rimSolverEnable = int(os.environ.get("RIM_SOLVER", "1") not in ("0", "", "false"))
# Separations. EE is the load-bearing one: a rim cell's face has inradius
# (a + b - c)/2 = (0.015 + 0.015 - 0.0212)/2 ~ 0.0044, so an edge crossing
# through the FACE INTERIOR passes ~0.0044 from all three boundary edges --
# an EE offset of 0.5*d_offset (0.0045) is marginally inert exactly there
# (measured: v1 with 0.0045 left the interior channel open and crossings
# tunneled through it, then persisted). 0.75*d_offset gives the interior
# crossing a real ~0.0023 recovery depth. VT stays at 0.5*d_offset (vertex
# paths near a face vertex are sealed by the vertex's own plane).
rimDOffsetVT = float(os.environ.get("RIM_D_FRAC_VT", "0.5")) * d_offset
rimDOffsetEE = float(os.environ.get("RIM_D_FRAC_EE", "0.75")) * d_offset
maxRimVT = 16384   # rim vertex-face pair capacity (a ~90-member patch builds ~3k)
maxRimEE = 32768   # rim edge-edge pair capacity (a ~90-member patch builds ~6k)

# --- Crossing resolver ("uncross") ---
# Every separation mechanism above (PDT truncation planes, the c<0 recovery
# push, contact repulsion) pushes each side of a close pair away from the
# other's surface ON THE SIDE IT CURRENTLY IS. Once two sheets have actually
# CROSSED (fast flick + fold pinch: a post-truncation mover carries a vertex
# through a near-zero gap), that rule is exactly wrong: the intersection
# contour is locked in place forever -- the violating band around the contour
# stays gap~0 while every mechanism maintains the crossing (measured: the
# persistent post-release violation clusters coincide 1:1 with exact
# edge-triangle intersections). The resolver runs ONCE PER FRAME outside the
# captured graph: an exact segment-triangle intersection sweep (hash-grid
# walk over current positions, Moller-Trumbore, min-id dedup) finds truly
# crossed (edge, face) pairs -- crossing is decided by an EXACT test, never a
# heuristic side guess, so a merely compressed contact can never be
# "resolved" into a crossing -- and each crossed edge gets its shallower free
# endpoint moved back across the face plane by (depth + margin), capped per
# frame. That flips the pair to the un-crossed configuration with minimal
# motion; the normal PDT/repulsion machinery then separates it correctly on
# the next substeps, and the contour shrinks frame over frame.
uncrossEnable = int(os.environ.get("UNCROSS", "1") not in ("0", "", "false"))
_UNCROSS_DEBUG = os.environ.get("UNCROSS_DEBUG", "") not in ("", "0")
maxCross = 4096                   # crossed-pair capacity per frame sweep
# Clearance past the face plane after the flip. Note a flip landing at
# 0.25*d_offset sits exactly at the c<0 barrier floor (farBarrierFloor), i.e.
# in the band where no barrier plane protects its return -- but raising the
# landing to 0.5*d_offset measured SLOWER on the recorded pleat core (longer
# flip segments displace more fabric per flip in an already-tight channel),
# so the default stays 0.25; the knob remains for experiments.
uncrossMargin = float(os.environ.get("UNCROSS_MARGIN", "0.25")) * d_offset
uncrossStep = 0.75 * d_offset     # per-vertex per-frame displacement cap
# Hard bound of the DEPTH-COMPLETE flip cap (see _resolve_crossings): a
# vertex's flip may exceed uncrossStep up to this, but only when its own
# deepest crossing NEEDS that much to land past the partner plane. The locked
# recorded 400x400 wad measured flip needs of 0.009-0.014 (depth+margin) vs
# the fixed 0.00675 cap -- capped flips land still-crossed inside the c<0
# barrier band and are re-locked by the substeps (a ~390-pair knot drained at
# ~2 pairs/frame = the user's permanent penetration). 2*d_offset covers the
# deepest measured need with margin; UNCROSS_STEP_MAX=0.75 restores the
# legacy fixed-cap behavior exactly.
uncrossStepMax = float(os.environ.get("UNCROSS_STEP_MAX", "2.0")) * d_offset
# Large-wad recovery burst: when a quiescent sweep finds MORE than
# uncrossBurstN crossed pairs, run up to uncrossBurstIters detect->flip
# rounds in that frame instead of uncrossIters. A big locked wad drains by
# CONTOUR PEELING -- only the ring of vertices currently in crossing pairs
# can flip, so one round per frame retracts one ring (~10-30 pairs) and a
# ~650-pair release wad needs more frames than a user watches (measured:
# FINAL 34 after the recorded session's 47 post-release frames -- still
# draining, just too slowly). Extra WITHIN-frame rounds peel deeper rings:
# each round's flips land past the partner plane (depth-complete cap), so
# the next round's sweep sees the newly-exposed ring, while the per-frame
# flip-lock (moved set) prevents ping-pong and the veto still blocks
# foreign-layer flips. WITHIN a frame the sweep counts still grow (it0 400
# -> it2 588 on the recorded 400x400 wad: the peel front's neighbor edges
# enter crossing as their vertices stay one ring behind) exactly like the
# old capped-flip cascade (366 -> 621) -- the difference is BETWEEN frames:
# depth-complete flips stick (the substeps relax the peeled band instead of
# re-locking it), so the frame-over-frame count collapses 400 -> 406 -> 252
# -> 188 -> 128 -> 84 -> 66 -> 0 (~6 frames to sub-100 vs ~15 single-round).
# Burst rounds only ever run on quiescent frames (the gated path), never
# during interaction. The recorded session releases wads of 600-775 pairs
# with ~47 user-visible frames before the harness FINAL probe; single-round
# frames drain them at ~10-30 pairs/frame (measured: one rep entered the
# window at ~600 and ended at 175, still draining, ~10 frames short), while
# burst frames measured up to ~50 pairs/frame on the same class. On the
# saved settled wads burst=3-with-dilation was within noise of burst=1
# (f44 vs f39 to zero on the 714 state, equal elsewhere), so the burst's
# value is the fresh-release regime, not settled cores.
uncrossBurstN = int(os.environ.get("UNCROSS_BURST_N", "100"))
uncrossBurstIters = int(os.environ.get("UNCROSS_BURST_ITERS", "3"))
# Regional flip dilation: diffuse each flip vector into the flipped vertex's
# same-sheet grid neighborhood (uncrossDilate rings of the 4-neighborhood,
# decayed by uncrossDilateGain per ring, veto-tested like the flips
# themselves). WHY: the residual wad class that resists plain flipping is a
# SELF-PLEAT -- a fold lobe pushed through itself (measured on the recorded
# 400x400 session: 113 pairs, edges and faces in the SAME 22x16-cell window,
# material ring 4-19). Flipping only the crossed ring while the lobe's
# interior stays put leaves the interior tension to yank the ring back
# through within the frame's 30 substeps (measured: ~90% of persisting pairs
# had a flipped endpoint -- flips applied, then undone; net drain ~2/frame).
# Dragging the 1-2 ring neighborhood along with each flip moves the lobe
# BODILY, so the flip sticks and the next ring enters the (exact) sweep on
# the following round/frame. UNCROSS_DILATE=0 disables.
uncrossDilate = int(os.environ.get("UNCROSS_DILATE", "2"))
uncrossDilateGain = float(os.environ.get("UNCROSS_DILATE_GAIN", "0.6"))
# Settle-interleaved recovery: when a quiescent frame's first sweep finds a
# large wad (> uncrossBurstN), split the frame's substep replays into
# uncrossInterleave chunks and run a full resolver pass between chunks --
# resolve / settle 10 substeps / resolve / settle / resolve / settle instead
# of one pass per frame. Unlike within-frame burst rounds (which re-sweep
# UNRELAXED positions and mostly re-see their own peel front), each
# interleaved pass acts on constraint-relaxed geometry, so it multiplies the
# frame-over-frame drain rate by ~the chunk count. This is what makes the
# recorded session's deepest release wads (~800 pairs, draining ~10/frame
# single-pass = ~50 frames short of the user's attention span) clear inside
# the post-release window. Costs ~2 extra host sweeps (~10-15 ms) per
# RECOVERY frame only; clean and small-wad frames are untouched.
# UNCROSS_INTERLEAVE=1 disables. Very large wads (> uncrossInterleaveBigN
# pairs -- fresh releases still being fed by the collapsing drape) escalate
# to uncrossInterleaveBig passes: the recorded session's ~850-pair worst
# releases drained ~15 pairs/frame at 3 passes (ended at 150 of the ~47
# post-release frames the user actually watches) and need ~25/frame.
uncrossInterleave = int(os.environ.get("UNCROSS_INTERLEAVE", "3"))
uncrossInterleaveBig = int(os.environ.get("UNCROSS_INTERLEAVE_BIG", "5"))
uncrossInterleaveBigN = int(os.environ.get("UNCROSS_INTERLEAVE_BIG_N", "300"))
# detect->flip rounds per frame. 1 is deliberate: repeated rounds with the
# flipped-vertex lock CASCADE in a band crossing -- flipping v resolves its
# pair but puts v's edges to its unflipped neighbors in crossing, and the
# lock then flips those neighbors too, so the flipped region GROWS each round
# (measured: within-frame counts 366 -> 621 and never converging). One round
# per frame lets the constraints/bending/ring-floor react to each flip, and
# the contour shrinks frame over frame instead.
uncrossIters = int(os.environ.get("UNCROSS_ITERS", "1"))
# Cadence under active forcing (grab held / sphere driven). Quiescent frames
# resolve at full rate (with idle backoff); while the user is actively
# forcing, the resolver stays OFF by default: resolving mid-plow FIGHTS the
# forcing (30 substeps re-cross what one host pass uncrossed) and its flips
# in a hard pinch occasionally trigger a strain blow-up (measured: 0.43-0.49
# sphere-metric transients + 125-viol locked residue ONLY in
# resolve-during-forcing configurations; never in quiescent-only). With the
# SOLVER_RING=1 prevention the in-drag crossing counts stay tiny (4-27 pairs
# vs ~500 before), so post-release cleanup is enough. uncrossForcedEvery can
# re-enable throttled in-drag pruning (every K-th frame, only while the knot
# is <= uncrossForcedMaxN pairs) for experiments; UNCROSS_GATED=0 = resolve
# every frame regardless (legacy always-on).
uncrossGated = int(os.environ.get("UNCROSS_GATED", "1") not in ("0", "", "false"))
# MASKED in-drag resolution (built for the round-10 lock fix, measured, and
# left DEFAULT-OFF): the recorded 400x400 session's crossing wad forms
# MID-DRAG at the plow front and then trails the moving anchor
# (near_anchor=0 at radius 0.2), so sweeps with the forcing sites masked out
# (crossed pairs within uncrossMaskR of any active grab anchor, or of the
# sphere shell while it is driven, are skipped) looked like the way to prune
# the wad before release. MEASURED VERDICT on the recorded session (7 reps,
# every-2-frames masked sweeps): pruning LOSES to the plow's creation rate
# -- the wad grew 96 -> 745 pairs THROUGH active pruning (apply ~1300/frame)
# because the "trailing" fabric is still the towed sheet, not quiescent
# cloth -- and the churned release state it left behind stalled the
# post-release drain (one rep ended at 714, worse than the no-pruning
# ceiling; in-drag flips metric 26-41 vs ~10). Meanwhile the post-release
# stack (depth-complete flips + regional dilation) clears even 700-pair
# release wads in < 45 frames on its own. UNCROSS_FORCED_EVERY > 0 re-arms
# the masked cadence for experiments.
uncrossForcedEvery = int(os.environ.get("UNCROSS_FORCED_EVERY", "0"))  # 0 = never
uncrossForcedMaxN = int(os.environ.get("UNCROSS_FORCED_MAXN", str(maxCross)))
uncrossMaskR = float(os.environ.get("UNCROSS_MASK_R", "0.25"))
# Flip-side policy: "cluster" = union-find the crossing edges into contour
# clusters and flip the coherent minority/least-depth side per cluster;
# "pair" = independent per-pair least-motion (can pick incoherent directions
# along a band).
uncrossVote = os.environ.get("UNCROSS_VOTE", "cluster")
# Vectorized no-new-crossing veto (see _resolve_crossings / _veto_flips_vec);
# UNCROSS_VETO_CHECK=1 cross-checks it against the reference loop every round.
uncrossVetoVec = int(os.environ.get("UNCROSS_VETO_VEC", "1") not in ("0", "", "false"))
uncrossVetoCheck = int(os.environ.get("UNCROSS_VETO_CHECK", "0") not in ("0", "", "false"))

# Self-collision is split into a hash-grid "detect" pass that caches candidate
# primitives and a query-free "narrowphase" pass that does the segment-segment /
# point-triangle truncation. The detect pass used to query warp's LBVH
# (mesh_query_aabb), whose fixed 32 KiB per-block shared-memory traversal stack
# capped occupancy at ~50% and pointer-chased the tree -- 63% of the substep. It
# now queries a wp.HashGrid over the frozen start-of-substep POINTS (built
# in-graph each substep; a build is ~0.07 ms vs ~1.2 ms of BVH queries) and
# expands each neighbor point into its precomputed incident faces (vertex-
# triangle) or incident edges (edge-edge). Query radii are derived per primitive
# from the exact narrowphase no-op criterion (gap <= d_offset + own sweep + max
# sweep), plus a Jung-type bound (longest-edge/sqrt(3), resp. /2) that converts
# "triangle/segment within range" into "some VERTEX within range"; the global
# longest-edge and max-sweep bounds are recomputed on the GPU every substep, so
# stretched cloth or fast motion enlarge the radii instead of missing candidates.
# vtBuf caches candidate FACE ids (dedup'd, 2-ring culled -- identical semantics
# to the old BVH cull); eeBuf now caches candidate EDGE ids directly (shared-
# vertex + ring culls applied at detect time), which removes the face->3-edge
# expansion and canonicalization from the hot edge-edge narrowphase. The
# `selfCollisionOverflow` counter still flags any capacity drop (which would
# allow penetration).
# The grid detect's acceptance reach (~d_offset + longest_edge slack, see
# detect_expand) is ~2x the old BVH boxes' +/-d_offset in the contact-normal
# direction, so compressed stacks (a sphere squeezing the draped cloth) yield
# 2-3x the old candidate counts. Peaks are hit MID-frame (worst substep), well
# above the end-of-frame counts test_counts.py reports.
maxVT = 256   # cached candidate faces per vertex (non-ring, dedup'd)
maxEE = 768   # cached candidate edges per edge (culled, dedup'd)

# Frozen-edge length cap for the vertex-centered broadphase walk radii (see
# detect_walk_radius). The radii used to add the GLOBAL longest frozen edge
# (bounds[0]) so stretched cloth widened the search -- but that lets ONE
# locally stretched edge (a fast sphere drag reaches 0.3-2.0 m) inflate EVERY
# vertex's walk radius; combined with multi-layer squeeze states this
# saturated the fixed candidate buffers (silent drops = potential tunneling).
# The radii now use L = min(bounds[0], edgeLenCap), which bounds every walk:
# primitives whose frozen edges all fit under the cap are covered by the
# capped walk exactly as before (the at-rest longest edge is the 0.0212 m quad
# diagonal, far below the cap, so rest behavior is unchanged), while OVERSIZED
# primitives (any frozen edge > edgeLenCap) are partitioned EXCLUSIVELY to a
# fallback: detect_walk_radius/detect_expand skip them (same `> edgeLenCap`
# comparison everywhere, so no pair is ever processed twice -- the C<0 push is
# not idempotent) and scatter_oversized / oversized_pairs append them into
# the same candidate buffers (see those kernels). 4x the 0.015 rest spacing.
edgeLenCap = 4.0 * 0.015

# Strain limiting: hard cap on edge elongation, enforced by a post-solve Jacobi
# projection (strain_limit kernel, strainLimitIters passes/substep). Real cloth
# stretches < ~10%; without a hard cap, dragging the sphere through a wrapped
# drape overwhelms the 2-iteration XPBD solve and edges stretch 25-100x rest --
# which (a) looks like the cloth "diverging", (b) explodes the self-collision
# contact density (the perf cliff: >1s frames + millions of dropped candidates
# in stretch wads), and (c) is what created oversized primitives in the first
# place. With strain capped at 1.2x, the longest possible edge is
# 1.2 * 0.0212 (rest diagonal) = 0.0255 << edgeLenCap, so the oversized
# fallback becomes a never-firing safety net and detect density stays bounded
# under any drag speed. The limiter engages ONLY beyond maxStrain (the XPBD
# solve keeps normal strain ~1-2%), so settled/draping behavior is unchanged.
maxStrain = 1.2          # max edge length as a multiple of rest length
minStrain = 0.5          # compression floor (settled drapes stay >= ~0.92x rest, so
                         # this is inert normally; crushing collapsed edges to ~0.1x
                         # rest = degenerate "inverted" triangles + contact-density
                         # blowup until this floor forces buckling instead)
strainLimitIters = 8     # Jacobi passes after the solve (pre-collision)
strainLimitPostIters = 4 # Jacobi passes after the collider, before update_velocity:
                         # the collider's nearest-surface ejection can split an edge
                         # across the sphere within one substep; limiting again before
                         # velocities are derived keeps that stretch out of vel (else
                         # (pos-prev)/dt bakes it in and integrate re-creates it).
strainRelax = 0.6        # under-relaxation (shared vertices, up to 8 edges each)

# Hash-grid cell width for the self-collision broadphase. Queries may use any
# radius (the grid walks more cells); the cell size just tunes performance. The
# quiet-state walk radius is ~0.022 (max of the vertex-triangle reach
# d_offset + rest_diag/sqrt(3) and the edge-edge reach
# sqrt(halflen^2 + (d_offset + rest_diag/2)^2)), so 0.024 keeps the walk at
# 3^3 cells with the tightest cell-granular over-return (validated by the
# broadphase microbench on the real crumpled state).
gridCellSize = 0.024

# Thread count for the single-slot detection-bounds reductions (grid-stride loops).
boundsReduceThreads = 16384

# FAR_DEBUG=1: snapshot pos at intra-substep pipeline boundaries into debug
# arrays (post-detect, pre-narrowphase, post-truncation, pre-fingertip) so a
# host probe stepping between graph replays can attribute exactly WHERE in the
# substep a self-crossing was created. Diagnostic only; default off.
farDebug = os.environ.get("FAR_DEBUG", "0") not in ("0", "", "false")

# Cached non-ring neighbor vertices per vertex (detect_gather -> detect_expand).
# Crumpled-plateau counts are ~20-30; sized for a compressed squeeze; the
# selfCollisionOverflow counter flags drops. Capacities are sized ~1.5-2x the
# worst MID-substep peaks measured on the 200x200 rising-sphere squeeze with
# the capped radii (nbr ~210, vt ~52, ee ~490): the squeeze density is real
# (multi-layer stack x fast ejection sweeps), the caps just need to clear it.
# Memory at 400x400: nbr 247 MB + vt 165 MB + ee 1.48 GB ~= 1.9 GB.
maxNbr = 384

# Fixed-size per-thread caches for detect_expand: the per-vertex incident-edge
# data is loop-invariant across the cached neighbors (high-water ~15 per vertex
# at a deep-settled 400x400 pile), and a neighbor's incident-edge data is
# invariant across the (up to 6) own edges it is tested against. Hoisting both into registers/local arrays removes the
# redundant global gathers that dominated the kernel. Grid-mesh vertex degree is
# <= 6 (faces and edges), asserted at build time.
expandDeg = 6
vec6i = wp.types.vector(length=expandDeg, dtype=wp.int32)
vec6f = wp.types.vector(length=expandDeg, dtype=wp.float32)
mat6x3f = wp.types.matrix(shape=(expandDeg, 3), dtype=wp.float32)
# detect_expand parallelizes over (vertex, neighbor slot): one thread per
# vertex left the whole ~(neighbors x incident) expansion as ONE serial
# dependent-load chain per thread, and 160K threads is barely a single wave on
# a large GPU -- no latency hiding and the pile's slowest vertices bound the
# kernel. expandK threads per vertex stride the cached neighbor list instead
# (the gather high-water at a deep-settled 400x400 pile is ~15, so each thread
# usually owns at most one neighbor).
expandK = 16

numIterations = 2
numSubsteps = 30  # ~2x frame rate vs 60. The "visible undulation" 30 used to cause was
                  # root-caused to (a) the bending constraint's per-substep impulse
                  # crossing its explicit-regime stability boundary at dt > timeStep/60
                  # (see the bending stability guard in step()) and (b) the XPBD damping
                  # term dotting gradients with absolute previous POSITIONS (a spurious
                  # dt-dependent bias, fixed in distance/bending_constraints). With both
                  # fixes a settled drape's frame-to-frame motion at 30 substeps matches
                  # 60 (~2e-4 m mean vs ~2e-2 unfixed); 60-substep behavior is unchanged
                  # (the guard is exactly 1.0 there).
timeStep = 1.0 / 30.0
epsilon = sys.float_info.epsilon

# Reference substep size the bending stability guard is calibrated to (the substep
# count the simulation was tuned and visually validated at). See step(): for
# dt <= bendStabilityDt the guard is exactly 1.0 (a no-op); only LARGER substeps
# (fewer than 60 substeps/frame) scale the bending relaxation down to keep the
# per-substep bending impulse inside the overshoot boundary observed at 60.
bendStabilityDt = timeStep / 60.0

wp.init()
wp.set_device("cuda")


class State(Flag):
    RUN = auto()
    STEP = auto()
    FRAME_STEP = auto()
    SMALL_STEP = auto()
    CONTACT_STEP = auto()
    SOLVER_STEP = auto()
    SELF_COLLISION = auto()
    CULL_FACE = auto()
    WIREFRAME = auto()


STEPS = State.FRAME_STEP | State.SMALL_STEP | State.CONTACT_STEP | State.SOLVER_STEP

state = State.RUN | State.FRAME_STEP | State.SELF_COLLISION | State.WIREFRAME


class _Perf:
    # Rolling frame-time buckets for the [perf] line (see Cloth.render).
    PERIOD = 120
    frames = 0
    t_last = None
    total = 0.0
    sim = 0.0
    draw = 0.0


_perf = _Perf()


class AnchorFlag(Flag):
    ACTIVE = auto()
    SELECTED = auto()
    LOCKED = auto()

@dataclass
class Particle:
    id: int
    screen: wp.vec2
    mass: float
    depth: float
    flags = AnchorFlag.ACTIVE
    origin: wp.vec2 = field(init=False)
    time: float = field(init=False)
    # Fingertip grab group: (particle id, original inv-mass, world offset from the
    # primary particle at grab time). All members are pinned and move rigidly with
    # the primary -- a single-vertex pin transmits absurd force through one thread
    # of fabric (and, being inv_mass==0, ignores collision), so grabbing a
    # fingertip-sized patch both feels natural and distributes the pull.
    group: list = field(default_factory=list)
    # Last frame's committed anchor point (per-substep sweep origin); None until
    # the first active update.
    prev_target = None
    # Sliding-grab state: consecutive frames of chronic pointer lag, and the
    # last frame's lag (a shrinking lag = the anchor is catching up after a
    # flick, NOT snagged -- see update_anchors).
    slide_frames = 0
    last_lag = 0.0

    def drag(self) -> Self:
        self.origin = wp.vec2(self.screen)
        self.time = time.time()
        return self

    def click(self) -> bool:
        return wp.length(self.screen - self.origin) < 1.0 and time.time() - self.time < 0.5

    def drop(self):
        self.flags &= ~(AnchorFlag.ACTIVE | AnchorFlag.LOCKED)

class Cloth(Input):

    def __init__(self, y_offset, num_x, num_y, spacing):
        super().__init__("cloth")

        # TODO: change for size
        self.spacing = spacing
        self.anchors: list[Particle] = []
        self._quad = None

        if num_x % 2 == 1:
            num_x = num_x + 1
        if num_y % 2 == 1:
            num_y = num_y + 1

        self.numParticles = (num_x + 1) * (num_y + 1)
        pos = np.zeros((self.numParticles, 3))
        inv_mass = np.zeros(self.numParticles)

        for xi in range(num_x + 1):
            for yi in range(num_y + 1):
                i = xi * (num_y + 1) + yi
                pos[i, 0] = (-num_x * 0.5 + xi) * spacing
                pos[i, 1] = y_offset
                pos[i, 2] = (-num_y * 0.5 + yi) * spacing
                inv_mass[i] = 1.0

        # distance constraints
        one_x = lambda xi, yi: (xi * (num_y + 1) + yi, (xi + 1) * (num_y + 1) + yi)
        one_y = lambda xi, yi: (xi * (num_y + 1) + yi, xi * (num_y + 1) + yi + 1)
        two_x = lambda xi, yi: (xi * (num_y + 1) + yi, (xi + 2) * (num_y + 1) + yi)
        two_y = lambda xi, yi: (xi * (num_y + 1) + yi, xi * (num_y + 1) + yi + 2)
        cross = lambda xi, yi: [(xi * (num_y + 1) + yi, (xi + 1) * (num_y + 1) + yi + 1),
                                ((xi + 1) * (num_y + 1) + yi, xi * (num_y + 1) + yi + 1)]

        self.distConstraints = DistConstraints(
            Constraint(
                (range(num_x + 1), range(0, num_y, 2), one_y),
                (range(num_x + 1), range(1, num_y, 2), one_y),
                (range(0, num_x, 2), range(num_y + 1), one_x),
                (range(1, num_x, 2), range(num_y + 1), one_x),
                parallel=False,
                ke=1.0e9,
                kd=10.0,
            ),
            Constraint(
                (range(0, num_x, 2), range(0, num_y, 2), cross),
                (range(0, num_x, 2), range(1, num_y, 2), cross),
                (range(1, num_x, 2), range(0, num_y, 2), cross),
                (range(1, num_x, 2), range(1, num_y, 2), cross),
                parallel=False,
                ke=1.0e7,
                kd=10.0,
            ),
            Constraint(
                (range(num_x + 1), range(0, num_y - 1, 3), two_y),
                (range(num_x + 1), range(1, num_y - 1, 3), two_y),
                (range(num_x + 1), range(2, num_y - 1, 3), two_y),
                (range(0, num_x - 1, 3), range(num_y + 1), two_x),
                (range(1, num_x - 1, 3), range(num_y + 1), two_x),
                (range(2, num_x - 1, 3), range(num_y + 1), two_x),
                parallel=True,
                relaxation=0.6,
                ke=1.0e7,
                kd=10.0,
            ),
        )

        # bending constraints
        square = lambda xi, yi: (xi * (num_y + 1) + yi + 1, (xi + 1) * (num_y + 1) + yi,
                                 xi * (num_y + 1) + yi, (xi + 1) * (num_y + 1) + yi + 1)
        diam_x = lambda xi, yi: (xi * (num_y + 1) + yi, (xi + 2) * (num_y + 1) + yi + 1,
                                 (xi + 1) * (num_y + 1) + yi, (xi + 1) * (num_y + 1) + yi + 1)
        diam_y = lambda xi, yi: (xi * (num_y + 1) + yi, (xi + 1) * (num_y + 1) + yi + 2,
                                 (xi + 1) * (num_y + 1) + yi + 1, xi * (num_y + 1) + yi + 1)

        self.bendConstraints = BendConstraints(
            Constraint(
                (range(num_x), range(num_y), square),
                (range(num_x), range(num_y - 1), diam_y),
                (range(num_x - 1), range(num_y), diam_x),
                parallel=True,
                relaxation=0.6,
                ke=1.0e5,
                kd=10.0,
            ),
        )

        self.constraints = Constraints.Chain(self.distConstraints, self.bendConstraints)

        # triangles
        self.numTris = 2 * num_x * num_y
        self.triDist = wp.zeros(self.numTris, dtype=float)
        self.hostTriIds = np.zeros((self.numTris, 3), dtype=np.int32)

        i = 0
        for xi in range(num_x):
            for yi in range(num_y):
                id0 = xi * (num_y + 1) + yi
                id1 = (xi + 1) * (num_y + 1) + yi
                id2 = (xi + 1) * (num_y + 1) + yi + 1
                id3 = xi * (num_y + 1) + yi + 1
                self.hostTriIds[i, 0] = id0
                self.hostTriIds[i, 1] = id1
                self.hostTriIds[i, 2] = id2
                i += 1
                self.hostTriIds[i, 0] = id0
                self.hostTriIds[i, 1] = id2
                self.hostTriIds[i, 2] = id3
                i += 1

        self.prevPos = wp.array(pos, dtype=wp.vec3)
        self.restPos = wp.clone(self.prevPos)
        self.invMass = wp.array(inv_mass, dtype=float)
        self.vel = wp.zeros_like(self.restPos)
        self.deltas = wp.zeros_like(self.restPos)

        self.hostInvMass = wp.array(inv_mass, dtype=float, device="cpu", copy=False, pinned=True)
        self.hostPos = wp.array(pos, dtype=wp.vec3, device="cpu", copy=False, pinned=True)
        self.hostTriDist = wp.zeros(self.numTris, dtype=float, device="cpu", pinned=True)

        self.pos = None
        self.pos_gl_buffer = GLuint()
        self.normals = None
        self.normals_gl_buffer = GLuint()
        self.triIds = None
        self.triIds_gl_buffer = GLuint()

        self.numCols = num_y + 1  # grid stride, for topological-neighbor exclusion
        self.truncation_ts = wp.zeros(self.numParticles, dtype=float)  # per-vertex PDT scale (atomic_min)
        self.push = wp.zeros_like(self.restPos)  # C<0 feasibility-recovery separation (atomic_add)
        # Click-time picking scratch (brute-force ray cast; no BVH in the sim).
        self._pickDist = wp.zeros(1, dtype=float)
        self._pickFace = wp.zeros(1, dtype=wp.int32)

        # Per-vertex grid (row, col) precomputed once so the self-collision ring cull
        # is a couple of int loads instead of 4 integer divisions per call. within_ring
        # is invoked up to 4x per candidate in the hottest (edge-edge) kernel, so on the
        # ring-cull-bound crumpled plateau this removes the dominant integer-div traffic.
        idx = np.arange(self.numParticles, dtype=np.int32)
        grid_rc = np.stack((idx // self.numCols, idx % self.numCols), axis=1)
        self.gridRC = wp.array(grid_rc, dtype=wp.int32)  # [numParticles, 2] = (row, col)


        # Sphere collider pose as single-element device arrays so the captured graph
        # can ADVANCE the sphere per substep (advance_sphere increments them in-graph):
        # each substep sweeps center[0]..center[0]+dc/numSubsteps, so the swept test
        # sees the true per-frame sphere velocity AND the last substep lands on the
        # rendered end pose (center+dc). Set from the module `sphere` each frame in
        # simulate(); the graph records the pointers, not the values.
        self.colliderCenter = wp.zeros(1, dtype=wp.vec3)
        self.colliderRadius = wp.zeros(1, dtype=float)

        # Per-SUBSTEP collider delta slices (frame delta / numSubsteps), also as
        # device arrays. They used to be baked into the captured graph as kernel
        # constants, forcing a RECAPTURE every frame; as device memory the graph
        # is frame-invariant, so simulate() captures once per flag combination and
        # replays it across frames (the in-graph hash-grid build makes per-frame
        # re-instantiation prohibitively slow: its mempool alloc nodes cost ~0.5 s
        # per instantiation).
        self.colliderDeltaC = wp.zeros(1, dtype=wp.vec3)
        self.colliderDeltaR = wp.zeros(1, dtype=float)
        self.colliderDeltaQ = wp.zeros(1, dtype=wp.quat)
        self._graphs = {}  # (iterations, integrate, self_collision, solve) -> captured graph

        # Grab-anchor sweep state: dragged (pinned) patches must move PER SUBSTEP
        # inside the captured graph, exactly like the sphere -- a host-side
        # once-per-frame teleport is invisible to collision (no sweep), so a drag
        # of a few cm/frame lands the frozen reference state already inside other
        # fabric and the PDT can neither prevent nor recover the interpenetration
        # (measured: dragging one flank across into the other collapsed the self-
        # collision gap to ~1e-5 with persistent violations and rising frame
        # times). simulate() packs these from self.activeGrabs each frame:
        # members are (particle id, owning grab index, world offset); anchorPos
        # holds each grab's CURRENT substep anchor point, advanced by anchorDelta
        # (frame motion / steps) once per substep.
        self.activeGrabs = []                     # [(prev vec3, target vec3, ids, offs)]
        self.anchorMemberIds = wp.zeros(maxGrabMembers, dtype=wp.int32)
        self.anchorMemberAx = wp.zeros(maxGrabMembers, dtype=wp.int32)
        self.anchorMemberOff = wp.zeros(maxGrabMembers, dtype=wp.vec3)
        self.anchorMemberCount = wp.zeros(1, dtype=wp.int32)
        self.anchorPos = wp.zeros(maxGrabs, dtype=wp.vec3)
        self.anchorDelta = wp.zeros(maxGrabs, dtype=wp.vec3)
        self.anchorCount = wp.zeros(1, dtype=wp.int32)
        # Load-yielding grip: per-grab plow-pressure accumulators (net vector and
        # magnitude sum), filled in-graph each substep by accumulate_grab_pressure
        # from the pressure shares the narrowphase records on grabbed members'
        # push[] slots; zeroed per frame in simulate(), read back after the frame
        # for the host-side grip yield in update_anchors (one frame of latency).
        self.grabPressure = wp.zeros(maxGrabs, dtype=wp.vec3)
        self.grabPressureMag = wp.zeros(maxGrabs, dtype=float)
        self.grabPressureHost = {}  # primary particle id -> (net vec3, |.| sum, n members)
        # Rim pair solver (see RIM_SOLVER): explicit ring-1 pair lists for the
        # active grab patches' rims, rebuilt host-side when grab membership
        # changes (grab / release / sliding re-grab), zero-count otherwise.
        # Allocated up front so the captured graph can reference them.
        self.rimVTPairs = wp.zeros((maxRimVT, 4), dtype=wp.int32)  # (v, i0, i1, i2)
        self.rimVTCount = wp.zeros(1, dtype=wp.int32)
        self.rimEEPairs = wp.zeros((maxRimEE, 4), dtype=wp.int32)  # (va, vb, vc, vd)
        self.rimEECount = wp.zeros(1, dtype=wp.int32)
        self._rimSig = None          # active-grab membership signature
        self._rimDirty = False
        self._rimStage = (np.zeros((0, 4), np.int32), np.zeros((0, 4), np.int32))

        # Self-collision candidate caches (detect writes, narrowphase reads) + a global
        # overflow counter (>0 means a per-primitive buffer filled and dropped a
        # candidate -> possible penetration; grow maxVT/maxEE).
        self.vtCount = wp.zeros(self.numParticles, dtype=wp.int32)
        self.vtBuf = wp.zeros(self.numParticles * maxVT, dtype=wp.int32)
        self.selfCollisionOverflow = wp.zeros(1, dtype=wp.int32)
        # Per-vertex crossing budget (see FAR_GUARD): reset each substep by
        # clamp_displacement, tightened by the narrowphase (atomic_min kappa*d
        # per pair), consumed by apply_truncation's push clamp, leftover read
        # by fingertip_project's eviction cap.
        self.pushLimit = wp.zeros(self.numParticles, dtype=float)
        # Per-vertex exact-crossing flags (see FLAG_GUARD): refreshed once per
        # frame by simulate() while a grab is active; read by the c<0 barrier
        # gating in the narrowphase (n_grabs-gated, so a stale buffer is
        # never consulted without a grab).
        self.crossedFlag = wp.zeros(self.numParticles, dtype=wp.int32)
        if farDebug:
            # Intra-substep pos snapshots (see farDebug): post-detect,
            # pre-narrowphase, post-truncation, pre-fingertip.
            self.dbgPosDet = wp.zeros(self.numParticles, dtype=wp.vec3)
            self.dbgPosNar = wp.zeros(self.numParticles, dtype=wp.vec3)
            self.dbgPosPDT = wp.zeros(self.numParticles, dtype=wp.vec3)
            self.dbgPosCol = wp.zeros(self.numParticles, dtype=wp.vec3)

        # Unique mesh edges (sorted vertex pairs) for edge-edge self-collision.
        edge_set = set()
        for tri in self.hostTriIds:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for u, w in ((a, b), (b, c), (c, a)):
                edge_set.add((u, w) if u < w else (w, u))
        edge_arr = np.array(sorted(edge_set), dtype=np.int32)
        self.numEdges = len(edge_arr)
        self.edgeIds = wp.array(edge_arr, dtype=wp.int32)  # [numEdges, 2], va < vb
        self.eeCount = wp.zeros(self.numEdges, dtype=wp.int32)
        self.eeBuf = wp.zeros(self.numEdges * maxEE, dtype=wp.int32)
        # Rest length per unique mesh edge, for the strain limiter.
        rest_len = np.linalg.norm(pos[edge_arr[:, 1]] - pos[edge_arr[:, 0]], axis=1)
        self.edgeRestLen = wp.array(rest_len.astype(np.float32), dtype=float)

        # Static pair list for the ring_floor kernel: every vertex pair within
        # Chebyshev ring distance <= 2 on the grid (the self-collision blind zone),
        # EXCLUDING actual mesh edges (governed by the strain limiter's compression
        # floor instead). Built vectorized: for each of the 12 canonical (dr, dc)
        # offsets covering the 5x5 neighborhood half-plane, pair every vertex with
        # its offset neighbor.
        rows = idx // self.numCols
        cols = idx % self.numCols
        nrows = num_x + 1
        pair_chunks = []
        for dr in range(0, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc <= 0:
                    continue  # half-plane: each unordered pair once
                ok = (rows + dr < nrows) & (cols + dc >= 0) & (cols + dc < self.numCols)
                a = idx[ok]
                b = a + dr * self.numCols + dc
                pair_chunks.append(np.stack((a, b), axis=1))
        ring_pairs = np.concatenate(pair_chunks, axis=0)
        # exclude mesh edges via a sorted-key set difference (vectorized hashing)
        key = ring_pairs.min(axis=1).astype(np.int64) * self.numParticles + ring_pairs.max(axis=1)
        ekey = edge_arr.min(axis=1).astype(np.int64) * self.numParticles + edge_arr.max(axis=1)
        keep = ~np.isin(key, ekey)
        ring_pairs = ring_pairs[keep]
        self.numRingPairs = len(ring_pairs)
        self.ringPairs = wp.array(ring_pairs.astype(np.int32), dtype=wp.int32)
        print(str(self.numRingPairs) + " ring-floor pairs created")
        print(str(self.numEdges) + " edges created")

        # Per-vertex incident-primitive tables (CSR) for the hash-grid broadphase:
        # a neighbor POINT found by the grid expands into its incident FACES
        # (vertex-triangle candidates) or incident EDGES (edge-edge candidates).
        # Grid-mesh degrees are tiny (<= 6 faces / 6 edges per vertex).
        corner_v = self.hostTriIds.reshape(-1)
        corner_f = np.repeat(np.arange(self.numTris, dtype=np.int32), 3)
        order = np.argsort(corner_v, kind="stable")
        vf_off = np.zeros(self.numParticles + 1, dtype=np.int32)
        vf_off[1:] = np.cumsum(np.bincount(corner_v, minlength=self.numParticles))
        self.vertFaceOff = wp.array(vf_off, dtype=wp.int32)
        self.vertFaceIds = wp.array(corner_f[order], dtype=wp.int32)

        end_v = edge_arr.reshape(-1)
        end_e = np.repeat(np.arange(self.numEdges, dtype=np.int32), 2)
        order = np.argsort(end_v, kind="stable")
        ve_off = np.zeros(self.numParticles + 1, dtype=np.int32)
        ve_off[1:] = np.cumsum(np.bincount(end_v, minlength=self.numParticles))
        self.vertEdgeOff = wp.array(ve_off, dtype=wp.int32)
        self.vertEdgeIds = wp.array(end_e[order], dtype=wp.int32)

        # detect_expand caches a vertex's incident-edge data in fixed-size
        # per-thread arrays (expandDeg entries); a larger degree would silently
        # index out of bounds, so fail loudly if the topology ever changes.
        max_vf = int(np.diff(vf_off).max())
        max_ve = int(np.diff(ve_off).max())
        assert max_vf <= expandDeg and max_ve <= expandDeg, \
            f"vertex degree {max_vf} faces / {max_ve} edges exceeds expandDeg={expandDeg}"

        # Edge -> (up to 2) adjacent faces, for collect_oversized's announced-
        # face table ([numEdges, 2], -1 padded). Vectorized: the Python loop over
        # 320k triangles took ~3 s of startup at 400x400. edge_arr rows are sorted
        # pairs in lexicographic order, so the a*N+b keys are ascending and
        # searchsorted maps each triangle edge to its edge id; a lexsort by
        # (edge, face) groups each edge's faces ascending, matching the loop's
        # face-major fill order.
        tri = self.hostTriIds.astype(np.int64)
        te = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]), axis=0)
        tkey = te.min(axis=1) * self.numParticles + te.max(axis=1)
        ekeys = edge_arr[:, 0].astype(np.int64) * self.numParticles + edge_arr[:, 1]
        flat_e = np.searchsorted(ekeys, tkey)
        flat_f = np.tile(np.arange(self.numTris, dtype=np.int32), 3)
        order = np.lexsort((flat_f, flat_e))  # by edge, then ascending face id
        se, sf = flat_e[order], flat_f[order]
        counts = np.bincount(se, minlength=self.numEdges)
        starts = np.searchsorted(se, np.arange(self.numEdges))
        ef = np.full((self.numEdges, 2), -1, dtype=np.int32)
        ef[counts >= 1, 0] = sf[starts[counts >= 1]]
        ef[counts >= 2, 1] = sf[starts[counts >= 2] + 1]
        self.edgeFaceIds = wp.array(ef, dtype=wp.int32)

        # Oversized-primitive partition state, rebuilt per substep on the frozen
        # reference state (see step()): per-primitive frozen edge lengths gate
        # the expand-side skips (`> edgeLenCap`, one float load in the hot
        # loops) AND provide the per-candidate Jung slack that keeps squeeze
        # states from saturating the candidate buffers; the compact id list
        # drives the oversized fallback (scatter_oversized / oversized_pairs,
        # O(k) per thread over the compact list).
        self.edgeLen = wp.zeros(self.numEdges, dtype=float)      # frozen |b - a| per edge
        self.faceLongest = wp.zeros(self.numTris, dtype=float)   # longest frozen edge per face
        self.oversizedIds = wp.zeros(self.numEdges, dtype=wp.int32)
        self.oversizedCount = wp.zeros(1, dtype=wp.int32)
        # Per-SLOT announce data for the oversized list, precomputed once per
        # substep by collect_oversized so the vertex-parallel scatter reads a
        # compact L2-resident table instead of re-deriving per (vertex, edge):
        # the faces the edge announces (it is their LONGEST edge; -1 = none),
        # the max apex-to-segment distance of those faces, the edge's own max
        # endpoint sweep, and whether both endpoints are pinned.
        self.ovF0 = wp.zeros(self.numEdges, dtype=wp.int32)
        self.ovF1 = wp.zeros(self.numEdges, dtype=wp.int32)
        self.ovApex = wp.zeros(self.numEdges, dtype=float)
        self.ovSweep = wp.zeros(self.numEdges, dtype=float)
        self.ovPinned = wp.zeros(self.numEdges, dtype=wp.int32)

        # Global per-substep detection bound, recomputed on the GPU inside the
        # captured graph: [0] = longest edge length (frozen reference state),
        # capped by edgeLenCap wherever the walk radii use it (see
        # detect_walk_radius / detect_expand).
        self.detectBounds = wp.zeros(2, dtype=float)

        # Neighbor cache between the two broadphase stages (gather -> expand).
        self.nbrCount = wp.zeros(self.numParticles, dtype=wp.int32)
        self.nbrBuf = wp.zeros(self.numParticles * maxNbr, dtype=wp.int32)

        # Hash grid replacing the LBVH for the self-collision broadphase. Created
        # here (device-side table allocation happens at first build, warmed up in
        # init()/init_headless() outside graph capture).
        self.grid = wp.HashGrid(128, 128, 128)

        # Crossing-resolver state (per-frame exact intersection sweep; see the
        # uncross* constants). hostEdgeIds mirrors edgeIds for the host-side
        # vote; crossBounds[0] = longest CURRENT edge (the sweep runs on current
        # positions, not the frozen reference, so it gets its own bound).
        self.crossPairs = wp.zeros((maxCross, 2), dtype=wp.int32)
        self.crossCount = wp.zeros(1, dtype=wp.int32)
        self.crossBounds = wp.zeros(1, dtype=float)
        self.hostEdgeIds = edge_arr
        self.hostVertFaceOff = vf_off
        self.hostVertFaceIds = self.vertFaceIds.numpy()

        print(str(self.numParticles) + " particles created")
        print(str(self.numTris) + " triangles created")
        print(str(self.distConstraints.count) + " distance constraints created")
        print(str(self.bendConstraints.count) + " bending constraints created")

    @staticmethod
    @wp.kernel
    def rest_distances(
            pos: wp.array(dtype=wp.vec3),
            const_ids: wp.array2d(dtype=wp.int32),
            rest_lengths: wp.array(dtype=float)):
        tid = wp.tid()
        p0 = pos[const_ids[tid, 0]]
        p1 = pos[const_ids[tid, 1]]
        rest_lengths[tid] = wp.length(p1 - p0)

    @staticmethod
    @wp.kernel
    def add_normals(
            pos: wp.array(dtype=wp.vec3),
            tri_ids: wp.array2d(dtype=wp.int32),
            normals: wp.array(dtype=wp.vec3)):
        tid = wp.tid()
        id0 = tri_ids[tid, 0]
        id1 = tri_ids[tid, 1]
        id2 = tri_ids[tid, 2]
        normal = wp.cross(pos[id1] - pos[id0], pos[id2] - pos[id0])
        wp.atomic_add(normals, id0, normal)
        wp.atomic_add(normals, id1, normal)
        wp.atomic_add(normals, id2, normal)

    @staticmethod
    @wp.kernel
    def normalize_normals(normals: wp.array(dtype=wp.vec3)):
        tid = wp.tid()
        normals[tid] = wp.normalize(normals[tid])

    @staticmethod
    @wp.func
    def closest_point_on_triangle(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3) -> wp.vec3:
        # Ericson, Real-Time Collision Detection: closest point on triangle (a,b,c) to p.
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = wp.dot(ab, ap)
        d2 = wp.dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return a
        bp = p - b
        d3 = wp.dot(ab, bp)
        d4 = wp.dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return b
        cp = p - c
        d5 = wp.dot(ab, cp)
        d6 = wp.dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return c
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return a + v * ab
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return a + w * ac
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return b + w * (c - b)
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return a + ab * v + ac * w

    @staticmethod
    @wp.func
    def planar_truncation_t(x: wp.vec3, dx: wp.vec3, n: wp.vec3, p: wp.vec3) -> float:
        # Fraction of displacement dx that reaches the plane (n, through p) from x.
        # Returns 1.0 when moving parallel to or away from the plane (no limit).
        denom = wp.dot(n, dx)
        if wp.abs(denom) < epsilon:
            return 1.0
        t = wp.dot(n, p - x) / denom
        if t < 0.0:
            return 1.0
        gamma_min = 1.0e-3
        return wp.clamp(wp.min(t * gamma_r, t - gamma_min), 0.0, 1.0)

    @staticmethod
    @wp.func
    def within_ring(a: wp.int32, b: wp.int32, num_cols: wp.int32) -> bool:
        # True when grid vertices a and b are within the 2-ring of each other.
        # The mesh is a regular grid: index = row * num_cols + col.
        dr = a // num_cols - b // num_cols
        dc = a % num_cols - b % num_cols
        if dr < 0:
            dr = -dr
        if dc < 0:
            dc = -dc
        return wp.max(dr, dc) <= 2

    @staticmethod
    @wp.func
    def within_ring_rc(grid_rc: wp.array2d(dtype=wp.int32), a: wp.int32, b: wp.int32) -> bool:
        # Division-free within_ring: (row, col) are precomputed per vertex, so the
        # 2-ring test is two int loads + subtracts instead of 4 integer divisions.
        dr = grid_rc[a, 0] - grid_rc[b, 0]
        dc = grid_rc[a, 1] - grid_rc[b, 1]
        if dr < 0:
            dr = -dr
        if dc < 0:
            dc = -dc
        return wp.max(dr, dc) <= 2

    @staticmethod
    @wp.func
    def ring2_rc(ar: wp.int32, ac: wp.int32, br: wp.int32, bc: wp.int32) -> bool:
        # within_ring_rc on PRE-LOADED (row, col) pairs: identical predicate, no
        # grid_rc loads (detect_expand hoists the rows it reuses).
        dr = ar - br
        dc = ac - bc
        if dr < 0:
            dr = -dr
        if dc < 0:
            dc = -dc
        return wp.max(dr, dc) <= 2

    @staticmethod
    @wp.func
    def within_ring_solver(grid_rc: wp.array2d(dtype=wp.int32), a: wp.int32, b: wp.int32) -> bool:
        # SOLVER-side ring cull (see SOLVER_RING). The metric kernels
        # (count_self_contacts / self_contact_gaps) keep the historical 2-ring
        # via within_ring_rc so reported numbers stay comparable.
        dr = grid_rc[a, 0] - grid_rc[b, 0]
        dc = grid_rc[a, 1] - grid_rc[b, 1]
        if dr < 0:
            dr = -dr
        if dc < 0:
            dc = -dc
        return wp.max(dr, dc) <= SOLVER_RING

    @staticmethod
    @wp.func
    def ring_solver(ar: wp.int32, ac: wp.int32, br: wp.int32, bc: wp.int32) -> bool:
        # within_ring_solver on PRE-LOADED (row, col) pairs.
        dr = ar - br
        dc = ac - bc
        if dr < 0:
            dr = -dr
        if dc < 0:
            dc = -dc
        return wp.max(dr, dc) <= SOLVER_RING

    @staticmethod
    @wp.func
    def point_segment_distance(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
        # Distance from point p to segment [a, b].
        ab = b - a
        ab2 = wp.dot(ab, ab)
        t = 0.0
        if ab2 > epsilon:
            t = wp.clamp(wp.dot(p - a, ab) / ab2, 0.0, 1.0)
        return wp.length(a + t * ab - p)

    @staticmethod
    @wp.func
    def closest_point_segment_segment(p1: wp.vec3, q1: wp.vec3,
                                      p2: wp.vec3, q2: wp.vec3):
        # Ericson, Real-Time Collision Detection: closest points between segments
        # [p1,q1] and [p2,q2]. Returns (c1, c2, s, t) with c1 = p1 + s*(q1-p1) and
        # c2 = p2 + t*(q2-p2); s, t are the barycentric parameters along each edge.
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2
        a = wp.dot(d1, d1)
        e = wp.dot(d2, d2)
        f = wp.dot(d2, r)
        s = float(0.0)
        t = float(0.0)
        if a <= epsilon and e <= epsilon:
            s = 0.0
            t = 0.0
        elif a <= epsilon:
            s = 0.0
            t = wp.clamp(f / e, 0.0, 1.0)
        else:
            cc = wp.dot(d1, r)
            if e <= epsilon:
                t = 0.0
                s = wp.clamp(-cc / a, 0.0, 1.0)
            else:
                b = wp.dot(d1, d2)
                denom = a * e - b * b
                if wp.abs(denom) > epsilon:
                    s = wp.clamp((b * f - cc * e) / denom, 0.0, 1.0)
                else:
                    s = 0.0
                t = (b * s + f) / e
                if t < 0.0:
                    t = 0.0
                    s = wp.clamp(-cc / a, 0.0, 1.0)
                elif t > 1.0:
                    t = 1.0
                    s = wp.clamp((b - cc) / a, 0.0, 1.0)
        c1 = p1 + d1 * s
        c2 = p2 + d2 * t
        return c1, c2, s, t

    @staticmethod
    @wp.func
    def radial(v: wp.vec3):
        # Collider-relative RADIAL part of a vector: the component the analytic
        # distance |radial(p - c)| - R acts on. Sphere (kind 0): the identity, so
        # every kind-0 formula below is bit-identical to the sphere-only code.
        # Infinite z-cylinder (kind 1): the xy projection -- z is free.
        if COLLIDER_KIND == 1:
            return wp.vec3(v[0], v[1], 0.0)
        return v

    @staticmethod
    @wp.func
    def axial(v: wp.vec3):
        # Complement of radial(): the collider's free direction. Zero for the
        # sphere (no free direction), the z component for the cylinder.
        if COLLIDER_KIND == 1:
            return wp.vec3(0.0, 0.0, v[2])
        return wp.vec3(0.0, 0.0, 0.0)

    @staticmethod
    @wp.func
    def swept_sphere_ccd(pos: wp.vec3,
                         vel: wp.vec3,
                         center: wp.vec3,
                         radius: float,
                         dc: wp.vec3,
                         dr: float,
                         dt: float):
        # Analytic continuous collision of a particle segment against a moving /
        # growing sphere: solve for the time-of-impact and return the contact point
        # on the swept surface. Exact for a convex analytic collider, so a fast
        # sphere drag cannot tunnel through the cloth regardless of speed.
        # COLLIDER_KIND 1: the same quadratic solved on the RADIAL (xy) projection
        # is exact for the infinite z-cylinder -- axial motion never changes the
        # distance to the surface.
        s = Cloth.radial(center - pos)
        vc = dc / dt
        vr = dr / dt
        v = Cloth.radial(vc - vel)

        c = wp.dot(s, s) - radius * radius
        if c < 0.0:
            # The particle is inside the sphere
            return False, wp.vec3()
        a = wp.dot(v, v) - vr * vr
        if wp.abs(a) < epsilon:
            # The particle is not moving relative to the sphere
            return False, wp.vec3()
        b = wp.dot(v, s) - radius * vr
        if b > 0.0:
            # The particle is not moving towards the sphere
            return False, wp.vec3()
        d = b * b - a * c
        if d < 0.0:
            # The particle segment does not intersect the sphere
            return False, wp.vec3()
        t = (-b - wp.sqrt(d)) / a
        if t >= dt:
            # The particle segment does not intersect the sphere
            return False, wp.vec3()

        p = pos + t * vel
        o = center + t * vc
        r = Cloth.radial(p - o)

        # CARRY-ALONG reconstruction: keep the sphere-relative contact direction the
        # particle had at the time of impact and evaluate it at the END pose -- the
        # particle rides the sphere. The previous reconstruction placed hits on the
        # flank of the swept tube (dead-ahead hits were pushed to an ARBITRARY
        # perpendicular), so a fast sphere actively parted the cloth, bored a hole
        # through the drape and let it close behind -- zero measured penetration,
        # but visually the sphere "passed through" the fabric. Carrying hit
        # particles forward is a plow: physical for fast motion, and safe now that
        # the strain limiter bounds the stretch it creates (carrying was the
        # over-fling hazard before strain limiting existed).
        d_rel = wp.length(r)
        if d_rel < epsilon:
            # Degenerate (particle at the TOI center): let Pass 2 resolve it.
            return False, wp.vec3()
        n = r / d_rel
        # Cylinder: the carried direction is radial (xy); the particle keeps its
        # own free-axis coordinate at the end of the substep. axial() is the zero
        # vector for the sphere, so kind 0 is unchanged.
        return True, center + dc + (radius + dr) * n \
            + Cloth.axial(pos + vel * dt - center - dc)

    @staticmethod
    @wp.kernel
    def integrate(
            dt: float,
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3)):
        tid = wp.tid()

        # Freeze the penetration-free reference state X = prev_pos for this substep.
        # Collisions (sphere, ground, self) are resolved afterwards by PDT truncation
        # and analytic collider projection, not here.
        if inv_mass[tid] == 0.0:
            prev_pos[tid] = pos[tid]
            return

        prev_pos[tid] = pos[tid]
        pos[tid] += (vel[tid] + gravity * dt) * dt

    @staticmethod
    @wp.kernel
    def max_edge_length(
            prev_pos: wp.array(dtype=wp.vec3),
            edge_ids: wp.array2d(dtype=wp.int32),
            bounds: wp.array(dtype=float)):   # out: bounds[0] = longest edge (atomic_max)
        # Longest edge in the frozen reference state. Grid-stride loop: coalesced
        # loads and only `boundsReduceThreads` conflicting atomics.
        t = wp.tid()
        num = edge_ids.shape[0]
        m = float(0.0)
        for e in range(t, num, boundsReduceThreads):
            m = wp.max(m, wp.length(prev_pos[edge_ids[e, 1]] - prev_pos[edge_ids[e, 0]]))
        wp.atomic_max(bounds, 0, m)

    @staticmethod
    @wp.func
    def detect_walk_radius(
            v: wp.int32,
            xv: wp.vec3,
            sweep_v: float,
            free_v: bool,
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            edge_ids: wp.array2d(dtype=wp.int32),
            vert_edge_off: wp.array(dtype=wp.int32),
            vert_edge_ids: wp.array(dtype=wp.int32),
            edge_len: wp.array(dtype=float),
            bounds: wp.array(dtype=float)):
        # Walk radius shared by gather and expand (must be the identical value):
        # r_vt reaches every face relevant to v (see detect_expand), and the
        # edge-edge reach covers e's rc-CAPSULE from its nearer endpoint -- any
        # capsule point is within sqrt(halflen^2 + rc^2) of the nearer endpoint
        # (the caps are within rc of an endpoint), NOT halflen + rc. Radii use
        # only the primitive's OWN sweep (baseline swept-box semantics: own swept
        # box vs frozen other side) plus the frozen longest-edge bound; a global
        # displacement term would let one fast particle inflate every query.
        # (Pinned vertices keep the edge reach: an edge with one free endpoint
        # still collides.)
        # The longest-edge bound is CAPPED at edgeLenCap: one locally stretched
        # edge must not inflate every vertex's walk radius (buffer saturation =
        # silent candidate drops). The capped walk therefore only guarantees
        # coverage of primitives whose frozen edges are all <= edgeLenCap;
        # oversized primitives are excluded here AND in detect_expand (the
        # incident-edge skip below keeps the exact halflen term of a stretched
        # edge from unbounding the radius too) and handled exclusively by
        # the oversized fallback (scatter_oversized / oversized_pairs).
        L = wp.min(bounds[0], edgeLenCap)
        r_vt = float(0.0)
        if free_v:
            r_vt = sweep_v + d_offset + 0.578 * L + 1.0e-4
        r_ee = float(0.0)
        for k in range(vert_edge_off[v], vert_edge_off[v + 1]):
            e = vert_edge_ids[k]
            if edge_len[e] > edgeLenCap:
                continue  # handled by the oversized fallback instead
            other = edge_ids[e, 0] + edge_ids[e, 1] - v  # the endpoint that is not v
            halflen = 0.5 * wp.length(prev_pos[other] - xv)
            sw_e = wp.max(sweep_v, wp.length(pos[other] - prev_pos[other]))
            rc = sw_e + d_offset + 0.5 * L + 1.0e-4
            r_ee = wp.max(r_ee, wp.sqrt(halflen * halflen + rc * rc))
        return wp.max(r_vt, r_ee)

    @staticmethod
    @wp.kernel
    def detect_gather(
            grid: wp.uint64,
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),   # frozen reference X (grid was built on it)
            pos: wp.array(dtype=wp.vec3),        # X + accumulated displacement
            grid_rc: wp.array2d(dtype=wp.int32),
            edge_ids: wp.array2d(dtype=wp.int32),
            vert_edge_off: wp.array(dtype=wp.int32),
            vert_edge_ids: wp.array(dtype=wp.int32),
            edge_len: wp.array(dtype=float),
            bounds: wp.array(dtype=float),       # [0] = longest frozen edge (capped use)
            nbr_count: wp.array(dtype=wp.int32),  # out: accepted neighbors per vertex
            nbr_buf: wp.array(dtype=wp.int32),    # out: neighbor vertex ids
            overflow: wp.array(dtype=wp.int32)):
        # Self-collision BROADPHASE stage 1: ONE hash-grid walk per VERTEX caches
        # the nearby non-ring vertices. The walk is the latency-critical part
        # (grid traversal + scattered point loads), so it lives in this LEAN
        # kernel that runs at high occupancy; the register-heavy candidate
        # expansion reads the cache in a separate query-free kernel
        # (detect_expand). Replaces the old per-vertex AND per-edge LBVH queries
        # (32 KiB traversal stacks, ~50% occupancy cap, 63% of the substep).
        v = wp.tid()
        xv = prev_pos[v]
        sweep_v = wp.length(pos[v] - xv)
        r = Cloth.detect_walk_radius(v, xv, sweep_v, inv_mass[v] != 0.0,
                                     prev_pos, pos, edge_ids,
                                     vert_edge_off, vert_edge_ids,
                                     edge_len, bounds)
        if not (r < 0.5 * maxQueryExtent):  # NaN/blown-up guard (was maxQueryExtent box)
            nbr_count[v] = 0
            return
        base = v * maxNbr
        n = wp.int32(0)
        query = wp.hash_grid_query(grid, xv, r)
        u = wp.int32(0)
        while wp.hash_grid_query_next(query, u):
            if wp.length(prev_pos[u] - xv) > r:
                continue  # the grid over-returns to cell granularity
            if Cloth.within_ring_solver(grid_rc, v, u):
                continue  # every face/edge candidate via u would fail the ring cull
            if n < maxNbr:
                nbr_buf[base + n] = u
                n += 1
            else:
                wp.atomic_add(overflow, 0, 1)
        nbr_count[v] = n

    @staticmethod
    @wp.kernel
    def detect_expand(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),   # frozen reference X
            pos: wp.array(dtype=wp.vec3),        # X + accumulated displacement
            grid_rc: wp.array2d(dtype=wp.int32),
            tri_ids: wp.array2d(dtype=wp.int32),
            edge_ids: wp.array2d(dtype=wp.int32),     # [numEdges, 2], va < vb
            vert_face_off: wp.array(dtype=wp.int32),  # CSR: vertex -> incident faces
            vert_face_ids: wp.array(dtype=wp.int32),
            vert_edge_off: wp.array(dtype=wp.int32),  # CSR: vertex -> incident edges
            vert_edge_ids: wp.array(dtype=wp.int32),
            face_longest: wp.array(dtype=float),  # longest frozen edge per face
            edge_len: wp.array(dtype=float),      # frozen length per edge
            bounds: wp.array(dtype=float),       # [0] = longest frozen edge (capped use)
            nbr_count: wp.array(dtype=wp.int32),  # cached neighbors (detect_gather)
            nbr_buf: wp.array(dtype=wp.int32),
            vt_count: wp.array(dtype=wp.int32),  # out: per-vertex candidate count (PRE-ZEROED, atomic)
            vt_buf: wp.array(dtype=wp.int32),    # out: candidate face ids
            ee_count: wp.array(dtype=wp.int32),  # out: per-edge candidate count (PRE-ZEROED, atomic)
            ee_buf: wp.array(dtype=wp.int32),    # out: candidate EDGE ids per edge
            overflow: wp.array(dtype=wp.int32)):
        # Self-collision BROADPHASE stage 2 (query-free): expand each cached
        # neighbor u into both candidate sets.
        #
        # Vertex-triangle: the old detect paired v with every frozen face whose
        # AABB touched v's swept box (own sweep + d_offset). Any triangle point
        # is within longest_edge/sqrt(3) of one of the triangle's VERTICES
        # (circumradius bound), so accepting neighbors within
        # r_vt = sweep(v) + d_offset + 0.578 * longest_edge and expanding them
        # into their incident faces covers every face that comes within
        # sweep(v) + d_offset of the vertex -- the pairs the truncation can act
        # on this substep (a face APPROACHING a slow vertex is protected, as
        # before, by the face's own vertices' queries and the edge-edge pass).
        # Faces are dedup'd (accepted only from their smallest in-range vertex)
        # so the narrowphase C<0 push (atomic_add, NOT idempotent) counts each
        # pair exactly once; the 2-ring cull is identical to the old BVH detect.
        #
        # Edge-edge: the old detect paired e one-sidedly with the edges of every
        # frozen face whose AABB touched e's swept box (own endpoint sweeps +
        # d_offset); one-sided discovery is safe because the narrowphase
        # truncates all 4 endpoints of a discovered pair. An edge f within
        # d_offset + sweep(e) of segment e has an ENDPOINT u inside e's capsule
        # of radius rc(e) = sweep(e) + d_offset + longest_edge/2 (endpoint gap
        # <= segment gap + halflen(f)), and that u is guaranteed to be in the
        # nearer e-endpoint's neighbor cache (see detect_walk_radius): each
        # vertex v expands every cached u into candidates (edges f incident to
        # u) for each of v's OWN incident edges e, appending to ee_buf[e]
        # through an atomic counter. Dedup: only the e-endpoint nearer to u
        # appends (both endpoints compute identical distances), and only from
        # f's smallest in-capsule endpoint -- each pair lands in ee_buf[e]
        # exactly once. The shared-vertex and pairwise 2-ring culls (the exact
        # predicates the old detect applied) run here, so the narrowphase
        # iterates clean EDGE ids.
        #
        # OVERSIZED partition: the longest-edge slack in both radii is capped at
        # edgeLenCap (see detect_walk_radius), so the coverage arguments above
        # only hold for candidates whose frozen edges fit under the cap. Any
        # oversized candidate face/edge -- and any OWN edge e that is oversized
        # (the gather cache no longer spans its capsule) -- is skipped here with
        # the same `> edgeLenCap` predicate the oversized fallback uses, so the
        # partition is exclusive: no pair is counted by both kernels (the C<0
        # push is not idempotent).
        #
        # PER-CANDIDATE slack: acceptance uses each candidate's OWN frozen
        # longest edge (face_longest / edge_len), not the capped global bound --
        # the Jung / halflen arguments are per-candidate, so this is the exact
        # original criterion, and it keeps a multi-layer squeeze (where the
        # global bound sits at the cap but local edges are at rest length ~1/3
        # of it) from tripling every acceptance radius and saturating the
        # buffers. The capped r_vt / rc below remain the coarse per-neighbor
        # pretests (and the gather radius, which must dominate them).
        v, j = wp.tid()  # (vertex, neighbor slot): slot j strides the neighbor cache
        n_nbr = wp.min(nbr_count[v], maxNbr)  # count may exceed capacity on overflow
        if j >= n_nbr:
            return  # vt_count/ee_count are pre-zeroed; appends are atomic

        xv = prev_pos[v]
        xpv = pos[v]
        sweep_v = wp.length(xpv - xv)

        L = wp.min(bounds[0], edgeLenCap)
        r_vt = float(0.0)
        if inv_mass[v] != 0.0:
            r_vt = sweep_v + d_offset + 0.578 * L + 1.0e-4

        v_row = grid_rc[v, 0]
        v_col = grid_rc[v, 1]

        # HOIST v's incident-edge data (loop-invariant across the cached
        # neighbors; reloading it per (neighbor x edge) was the kernel's main
        # memory traffic). Oversized own edges are excluded here once (the
        # oversized fallback owns their pairs), so the neighbor loop iterates a
        # compact list. All cached values are bitwise-identical to the loads
        # they replace, so the responsibility / dedup tie-breaks still agree
        # across endpoint threads.
        e_id = vec6i()
        e_other = vec6i()
        e_xo = mat6x3f()   # prev_pos[other]
        e_rc = vec6f()     # coarse capsule radius (sw_e + d_offset + 0.5*L + eps)
        e_swe = vec6f()    # own-sweep term sw_e (per-candidate rcf below)
        e_or = vec6i()     # grid_rc[other] (pairwise ring culls)
        e_oc = vec6i()
        n_e = wp.int32(0)
        ee_skip_r = float(0.0)
        for k in range(vert_edge_off[v], vert_edge_off[v + 1]):
            e = vert_edge_ids[k]
            elen = edge_len[e]
            if elen > edgeLenCap:
                continue  # capped cache no longer spans e's capsule; the oversized fallback owns e's pairs
            other = edge_ids[e, 0] + edge_ids[e, 1] - v
            xo = prev_pos[other]
            sw_e = wp.max(sweep_v, wp.length(pos[other] - xo))
            rc = sw_e + d_offset + 0.5 * L + 1.0e-4
            e_id[n_e] = e
            e_other[n_e] = other
            e_xo[n_e] = xo
            e_rc[n_e] = rc
            e_swe[n_e] = sw_e
            e_or[n_e] = grid_rc[other, 0]
            e_oc[n_e] = grid_rc[other, 1]
            n_e += 1
            # d(u, seg e) >= d(u, v) - len(e), so a neighbor beyond rc + len(e)
            # cannot pass e's du_seg <= rc pretest: max over the own edges gives
            # a whole-loop skip radius for far neighbors (pure pruning of cases
            # the original rejected at du_seg > rc; no acceptance change).
            ee_skip_r = wp.max(ee_skip_r, rc + elen)

        base = v * maxVT
        nbase = v * maxNbr

        # Per-NEIGHBOR incident-edge cache (lazy: filled on u's first own edge
        # that passes the capsule pretest, then reused across the remaining own
        # edges -- the same candidate f was previously re-fetched per own edge).
        f_id = vec6i()
        f_w = vec6i()      # f's other endpoint (f = (u, w))
        f_lf = vec6f()     # frozen length of f
        f_wr = vec6i()     # grid_rc[w]
        f_wc = vec6i()

        for ni in range(j, n_nbr, expandK):
            u = nbr_buf[nbase + ni]
            xu = prev_pos[u]
            duv = wp.length(xu - xv)

            # -- vertex-triangle expansion --
            if duv <= r_vt:
                # v's relevance region is the CAPSULE around its own displacement
                # segment [xv, pos(v)] (the old swept-box semantics; truncation
                # keeps the committed point on that segment), NOT the ball of
                # radius sweep+slack around xv: for a fast vertex the ball
                # over-accepts by ~(sweep/slack)^2 and saturates vtBuf exactly in
                # the ejection substeps of a multi-layer squeeze. duv <= r_vt
                # (the ball) stays as the coarse per-neighbor pretest.
                du_path = Cloth.point_segment_distance(xu, xv, xpv)
                for k in range(vert_face_off[u], vert_face_off[u + 1]):
                    face = vert_face_ids[k]
                    fl = face_longest[face]
                    if fl > edgeLenCap:
                        continue  # announced by its longest edge (scatter_oversized)
                    # Per-face acceptance radius (exact Jung slack for THIS face).
                    rf = d_offset + 0.578 * fl + 1.0e-4
                    if du_path > rf:
                        continue
                    i0 = tri_ids[face, 0]
                    i1 = tri_ids[face, 1]
                    i2 = tri_ids[face, 2]
                    if (Cloth.ring_solver(v_row, v_col, grid_rc[i0, 0], grid_rc[i0, 1])
                            or Cloth.ring_solver(v_row, v_col, grid_rc[i1, 0], grid_rc[i1, 1])
                            or Cloth.ring_solver(v_row, v_col, grid_rc[i2, 0], grid_rc[i2, 1])):
                        continue
                    # Dedup: accept the face only from its smallest in-range vertex
                    # (same segment and per-face radius for all of the face's
                    # vertices, so the smallest in-range one is well defined
                    # within this thread).
                    if i0 < u and Cloth.point_segment_distance(prev_pos[i0], xv, xpv) <= rf:
                        continue
                    if i1 < u and Cloth.point_segment_distance(prev_pos[i1], xv, xpv) <= rf:
                        continue
                    if i2 < u and Cloth.point_segment_distance(prev_pos[i2], xv, xpv) <= rf:
                        continue
                    slot_vt = wp.atomic_add(vt_count, v, 1)
                    if slot_vt < maxVT:
                        vt_buf[base + slot_vt] = face
                    else:
                        wp.atomic_add(overflow, 0, 1)

            # -- edge-edge expansion: u gates its incident edges f, tested
            #    against each of v's (cached) incident edges e --
            if duv > ee_skip_r:
                continue  # u beyond every own edge's capsule pretest
            u_row = grid_rc[u, 0]
            u_col = grid_rc[u, 1]
            n_f = wp.int32(-1)  # u's incident-edge cache not filled yet
            for ke in range(n_e):
                other = e_other[ke]
                # (u == other is impossible: `other` is a mesh neighbor of v,
                # within grid ring 1, and the gather ring cull kept only
                # neighbors beyond ring SOLVER_RING >= 1.)
                xo = e_xo[ke]
                # Responsibility: only e's endpoint NEARER to u appends (ties -> va).
                # Both endpoint threads compute bitwise-identical distances here
                # (same subtractions), so exactly one appends.
                dou = wp.length(xu - xo)
                if duv > dou or (duv == dou and v > other):
                    continue  # (v != va <=> v > other under canonical va < vb)
                # Canonical (va, vb) argument order: the capsule and dedup
                # distances must be BITWISE identical no matter which endpoint's
                # thread evaluates them, or a pair could be double-counted or
                # dropped by both.
                pa = xv
                pb = xo
                if v > other:
                    pa = xo
                    pb = xv
                du_seg = Cloth.point_segment_distance(xu, pa, pb)
                if du_seg > e_rc[ke]:
                    continue  # u outside e's widest capsule (coarse pretest)
                if Cloth.ring_solver(e_or[ke], e_oc[ke], u_row, u_col):
                    continue  # pairwise ring cull ((v,u) already culled by gather)
                if n_f < 0:
                    # First passing own edge: cache u's incident edges once
                    # (oversized candidates excluded here, same predicate as
                    # before: scatter_oversized appends them instead).
                    n_f = wp.int32(0)
                    for kk in range(vert_edge_off[u], vert_edge_off[u + 1]):
                        f = vert_edge_ids[kk]
                        lf = edge_len[f]
                        if lf > edgeLenCap:
                            continue
                        w = edge_ids[f, 0] + edge_ids[f, 1] - u
                        f_id[n_f] = f
                        f_w[n_f] = w
                        f_lf[n_f] = lf
                        f_wr[n_f] = grid_rc[w, 0]
                        f_wc[n_f] = grid_rc[w, 1]
                        n_f += 1
                sw_e = e_swe[ke]
                for kf in range(n_f):
                    w = f_w[kf]
                    # Skip pairs that share a vertex (adjacent edges never
                    # separate). u is not an endpoint of e (see above), so
                    # sharing a vertex means w is one of e's endpoints.
                    if w == v or w == other:
                        continue
                    # Per-candidate capsule radius (exact halflen slack for f).
                    rcf = sw_e + d_offset + 0.5 * f_lf[kf] + 1.0e-4
                    if du_seg > rcf:
                        continue
                    # Pairwise 2-ring cull: (va,u)/(vb,u) are done above; check
                    # both e endpoints against f's OTHER endpoint.
                    if (Cloth.ring_solver(v_row, v_col, f_wr[kf], f_wc[kf])
                            or Cloth.ring_solver(e_or[ke], e_oc[ke], f_wr[kf], f_wc[kf])):
                        continue
                    # Dedup: accept f only from its smallest in-capsule endpoint
                    # (same per-candidate radius from both endpoint threads).
                    # u == vd (f's larger endpoint) <=> u > w, and then vc == w.
                    if u > w and Cloth.point_segment_distance(prev_pos[w], pa, pb) <= rcf:
                        continue
                    # EXACT acceptance: keep the pair only if the FROZEN
                    # segment-segment gap is closable by the pair's OWN sweeps
                    # (d_offset + sweep(e) + sweep(f), the same swept-magnitude
                    # criterion scatter_oversized applies; rcf stays as the
                    # coarse pretest and the dedup radius above). BOTH sweeps
                    # matter: with sweep(e) alone, two edges approaching each
                    # other could each drop the pair from their own list while
                    # their combined motion closes the gap this substep. A
                    # dropped pair satisfies gap - sw_e - sw_f > d_offset, and
                    # interior displacement is a convex combination of endpoint
                    # displacements, so gap(t) > d_offset for the whole
                    # substep: no truncation can bind and no C<0 push is
                    # possible. The filter is symmetric in the pair and
                    # runs only in the thread the dedup selected, so the
                    # exactly-once accept becomes once-or-zero -- no pair can
                    # be double-counted. Canonical (vc, vd) = (min, max)
                    # argument order keeps the value bitwise-identical no
                    # matter which endpoint thread evaluates it.
                    xw = prev_pos[w]
                    qc = xu
                    qd = xw
                    if u > w:
                        qc = xw
                        qd = xu
                    sw_f = wp.max(wp.length(pos[u] - xu), wp.length(pos[w] - xw))
                    cse, csf, s_e, s_f = Cloth.closest_point_segment_segment(pa, pb, qc, qd)
                    if wp.length(cse - csf) > sw_e + sw_f + d_offset + 1.0e-4:
                        continue
                    e = e_id[ke]
                    slot = wp.atomic_add(ee_count, e, 1)
                    if slot < maxEE:
                        ee_buf[e * maxEE + slot] = f_id[kf]
                    else:
                        wp.atomic_add(overflow, 0, 1)

    @staticmethod
    @wp.kernel
    def collect_oversized(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            edge_ids: wp.array2d(dtype=wp.int32),
            tri_ids: wp.array2d(dtype=wp.int32),
            edge_face_ids: wp.array2d(dtype=wp.int32),  # [numEdges, 2], -1 pad
            edge_len: wp.array(dtype=float),            # out: frozen length per edge
            oversized_ids: wp.array(dtype=wp.int32),    # out: compact oversized id list
            oversized_count: wp.array(dtype=wp.int32),  # out: list length (PRE-ZEROED)
            ov_f0: wp.array(dtype=wp.int32),            # out, per slot: announced faces (-1 = none)
            ov_f1: wp.array(dtype=wp.int32),
            ov_apex: wp.array(dtype=float),             # out, per slot: max announced apex-to-segment dist
            ov_sweep: wp.array(dtype=float),            # out, per slot: max endpoint sweep
            ov_pinned: wp.array(dtype=wp.int32)):       # out, per slot: both endpoints pinned
        # Cache every edge's FROZEN length (detect's per-candidate slack reads
        # it) and build the compact list of OVERSIZED edges (> edgeLenCap) the
        # fallback kernels iterate. `edge_len[..] > edgeLenCap` is THE partition
        # predicate: detect_walk_radius / detect_expand skip such primitives,
        # scatter_oversized / oversized_pairs handle exactly those (same
        # comparison on the same frozen lengths -> exclusive, no pair is ever
        # double-counted).
        #
        # Faces an oversized edge ANNOUNCES: E must be the face's LONGEST edge
        # (length ties -> smallest sorted vertex pair), so a face with several
        # oversized edges is announced by exactly one slot and each (vertex,
        # face) pair is appended once by the scatter (the C<0 push is not
        # idempotent). The <= 3 threads that evaluate a face compare
        # bitwise-identical lengths (same subtractions up to sign), so the
        # winner is unique -- and announcing from the LONGEST edge minimizes
        # the apex distance (height <= the face's shortest side).
        e = wp.tid()
        a = edge_ids[e, 0]
        b = edge_ids[e, 1]
        xa = prev_pos[a]
        xb = prev_pos[b]
        l = wp.length(xb - xa)
        edge_len[e] = l
        if not (l > edgeLenCap):
            return
        slot = wp.atomic_add(oversized_count, 0, 1)
        oversized_ids[slot] = e
        ov_sweep[slot] = wp.max(wp.length(pos[a] - xa), wp.length(pos[b] - xb))
        pinned = wp.int32(0)
        if inv_mass[a] == 0.0 and inv_mass[b] == 0.0:
            pinned = 1
        ov_pinned[slot] = pinned
        f0 = wp.int32(-1)
        f1 = wp.int32(-1)
        apex = float(0.0)
        for j in range(2):
            f = edge_face_ids[e, j]
            if f < 0:
                continue
            c3 = tri_ids[f, 0] + tri_ids[f, 1] + tri_ids[f, 2] - a - b
            xc = prev_pos[c3]
            la = wp.length(xc - xa)  # edge (a, c3)
            lb = wp.length(xc - xb)  # edge (b, c3)
            wins = wp.int32(0)
            if l > la or (l == la and Cloth.pair_less(a, b, wp.min(a, c3), wp.max(a, c3))):
                if l > lb or (l == lb and Cloth.pair_less(a, b, wp.min(b, c3), wp.max(b, c3))):
                    wins = 1
            if wins == 1:
                apex = wp.max(apex, Cloth.point_segment_distance(xc, xa, xb))
                if f0 < 0:
                    f0 = f
                else:
                    f1 = f
        ov_f0[slot] = f0
        ov_f1[slot] = f1
        ov_apex[slot] = apex

    @staticmethod
    @wp.kernel
    def face_longest_edges(
            prev_pos: wp.array(dtype=wp.vec3),
            tri_ids: wp.array2d(dtype=wp.int32),
            face_longest: wp.array(dtype=float)):  # out: longest frozen edge per face
        # Longest frozen edge per face: detect_expand's per-face Jung slack
        # reads it, and `> edgeLenCap` marks the faces the capped walk no longer
        # covers. Same lengths / same comparison as collect_oversized (|x-y| and
        # |y-x| are bitwise-equal), so the face partition is exactly "some edge
        # of the face is oversized", and collect_oversized's longest-edge
        # announcer is always an oversized edge.
        f = wp.tid()
        p0 = prev_pos[tri_ids[f, 0]]
        p1 = prev_pos[tri_ids[f, 1]]
        p2 = prev_pos[tri_ids[f, 2]]
        face_longest[f] = wp.max(wp.length(p1 - p0),
                                 wp.max(wp.length(p2 - p1), wp.length(p0 - p2)))

    @staticmethod
    @wp.func
    def pair_less(a0: wp.int32, a1: wp.int32, b0: wp.int32, b1: wp.int32) -> bool:
        # Lexicographic order on sorted vertex pairs == edge id order (edgeIds is
        # built sorted), used for deterministic tie-breaks on equal edge lengths.
        return a0 < b0 or (a0 == b0 and a1 < b1)

    @staticmethod
    @wp.kernel
    def scatter_oversized(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),   # frozen reference X
            pos: wp.array(dtype=wp.vec3),        # X + accumulated displacement
            grid_rc: wp.array2d(dtype=wp.int32),
            tri_ids: wp.array2d(dtype=wp.int32),
            edge_ids: wp.array2d(dtype=wp.int32),  # [numEdges, 2], va < vb
            vert_edge_off: wp.array(dtype=wp.int32),
            vert_edge_ids: wp.array(dtype=wp.int32),
            edge_len: wp.array(dtype=float),
            oversized_ids: wp.array(dtype=wp.int32),
            oversized_count: wp.array(dtype=wp.int32),
            ov_f0: wp.array(dtype=wp.int32),
            ov_f1: wp.array(dtype=wp.int32),
            ov_apex: wp.array(dtype=float),
            ov_sweep: wp.array(dtype=float),
            ov_pinned: wp.array(dtype=wp.int32),
            vt_count: wp.array(dtype=wp.int32),   # in/out: appended ATOMICALLY (runs after expand)
            vt_buf: wp.array(dtype=wp.int32),
            ee_count: wp.array(dtype=wp.int32),
            ee_buf: wp.array(dtype=wp.int32),
            overflow: wp.array(dtype=wp.int32)):
        # FALLBACK broadphase, vertex side: OVERSIZED primitives (frozen edge >
        # edgeLenCap) are excluded from the capped vertex walk, so every VERTEX
        # scans the compact oversized list directly -- O(numParticles * k) with
        # a tiny L2-resident table, gated by an early-out when k == 0 (the
        # normal state). An earlier design walked the hash grid along each
        # oversized edge instead (one thread per edge, sampled sub-queries);
        # with a drag-front wad of thousands of stretched edges every such
        # thread iterated most of the wad through fat global-sweep radii, and
        # ONE substep cost ~450 ms. The scan needs no query geometry at all:
        # every accept below uses the vertex's OWN sweep plus the pair's own
        # primitives, so nothing global inflates anything.
        # MUST run AFTER detect_expand: both append atomically to the same
        # pre-zeroed counters, and stream order keeps the fallback's candidates
        # after the expand ones within each per-primitive list.
        v = wp.tid()
        n_over = oversized_count[0]
        if n_over == 0:
            return
        xv = prev_pos[v]
        sweep_v = wp.length(pos[v] - xv)
        free_v = inv_mass[v] != 0.0
        for j in range(n_over):
            E = oversized_ids[j]
            a = edge_ids[E, 0]
            b = edge_ids[E, 1]
            if v == a or v == b:
                continue
            xa = prev_pos[a]
            xb = prev_pos[b]
            du = Cloth.point_segment_distance(xv, xa, xb)
            # Per-pair reach: d_offset + E's own sweep + v's own sweep (+ the
            # slack converting segment distance into primitive distance).
            t0 = d_offset + ov_sweep[j] + sweep_v + 1.0e-4
            apex = ov_apex[j]
            if du > t0 + wp.max(edgeLenCap, apex):
                continue

            # -- vertex-triangle: E's announced faces vs v --
            # (pinned v gets no VT candidates, matching expand's r_vt = 0; the
            # apex slack converts distance-to-face into distance-to-segment,
            # the exact triangle test below is the accept.)
            if free_v and du <= t0 + apex:
                for jj in range(2):
                    f = ov_f0[j]
                    if jj == 1:
                        f = ov_f1[j]
                    if f < 0:
                        continue
                    i0 = tri_ids[f, 0]
                    i1 = tri_ids[f, 1]
                    i2 = tri_ids[f, 2]
                    if v == i0 or v == i1 or v == i2:
                        continue
                    if (Cloth.within_ring_solver(grid_rc, v, i0)
                            or Cloth.within_ring_solver(grid_rc, v, i1)
                            or Cloth.within_ring_solver(grid_rc, v, i2)):
                        continue
                    # Exact relevance (swept-magnitude criterion, >= expand's
                    # one-sided parity): v's own motion plus E's own sweep must
                    # be able to close the frozen gap to THIS face.
                    cp = Cloth.closest_point_on_triangle(
                        prev_pos[i0], prev_pos[i1], prev_pos[i2], xv)
                    if wp.length(xv - cp) > t0:
                        continue
                    slot = wp.atomic_add(vt_count, v, 1)
                    if slot < maxVT:
                        vt_buf[v * maxVT + slot] = f
                    else:
                        wp.atomic_add(overflow, 0, 1)

            # -- edge-edge: E vs v's SHORT incident edges --
            # (edgeLenCap slack: an edge within reach of E has an endpoint
            # within segment-gap + its own length of E's segment, so the
            # faster endpoint always accepts; the exact gap test is the
            # accept.)
            if du > t0 + edgeLenCap:
                continue
            for k in range(vert_edge_off[v], vert_edge_off[v + 1]):
                eu = vert_edge_ids[k]
                vc = edge_ids[eu, 0]
                vd = edge_ids[eu, 1]
                if vc == a or vc == b or vd == a or vd == b:
                    continue  # shares a vertex with E
                if edge_len[eu] > edgeLenCap:
                    continue  # oversized-vs-oversized: oversized_pairs
                w = vc + vd - v
                # Pairwise ring culls, identical predicates to expand.
                if (Cloth.within_ring_solver(grid_rc, a, v)
                        or Cloth.within_ring_solver(grid_rc, b, v)
                        or Cloth.within_ring_solver(grid_rc, a, w)
                        or Cloth.within_ring_solver(grid_rc, b, w)):
                    continue
                # Exact relevance (swept-magnitude criterion, matching expand):
                # the frozen SEGMENT gap must be closable by the pair's own
                # sweeps. Canonical (vc, vd) / (xa, xb) order keeps the value
                # bitwise identical from either endpoint's thread.
                yc = prev_pos[vc]
                yd = prev_pos[vd]
                sw_eu = wp.max(wp.length(pos[vc] - yc), wp.length(pos[vd] - yd))
                cc1, cc2, s_e, t_e = Cloth.closest_point_segment_segment(xa, xb, yc, yd)
                if wp.length(cc1 - cc2) > d_offset + ov_sweep[j] + sw_eu + 1.0e-4:
                    continue
                if v > w:
                    # Dedup: w's thread appends iff w passes ITS accept (the
                    # mirrored expression on the same values -> bitwise equal
                    # across threads), so append from v only when w is out of
                    # range.
                    xw = prev_pos[w]
                    t0_w = d_offset + ov_sweep[j] + wp.length(pos[w] - xw) + 1.0e-4
                    if Cloth.point_segment_distance(xw, xa, xb) <= t0_w + edgeLenCap:
                        continue
                if inv_mass[vc] == 0.0 and inv_mass[vd] == 0.0:
                    # eu's narrowphase thread early-outs (both endpoints
                    # host-driven): hold the pair in E's OWN buffer instead so
                    # free E is still truncated against the pinned edge.
                    # (expand never writes ee_buf[E] for oversized E, so no
                    # double-count.)
                    if ov_pinned[j] == 0:
                        slot = wp.atomic_add(ee_count, E, 1)
                        if slot < maxEE:
                            ee_buf[E * maxEE + slot] = eu
                        else:
                            wp.atomic_add(overflow, 0, 1)
                    continue
                slot = wp.atomic_add(ee_count, eu, 1)
                if slot < maxEE:
                    ee_buf[eu * maxEE + slot] = E
                else:
                    wp.atomic_add(overflow, 0, 1)

    @staticmethod
    @wp.kernel
    def oversized_pairs(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            grid_rc: wp.array2d(dtype=wp.int32),
            edge_ids: wp.array2d(dtype=wp.int32),
            edge_len: wp.array(dtype=float),
            oversized_ids: wp.array(dtype=wp.int32),
            oversized_count: wp.array(dtype=wp.int32),
            ov_sweep: wp.array(dtype=float),
            ee_count: wp.array(dtype=wp.int32),
            ee_buf: wp.array(dtype=wp.int32),
            overflow: wp.array(dtype=wp.int32)):
        # FALLBACK broadphase, oversized-vs-oversized side: two locally
        # stretched edges can cross mid-span with all four endpoints far from
        # the other segment (the classic "X" between long edges) -- neither the
        # capped vertex walk nor the vertex scan (which anchors on ENDPOINTS)
        # reaches those, so pair the oversized set directly: O(k) per oversized
        # thread over the compact list, ~free outside pathological stretch.
        # Each thread appends into its OWN buffer and its counterpart does the
        # converse, so both sides hold the pair once (the both-sided
        # narrowphase norm).
        E = wp.tid()
        if not (edge_len[E] > edgeLenCap):
            return
        a = edge_ids[E, 0]
        b = edge_ids[E, 1]
        if inv_mass[a] == 0.0 and inv_mass[b] == 0.0:
            return  # own narrowphase thread would early-out; the free side appends
        xa = prev_pos[a]
        xb = prev_pos[b]
        own_sweep = wp.max(wp.length(pos[a] - xa), wp.length(pos[b] - xb))
        ebase = E * maxEE
        n_over = oversized_count[0]
        for j in range(n_over):
            F = oversized_ids[j]
            if F == E:
                continue
            vc = edge_ids[F, 0]
            vd = edge_ids[F, 1]
            if vc == a or vc == b or vd == a or vd == b:
                continue
            if (Cloth.within_ring_solver(grid_rc, a, vc)
                    or Cloth.within_ring_solver(grid_rc, a, vd)
                    or Cloth.within_ring_solver(grid_rc, b, vc)
                    or Cloth.within_ring_solver(grid_rc, b, vd)):
                continue
            yc = prev_pos[vc]
            yd = prev_pos[vd]
            c1, c2, s_ab, t_cd = Cloth.closest_point_segment_segment(xa, xb, yc, yd)
            if wp.length(c1 - c2) > d_offset + own_sweep + ov_sweep[j] + 1.0e-4:
                continue  # frozen gap cannot close this substep
            slot = wp.atomic_add(ee_count, E, 1)
            if slot < maxEE:
                ee_buf[ebase + slot] = F
            else:
                wp.atomic_add(overflow, 0, 1)

    @staticmethod
    @wp.kernel
    def self_collision_truncate(
            tri_ids: wp.array2d(dtype=wp.int32),   # (numTris, 3)
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),  # frozen reference state X
            pos: wp.array(dtype=wp.vec3),       # X + accumulated displacement
            vt_count: wp.array(dtype=wp.int32),    # cached candidate counts (detect_expand)
            vt_buf: wp.array(dtype=wp.int32),      # cached candidate tri ids
            n_grabs: wp.array(dtype=wp.int32),     # active grab count (c>=0 budget gate)
            crossed: wp.array(dtype=wp.int32),     # exact crossed-vertex flags (see FLAG_GUARD)
            truncation_ts: wp.array(dtype=float),  # pre-filled with 1.0 (atomic_min)
            push: wp.array(dtype=wp.vec3),         # pre-zeroed C<0 recovery (atomic_add)
            push_limit: wp.array(dtype=float)):    # crossing budget (atomic_min, see FAR_GUARD)
        # Vertex-triangle NARROWPHASE (query-free): iterate the cached candidate faces
        # (detect_gather+detect_expand did the broadphase + 2-ring cull) and do the DIVIDE/TRUNCATE.
        # No BVH query here -> no 32 KiB traversal stack -> high occupancy.
        v = wp.tid()
        if inv_mass[v] == 0.0:
            return

        # The c >= 0 budget writes only have protective value while a grab is
        # active: with no grab, fingertip_project is a no-op and the recovery
        # pushes alone cannot cross a c >= 0 pair (|pushA| + |pushB| <=
        # 2*pushClamp = d_offset <= d). Gating them recovers the pristine
        # narrowphase atomics traffic for grab-free scenes (the 400x400 drape
        # perf gate measured the ungated writes at ~+1.4 ms/frame).
        budget_on = FAR_GUARD != 0 and n_grabs[0] > 0

        xv = prev_pos[v]
        dxv = pos[v] - xv

        n_cand = wp.min(vt_count[v], maxVT)  # scatter_oversized appends atomically; count may exceed capacity on overflow
        base = v * maxVT
        for kc in range(n_cand):
            face = vt_buf[base + kc]
            i0 = tri_ids[face, 0]
            i1 = tri_ids[face, 1]
            i2 = tri_ids[face, 2]

            # Narrowphase + DIVIDE: build one separating plane from the frozen state.
            p0 = prev_pos[i0]
            p1 = prev_pos[i1]
            p2 = prev_pos[i2]
            cp = Cloth.closest_point_on_triangle(p0, p1, p2, xv)
            n_hat = xv - cp
            d = wp.length(n_hat)
            if d < epsilon:
                continue
            n = n_hat / d
            c = d - d_offset  # signed gap (>= 0 at a feasible start)

            dt0 = pos[i0] - p0
            dt1 = pos[i1] - p1
            dt2 = pos[i2] - p2

            if c < 0.0:
                # Feasibility recovery: the pair starts already inside the offset,
                # so truncation cannot guarantee separation. Apply a one-sided push
                # to restore the offset (weighted by each side's approach), and skip
                # the truncation plane this substep.
                delta_v_n = wp.max(-wp.dot(n, dxv), 0.0)
                delta_t_n = wp.max(wp.max(wp.dot(n, dt0), wp.dot(n, dt1)),
                                   wp.max(wp.dot(n, dt2), 0.0))
                s = delta_v_n + delta_t_n
                if s == 0.0:
                    lmbd = 0.5
                else:
                    lmbd = wp.clamp(delta_t_n / s, 0.05, 0.95)
                # If the triangle side cannot move (fully pinned -- e.g. a dragged
                # grab patch plowing through fabric), its lmbd share of the
                # correction would be silently DISCARDED below and the free vertex
                # would receive as little as 5% of the needed separation per
                # substep -- guaranteed crossing under a pinned plow (the fuzzer's
                # dominant failure). Reassign the undeliverable share to the free
                # side instead.
                if inv_mass[i0] == 0.0 and inv_mass[i1] == 0.0 and inv_mass[i2] == 0.0:
                    lmbd = 0.0
                depth = -c  # positive penetration
                tp = lmbd * depth / 3.0
                # Pinned (grabbed) triangle vertices record the share they COULD
                # NOT take as a pressure signal (sim-inert: apply_truncation
                # skips inv_mass==0); see the load-yielding grip notes.
                # With GRAB_EVADE on, each pinned vertex's undeliverable third
                # (tp) is reassigned to the free vertex v -- the PER-VERTEX
                # generalization of the all-pinned lmbd=0 rule above (which it
                # reproduces exactly: lmbd=0 makes tp 0 and v_share depth).
                # A PARTIALLY pinned triangle (the grab patch RIM -- exactly
                # where fabric is pinched at grip closure) otherwise discards
                # the pinned share and under-delivers separation.
                pp = depth / 3.0
                v_share = (1.0 - lmbd) * depth
                if inv_mass[i0] != 0.0:
                    wp.atomic_add(push, i0, -tp * n)
                else:
                    wp.atomic_add(push, i0, -pp * n)
                    if GRAB_EVADE != 0:
                        v_share += tp
                if inv_mass[i1] != 0.0:
                    wp.atomic_add(push, i1, -tp * n)
                else:
                    wp.atomic_add(push, i1, -pp * n)
                    if GRAB_EVADE != 0:
                        v_share += tp
                if inv_mass[i2] != 0.0:
                    wp.atomic_add(push, i2, -tp * n)
                else:
                    wp.atomic_add(push, i2, -pp * n)
                    if GRAB_EVADE != 0:
                        v_share += tp
                wp.atomic_add(push, v, v_share * n)
                if FAR_GUARD != 0:
                    # CROSSING BUDGET (see FAR_GUARD): tighten so the
                    # post-truncation movers (recovery push, fingertip
                    # eviction) cannot compose across the remaining gap d.
                    # BUDGET EXEMPTION for a fully-pinned opponent: against a
                    # dragged grab patch the bounded push/evasion is the ONLY
                    # separator (lmbd = 0 above), and the burst class this
                    # budget exists for is free-free -- so a pinned-plow pair
                    # keeps full evasion bandwidth.
                    if not (inv_mass[i0] == 0.0 and inv_mass[i1] == 0.0
                            and inv_mass[i2] == 0.0):
                        wp.atomic_min(push_limit, v, farGuardKappa * d)
                    if inv_mass[i0] != 0.0:
                        wp.atomic_min(push_limit, i0, farGuardKappa * d)
                    if inv_mass[i1] != 0.0:
                        wp.atomic_min(push_limit, i1, farGuardKappa * d)
                    if inv_mass[i2] != 0.0:
                        wp.atomic_min(push_limit, i2, farGuardKappa * d)
                    bar_floor = farBarrierFloor
                    if FLAG_GUARD != 0 and n_grabs[0] > 0 \
                            and crossed[v] == 0 and crossed[i0] == 0 \
                            and crossed[i1] == 0 and crossed[i2] == 0:
                        # Side-aware floor (see FLAG_GUARD): the exact sweep
                        # says nobody here is crossed, so the "wrong-side
                        # depth" lock hazard does not apply -- protect the
                        # sub-floor gap the 400^2 plow front presses pairs
                        # into before they cross.
                        bar_floor = farBarrierEps
                    if d > bar_floor * d_offset:
                        # CROSSING BARRIER: the offset plane is unreachable
                        # (c < 0) but the frozen surfaces are still d apart --
                        # truncate each free vertex against the remaining-gap
                        # split plane so no truncated writer can complete the
                        # crossing this substep. The recovery push above still
                        # separates the pair exactly as before. Floor-gated
                        # (farBarrierFloor): an ALREADY-CROSSED pair measures
                        # its wrong-side depth as d ~ 0 here, and an unfloored
                        # barrier would truncate the RETURN motion and lock
                        # the crossing (measured as a growing post-release
                        # crossing band). With FLAG_GUARD the floor is side-
                        # aware (bar_floor above).
                        p_bar = cp + (lmbd * d) * n
                        wp.atomic_min(truncation_ts, v,
                                      Cloth.planar_truncation_t(xv, dxv, n, p_bar))
                        if inv_mass[i0] != 0.0:
                            wp.atomic_min(truncation_ts, i0,
                                          Cloth.planar_truncation_t(p0, dt0, -n, p_bar))
                        if inv_mass[i1] != 0.0:
                            wp.atomic_min(truncation_ts, i1,
                                          Cloth.planar_truncation_t(p1, dt1, -n, p_bar))
                        if inv_mass[i2] != 0.0:
                            wp.atomic_min(truncation_ts, i2,
                                          Cloth.planar_truncation_t(p2, dt2, -n, p_bar))
                continue

            # TRUNCATE: split the gap so the harder-approaching side gives more.
            delta_v_n = wp.max(-wp.dot(n, dxv), 0.0)
            delta_t_n = wp.max(wp.max(wp.dot(n, dt0), wp.dot(n, dt1)),
                               wp.max(wp.dot(n, dt2), 0.0))
            s = delta_v_n + delta_t_n
            if s == 0.0:
                lmbd = 0.5
            else:
                lmbd = wp.clamp(delta_t_n / s, 0.05, 0.95)
            p_plane = cp + (d_offset + lmbd * c) * n

            wp.atomic_min(truncation_ts, v, Cloth.planar_truncation_t(xv, dxv, n, p_plane))
            if budget_on:
                # Budget the post-truncation movers on near pairs too (a
                # fingertip eviction step can cross a gap slightly ABOVE
                # d_offset) -- only while a grab is active (see budget_on):
                # without one, pushes alone cannot cross a c >= 0 pair.
                # Fully-pinned opponent exemption: see the c<0 branch note.
                if not (inv_mass[i0] == 0.0 and inv_mass[i1] == 0.0
                        and inv_mass[i2] == 0.0):
                    wp.atomic_min(push_limit, v, farGuardKappa * d)
                if inv_mass[i0] != 0.0:
                    wp.atomic_min(push_limit, i0, farGuardKappa * d)
                if inv_mass[i1] != 0.0:
                    wp.atomic_min(push_limit, i1, farGuardKappa * d)
                if inv_mass[i2] != 0.0:
                    wp.atomic_min(push_limit, i2, farGuardKappa * d)
            # Pinned triangle vertices cannot be truncated; with the pre-contact
            # flag on, record how far their substep displacement would overshoot
            # the shared plane as pressure (sim-inert; back-off direction -n).
            # With GRAB_EVADE also on, reassign the worst pinned overshoot to
            # the free vertex v as an immediate evasion along +n (v holds the
            # contact point on its side, weight 1): the plane recedes by m, so
            # v retreats by m -- the patch pushes v out of its own path.
            m_ev = float(0.0)
            if inv_mass[i0] != 0.0:
                wp.atomic_min(truncation_ts, i0, Cloth.planar_truncation_t(p0, dt0, -n, p_plane))
            elif GRAB_YIELD_PRECONTACT != 0:
                m0 = (1.0 - Cloth.planar_truncation_t(p0, dt0, -n, p_plane)) \
                    * wp.max(wp.dot(n, dt0), 0.0)
                if m0 > 0.0:
                    wp.atomic_add(push, i0, -m0 * n)
                    m_ev = wp.max(m_ev, m0)
            if inv_mass[i1] != 0.0:
                wp.atomic_min(truncation_ts, i1, Cloth.planar_truncation_t(p1, dt1, -n, p_plane))
            elif GRAB_YIELD_PRECONTACT != 0:
                m1 = (1.0 - Cloth.planar_truncation_t(p1, dt1, -n, p_plane)) \
                    * wp.max(wp.dot(n, dt1), 0.0)
                if m1 > 0.0:
                    wp.atomic_add(push, i1, -m1 * n)
                    m_ev = wp.max(m_ev, m1)
            if inv_mass[i2] != 0.0:
                wp.atomic_min(truncation_ts, i2, Cloth.planar_truncation_t(p2, dt2, -n, p_plane))
            elif GRAB_YIELD_PRECONTACT != 0:
                m2 = (1.0 - Cloth.planar_truncation_t(p2, dt2, -n, p_plane)) \
                    * wp.max(wp.dot(n, dt2), 0.0)
                if m2 > 0.0:
                    wp.atomic_add(push, i2, -m2 * n)
                    m_ev = wp.max(m_ev, m2)
            if GRAB_EVADE != 0 and m_ev > 0.0:
                wp.atomic_add(push, v, (grabEvadeGain * m_ev) * n)

    @staticmethod
    @wp.kernel(launch_bounds=(256, 4))  # cap regs (~80->64) -> 4 blocks/SM (67% occ) for
                                        # this hot query-free narrowphase; ptxas verified
                                        # to fit without local-memory spills.
    def self_collision_truncate_edges(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),  # frozen reference state X
            pos: wp.array(dtype=wp.vec3),       # X + accumulated displacement
            edge_ids: wp.array2d(dtype=wp.int32),  # [numEdges, 2], va < vb
            ee_count: wp.array(dtype=wp.int32),    # cached candidate counts (detect_expand)
            ee_buf: wp.array(dtype=wp.int32),      # cached candidate EDGE ids
            n_grabs: wp.array(dtype=wp.int32),     # active grab count (c>=0 budget gate)
            crossed: wp.array(dtype=wp.int32),     # exact crossed-vertex flags (see FLAG_GUARD)
            truncation_ts: wp.array(dtype=float),  # pre-filled with 1.0 (atomic_min)
            push: wp.array(dtype=wp.vec3),         # pre-zeroed C<0 recovery (atomic_add)
            push_limit: wp.array(dtype=float)):    # crossing budget (atomic_min, see FAR_GUARD)
        # Edge-edge NARROWPHASE (query-free): iterate the cached candidate EDGES
        # (detect_expand did the broadphase, the shared-vertex/2-ring culls and the dedup)
        # and do the DIVIDE/TRUNCATE. Catches folds where two edges cross with no vertex
        # near either face (the classic "X" configuration). No query -> high occupancy.
        # No pair-key dedup across THREADS: each edge processes every pair its own query
        # discovers (both sides usually find it); atomic_min makes the redundant
        # constraint order-independent, and the C<0 push touches only the thread's OWN
        # endpoints with complementary (1 - lmbd) weights, so there is no double-count.
        e = wp.tid()
        va = edge_ids[e, 0]
        vb = edge_ids[e, 1]
        wa = inv_mass[va]
        wb = inv_mass[vb]
        if wa == 0.0 and wb == 0.0:
            return  # both endpoints host-driven

        # c >= 0 budget writes gated on an active grab (see the VT kernel).
        budget_on = FAR_GUARD != 0 and n_grabs[0] > 0

        xa = prev_pos[va]
        xb = prev_pos[vb]

        n_cand = wp.min(ee_count[e], maxEE)  # count may exceed capacity on overflow
        ebase = e * maxEE
        for ci in range(n_cand):
            f = ee_buf[ebase + ci]
            vc = edge_ids[f, 0]
            vd = edge_ids[f, 1]

            yc = prev_pos[vc]
            yd = prev_pos[vd]
            # DIVIDE: separating plane through the frozen-state closest points.
            ca, cb, se, sf = Cloth.closest_point_segment_segment(xa, xb, yc, yd)
            n_hat = ca - cb
            d = wp.length(n_hat)
            if d < epsilon:
                continue
            n = n_hat / d
            c = d - d_offset  # signed gap (>= 0 at a feasible start)

            # Per-ENDPOINT displacements: truncate each of the four endpoints by its
            # OWN displacement against the shared plane (newton-style per-vertex
            # atomic_min), not one blended-contact factor applied to both endpoints.
            d_va = pos[va] - xa
            d_vb = pos[vb] - xb
            d_vc = pos[vc] - yc
            d_vd = pos[vd] - yd

            # Normal approach of each side (toward the other), for the lambda split.
            delta_e_n = wp.max(wp.max(-wp.dot(n, d_va), -wp.dot(n, d_vb)), 0.0)  # ab side
            delta_f_n = wp.max(wp.max(wp.dot(n, d_vc), wp.dot(n, d_vd)), 0.0)    # cd side
            ssum = delta_e_n + delta_f_n
            if ssum == 0.0:
                lmbd = 0.5
            else:
                lmbd = wp.clamp(delta_f_n / ssum, 0.05, 0.95)

            if c < 0.0:
                # Feasibility recovery for an already-overlapping pair: push each edge's
                # OWN endpoints apart (weighted by the reference barycentric coord). The
                # two threads' (1 - lmbd) weights are complementary, so the pair
                # separates by ~depth without an atomic_add cross-push double-count.
                # If the CANDIDATE edge is fully pinned (a dragged grab patch), its
                # thread early-outs and its complementary lmbd share is never
                # delivered -- the free edge would receive as little as 5% of the
                # separation per substep (guaranteed crossing under a pinned plow).
                # Take the full correction on this side instead.
                depth = -c
                if inv_mass[vc] == 0.0 and inv_mass[vd] == 0.0:
                    lmbd = 0.0
                    # Fully pinned candidate edge (a dragged grab patch): its own
                    # thread early-outed, so no thread records the plow load on
                    # it -- write the candidate side's undelivered share as a
                    # pressure signal (sim-inert: apply_truncation skips pinned).
                    wp.atomic_add(push, vc, -depth * (1.0 - sf) * n)
                    wp.atomic_add(push, vd, -depth * sf * n)
                elif GRAB_EVADE != 0 and (inv_mass[vc] == 0.0 or inv_mass[vd] == 0.0):
                    # PARTIALLY pinned candidate (the grab patch RIM): its own
                    # thread delivers its lmbd share with barycentric weights,
                    # and the pinned endpoint's portion of that is sim-inert --
                    # take exactly that undeliverable portion on this side
                    # (lmbd_eff = lmbd * delivered fraction; the mirrored
                    # thread's own-endpoint-only writes keep the pair total at
                    # depth with no double count). Per-vertex generalization of
                    # the fully-pinned rule above, which it reproduces at
                    # extra = 1.
                    extra = float(0.0)
                    if inv_mass[vc] == 0.0:
                        extra += 1.0 - sf
                    if inv_mass[vd] == 0.0:
                        extra += sf
                    lmbd = lmbd * (1.0 - extra)
                # Unguarded own-endpoint writes: a free endpoint takes its usual
                # share; a pinned one records it as pressure (sim-inert).
                wp.atomic_add(push, va, (1.0 - lmbd) * depth * (1.0 - se) * n)
                wp.atomic_add(push, vb, (1.0 - lmbd) * depth * se * n)
                if FAR_GUARD != 0:
                    # CROSSING BUDGET (see FAR_GUARD and the VT branch).
                    # Fully-pinned candidate exemption for the va/vb budget:
                    # against a dragged patch the bounded push is the only
                    # separator -- keep its bandwidth. vc/vd keep the budget
                    # (their opponent e has a free endpoint here, or this
                    # thread would have early-outed).
                    cand_pinned = inv_mass[vc] == 0.0 and inv_mass[vd] == 0.0
                    if wa != 0.0 and not cand_pinned:
                        wp.atomic_min(push_limit, va, farGuardKappa * d)
                    if wb != 0.0 and not cand_pinned:
                        wp.atomic_min(push_limit, vb, farGuardKappa * d)
                    if inv_mass[vc] != 0.0:
                        wp.atomic_min(push_limit, vc, farGuardKappa * d)
                    if inv_mass[vd] != 0.0:
                        wp.atomic_min(push_limit, vd, farGuardKappa * d)
                    bar_floor = farBarrierFloor
                    if FLAG_GUARD != 0 and n_grabs[0] > 0 \
                            and crossed[va] == 0 and crossed[vb] == 0 \
                            and crossed[vc] == 0 and crossed[vd] == 0:
                        # Side-aware floor (see FLAG_GUARD and the VT branch).
                        bar_floor = farBarrierEps
                    if d > bar_floor * d_offset:
                        # CROSSING BARRIER, floor-gated against wrong-side
                        # locking of already-crossed pairs (see the VT
                        # branch): truncate every free endpoint against the
                        # remaining-gap split plane (planar_truncation_t is
                        # sign-symmetric in n). All four endpoints, like the
                        # c >= 0 path: one-sided discovery must protect both
                        # edges.
                        p_bar = cb + (lmbd * d) * n
                        if wa != 0.0:
                            wp.atomic_min(truncation_ts, va,
                                          Cloth.planar_truncation_t(xa, d_va, n, p_bar))
                        if wb != 0.0:
                            wp.atomic_min(truncation_ts, vb,
                                          Cloth.planar_truncation_t(xb, d_vb, n, p_bar))
                        if inv_mass[vc] != 0.0:
                            wp.atomic_min(truncation_ts, vc,
                                          Cloth.planar_truncation_t(yc, d_vc, n, p_bar))
                        if inv_mass[vd] != 0.0:
                            wp.atomic_min(truncation_ts, vd,
                                          Cloth.planar_truncation_t(yd, d_vd, n, p_bar))
                continue

            # TRUNCATE: each endpoint against the shared plane by its own displacement
            # (planar_truncation_t is sign-symmetric in n, so both sides use +n).
            p_plane = cb + (d_offset + lmbd * c) * n
            if budget_on:
                # Crossing budget on near pairs, grab-gated (see the VT
                # branch notes, incl. the fully-pinned candidate exemption).
                if not (inv_mass[vc] == 0.0 and inv_mass[vd] == 0.0):
                    if wa != 0.0:
                        wp.atomic_min(push_limit, va, farGuardKappa * d)
                    if wb != 0.0:
                        wp.atomic_min(push_limit, vb, farGuardKappa * d)
                if inv_mass[vc] != 0.0:
                    wp.atomic_min(push_limit, vc, farGuardKappa * d)
                if inv_mass[vd] != 0.0:
                    wp.atomic_min(push_limit, vd, farGuardKappa * d)
            # Pinned endpoints cannot be truncated; with the pre-contact flag on,
            # record their would-be plane overshoot as pressure (sim-inert).
            # Own side approaches along -n (back-off +n); candidate along +n.
            if wa != 0.0:
                wp.atomic_min(truncation_ts, va, Cloth.planar_truncation_t(xa, d_va, n, p_plane))
            elif GRAB_YIELD_PRECONTACT != 0:
                ma = (1.0 - Cloth.planar_truncation_t(xa, d_va, n, p_plane)) \
                    * wp.max(-wp.dot(n, d_va), 0.0)
                if ma > 0.0:
                    wp.atomic_add(push, va, ma * n)
            if wb != 0.0:
                wp.atomic_min(truncation_ts, vb, Cloth.planar_truncation_t(xb, d_vb, n, p_plane))
            elif GRAB_YIELD_PRECONTACT != 0:
                mb = (1.0 - Cloth.planar_truncation_t(xb, d_vb, n, p_plane)) \
                    * wp.max(-wp.dot(n, d_vb), 0.0)
                if mb > 0.0:
                    wp.atomic_add(push, vb, mb * n)
            # GRAB_EVADE: gather the CANDIDATE side's pinned overshoot at the
            # contact point (barycentric mix at sf; a free candidate endpoint
            # is truncated, so its overshoot is 0) and reassign it to the OWN
            # free endpoints below -- own-endpoint-only writes, so the
            # mirrored discovery (if any) cannot double-deliver.
            m_cand = float(0.0)
            if inv_mass[vc] != 0.0:
                wp.atomic_min(truncation_ts, vc, Cloth.planar_truncation_t(yc, d_vc, n, p_plane))
            elif GRAB_YIELD_PRECONTACT != 0:
                mc = (1.0 - Cloth.planar_truncation_t(yc, d_vc, n, p_plane)) \
                    * wp.max(wp.dot(n, d_vc), 0.0)
                if mc > 0.0:
                    wp.atomic_add(push, vc, -mc * n)
                    m_cand += (1.0 - sf) * mc
            if inv_mass[vd] != 0.0:
                wp.atomic_min(truncation_ts, vd, Cloth.planar_truncation_t(yd, d_vd, n, p_plane))
            elif GRAB_YIELD_PRECONTACT != 0:
                md = (1.0 - Cloth.planar_truncation_t(yd, d_vd, n, p_plane)) \
                    * wp.max(wp.dot(n, d_vd), 0.0)
                if md > 0.0:
                    wp.atomic_add(push, vd, -md * n)
                    m_cand += sf * md
            if GRAB_EVADE != 0 and m_cand > 0.0:
                # The candidate's contact point invades the own side by m_cand
                # along +n; the own free endpoints absorb it with the same
                # barycentric share split the c<0 recovery uses. Pinned own
                # endpoints take theirs as pressure (the write below would be
                # sim-inert anyway, but their share is genuinely undeliverable
                # -- leave it to the yield stall).
                ev = grabEvadeGain * m_cand
                if wa != 0.0:
                    wp.atomic_add(push, va, (ev * (1.0 - se)) * n)
                if wb != 0.0:
                    wp.atomic_add(push, vb, (ev * se) * n)

    @staticmethod
    @wp.kernel
    def rim_truncate_vt(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),      # frozen reference state X
            pos: wp.array(dtype=wp.vec3),           # X + accumulated displacement
            pairs: wp.array2d(dtype=wp.int32),      # [maxRimVT, 4] = (v, i0, i1, i2)
            count: wp.array(dtype=wp.int32),
            truncation_ts: wp.array(dtype=float),   # shared with the main narrowphase
            push: wp.array(dtype=wp.vec3)):
        # Rim vertex-face DIVIDE/TRUNCATE (see RIM_SOLVER): the standard PDT
        # plane on an explicit grab-rim pair list, at the REDUCED separation
        # rimDOffset, with every pinned share/overshoot reassigned to the free
        # side. One thread per pair (the list has no mirrored duplicates), no
        # pressure recording on pinned slots (grip yield unchanged).
        if RIM_SOLVER == 0:
            return
        t = wp.tid()
        if t >= count[0]:
            return
        v = pairs[t, 0]
        i0 = pairs[t, 1]
        i1 = pairs[t, 2]
        i2 = pairs[t, 3]
        wv = inv_mass[v]
        w0 = inv_mass[i0]
        w1 = inv_mass[i1]
        w2 = inv_mass[i2]

        xv = prev_pos[v]
        dxv = pos[v] - xv
        p0 = prev_pos[i0]
        p1 = prev_pos[i1]
        p2 = prev_pos[i2]
        cp = Cloth.closest_point_on_triangle(p0, p1, p2, xv)
        n_hat = xv - cp
        d = wp.length(n_hat)
        if d < epsilon:
            return
        n = n_hat / d
        c = d - rimDOffsetVT

        dt0 = pos[i0] - p0
        dt1 = pos[i1] - p1
        dt2 = pos[i2] - p2

        delta_v_n = wp.max(-wp.dot(n, dxv), 0.0)
        delta_t_n = wp.max(wp.max(wp.dot(n, dt0), wp.dot(n, dt1)),
                           wp.max(wp.dot(n, dt2), 0.0))
        s = delta_v_n + delta_t_n
        if s == 0.0:
            lmbd = 0.5
        else:
            lmbd = wp.clamp(delta_t_n / s, 0.05, 0.95)

        nf_t = float(0.0)  # free face-vertex count
        if w0 != 0.0:
            nf_t += 1.0
        if w1 != 0.0:
            nf_t += 1.0
        if w2 != 0.0:
            nf_t += 1.0

        if c < 0.0:
            # Feasibility recovery at the rim: one-sided push toward rimDOffset.
            # Pinned shares are undeliverable -- reassign them across the pair
            # so the RELATIVE separation stays ~depth (the c<0 lmbd rules'
            # generalization; the pair list has no mirror thread, so this
            # thread delivers both sides).
            depth = -c
            tp = lmbd * depth / 3.0
            v_share = (1.0 - lmbd) * depth
            t_undeliv = (3.0 - nf_t) * tp
            extra_t = float(0.0)
            if wv != 0.0:
                wp.atomic_add(push, v, (v_share + t_undeliv) * n)
            else:
                if nf_t == 0.0:
                    return  # fully pinned pair: nothing can move
                extra_t = (v_share + t_undeliv) / nf_t
            if w0 != 0.0:
                wp.atomic_add(push, i0, -(tp + extra_t) * n)
            if w1 != 0.0:
                wp.atomic_add(push, i1, -(tp + extra_t) * n)
            if w2 != 0.0:
                wp.atomic_add(push, i2, -(tp + extra_t) * n)
            return

        # TRUNCATE free vertices against the shared plane; a pinned vertex
        # cannot be truncated, so its plane overshoot (it WILL cross by this
        # much -- pinned motion ignores planes) is delivered to the free side
        # as an immediate evasion displacement, GRAB_EVADE-style: the plane
        # effectively recedes by m, so the free side retreats by m within the
        # same substep instead of being crossed next substep.
        p_plane = cp + (rimDOffsetVT + lmbd * c) * n
        tv = Cloth.planar_truncation_t(xv, dxv, n, p_plane)
        m_v = float(0.0)   # pinned v overshoot -> free face vertices
        if wv != 0.0:
            wp.atomic_min(truncation_ts, v, tv)
        else:
            m_v = (1.0 - tv) * wp.max(-wp.dot(n, dxv), 0.0)
        m_t = float(0.0)   # worst pinned face-vertex overshoot -> v
        t0 = Cloth.planar_truncation_t(p0, dt0, n, p_plane)
        if w0 != 0.0:
            wp.atomic_min(truncation_ts, i0, t0)
        else:
            m_t = wp.max(m_t, (1.0 - t0) * wp.max(wp.dot(n, dt0), 0.0))
        t1 = Cloth.planar_truncation_t(p1, dt1, n, p_plane)
        if w1 != 0.0:
            wp.atomic_min(truncation_ts, i1, t1)
        else:
            m_t = wp.max(m_t, (1.0 - t1) * wp.max(wp.dot(n, dt1), 0.0))
        t2 = Cloth.planar_truncation_t(p2, dt2, n, p_plane)
        if w2 != 0.0:
            wp.atomic_min(truncation_ts, i2, t2)
        else:
            m_t = wp.max(m_t, (1.0 - t2) * wp.max(wp.dot(n, dt2), 0.0))
        if wv != 0.0 and m_t > 0.0:
            wp.atomic_add(push, v, m_t * n)
        if m_v > 0.0 and nf_t > 0.0:
            if w0 != 0.0:
                wp.atomic_add(push, i0, -m_v * n)
            if w1 != 0.0:
                wp.atomic_add(push, i1, -m_v * n)
            if w2 != 0.0:
                wp.atomic_add(push, i2, -m_v * n)

    @staticmethod
    @wp.kernel
    def rim_truncate_ee(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),      # frozen reference state X
            pos: wp.array(dtype=wp.vec3),           # X + accumulated displacement
            pairs: wp.array2d(dtype=wp.int32),      # [maxRimEE, 4] = (va, vb, vc, vd)
            count: wp.array(dtype=wp.int32),
            truncation_ts: wp.array(dtype=float),   # shared with the main narrowphase
            push: wp.array(dtype=wp.vec3)):
        # Rim edge-edge DIVIDE/TRUNCATE (see RIM_SOLVER): per-endpoint
        # truncation against the shared frozen-closest-point plane at the
        # reduced rim separation. One thread per unordered pair, so this
        # thread delivers BOTH sides' c<0 shares (barycentric weights, pinned
        # shares redistributed within the side, a fully pinned side's share
        # folded into the other side) -- no mirrored-discovery double count.
        if RIM_SOLVER == 0:
            return
        t = wp.tid()
        if t >= count[0]:
            return
        va = pairs[t, 0]
        vb = pairs[t, 1]
        vc = pairs[t, 2]
        vd = pairs[t, 3]
        wa = inv_mass[va]
        wb = inv_mass[vb]
        wc = inv_mass[vc]
        wd = inv_mass[vd]

        xa = prev_pos[va]
        xb = prev_pos[vb]
        yc = prev_pos[vc]
        yd = prev_pos[vd]
        ca, cb, se, sf = Cloth.closest_point_segment_segment(xa, xb, yc, yd)
        n_hat = ca - cb
        d = wp.length(n_hat)
        if d < epsilon:
            return
        n = n_hat / d
        c = d - rimDOffsetEE

        d_va = pos[va] - xa
        d_vb = pos[vb] - xb
        d_vc = pos[vc] - yc
        d_vd = pos[vd] - yd

        delta_e_n = wp.max(wp.max(-wp.dot(n, d_va), -wp.dot(n, d_vb)), 0.0)
        delta_f_n = wp.max(wp.max(wp.dot(n, d_vc), wp.dot(n, d_vd)), 0.0)
        ssum = delta_e_n + delta_f_n
        if ssum == 0.0:
            lmbd = 0.5
        else:
            lmbd = wp.clamp(delta_f_n / ssum, 0.05, 0.95)

        if c < 0.0:
            depth = -c
            share_e = (1.0 - lmbd) * depth
            share_f = lmbd * depth
            fe = float(0.0)   # deliverable barycentric weight, e side
            if wa != 0.0:
                fe += 1.0 - se
            if wb != 0.0:
                fe += se
            ff = float(0.0)   # deliverable barycentric weight, f side
            if wc != 0.0:
                ff += 1.0 - sf
            if wd != 0.0:
                ff += sf
            if fe == 0.0 and ff == 0.0:
                return
            if fe == 0.0:
                share_f += share_e
                share_e = 0.0
            if ff == 0.0:
                share_e += share_f
                share_f = 0.0
            if share_e > 0.0 and fe > 0.0:
                if wa != 0.0:
                    wp.atomic_add(push, va, (share_e * (1.0 - se) / fe) * n)
                if wb != 0.0:
                    wp.atomic_add(push, vb, (share_e * se / fe) * n)
            if share_f > 0.0 and ff > 0.0:
                if wc != 0.0:
                    wp.atomic_add(push, vc, -(share_f * (1.0 - sf) / ff) * n)
                if wd != 0.0:
                    wp.atomic_add(push, vd, -(share_f * sf / ff) * n)
            return

        # TRUNCATE each free endpoint by its own displacement; pinned-endpoint
        # plane overshoots are gathered per side (barycentric mix, like the
        # main EE evade) and delivered to the OTHER side's free endpoints.
        p_plane = cb + (rimDOffsetEE + lmbd * c) * n
        m_e = float(0.0)   # e-side pinned overshoot (approach along -n)
        ta = Cloth.planar_truncation_t(xa, d_va, n, p_plane)
        if wa != 0.0:
            wp.atomic_min(truncation_ts, va, ta)
        else:
            m_e += (1.0 - se) * (1.0 - ta) * wp.max(-wp.dot(n, d_va), 0.0)
        tb = Cloth.planar_truncation_t(xb, d_vb, n, p_plane)
        if wb != 0.0:
            wp.atomic_min(truncation_ts, vb, tb)
        else:
            m_e += se * (1.0 - tb) * wp.max(-wp.dot(n, d_vb), 0.0)
        m_f = float(0.0)   # f-side pinned overshoot (approach along +n)
        tc = Cloth.planar_truncation_t(yc, d_vc, n, p_plane)
        if wc != 0.0:
            wp.atomic_min(truncation_ts, vc, tc)
        else:
            m_f += (1.0 - sf) * (1.0 - tc) * wp.max(wp.dot(n, d_vc), 0.0)
        td = Cloth.planar_truncation_t(yd, d_vd, n, p_plane)
        if wd != 0.0:
            wp.atomic_min(truncation_ts, vd, td)
        else:
            m_f += sf * (1.0 - td) * wp.max(wp.dot(n, d_vd), 0.0)
        if m_f > 0.0:
            if wa != 0.0:
                wp.atomic_add(push, va, (m_f * (1.0 - se)) * n)
            if wb != 0.0:
                wp.atomic_add(push, vb, (m_f * se) * n)
        if m_e > 0.0:
            if wc != 0.0:
                wp.atomic_add(push, vc, -(m_e * (1.0 - sf)) * n)
            if wd != 0.0:
                wp.atomic_add(push, vd, -(m_e * sf) * n)

    @staticmethod
    @wp.kernel(launch_bounds=(256, 4))
    def contact_repulsion(
            tri_ids: wp.array2d(dtype=wp.int32),
            inv_mass: wp.array(dtype=float),
            pos: wp.array(dtype=wp.vec3),          # CURRENT positions (not the frozen reference)
            prev_pos: wp.array(dtype=wp.vec3),     # frozen reference (approach gate only)
            vt_count: wp.array(dtype=wp.int32),    # cached candidate counts (detect_expand, read-only)
            vt_buf: wp.array(dtype=wp.int32),      # cached candidate tri ids (read-only)
            deltas: wp.array(dtype=wp.vec3)):      # pre-zeroed, applied by add_deltas
        # Unilateral vertex-triangle contact repulsion: soft pressure between
        # candidate pairs closer than the engage distance, pushing them apart
        # along the current closest-point direction. This is a FORCE-like pass,
        # not a PDT plane: it reads current geometry only, so there is no
        # reference-state contract to honor, and the truncation narrowphase runs
        # AFTER it in the substep and truncates any resulting crossing among
        # candidate pairs. Step-capped (repulsionCap <= 0.5*d_offset) so a
        # feasible-start pair (gap >= d_offset) can never be pushed across a
        # neighboring sheet in one pass.
        v = wp.tid()
        if inv_mass[v] == 0.0:
            return  # pinned targets get pressure only via the c<0 push signal
        xv = pos[v]
        engage = repulsionEngage * d_offset
        n_cand = wp.min(vt_count[v], maxVT)
        base = v * maxVT
        for kc in range(n_cand):
            face = vt_buf[base + kc]
            i0 = tri_ids[face, 0]
            i1 = tri_ids[face, 1]
            i2 = tri_ids[face, 2]
            cp = Cloth.closest_point_on_triangle(pos[i0], pos[i1], pos[i2], xv)
            n_hat = xv - cp
            d = wp.length(n_hat)
            if d < epsilon or d >= engage:
                continue
            n = n_hat / d
            corr = wp.min(repulsionK * (engage - d), repulsionCap)
            if REPULSION_APPROACH != 0:
                xpv = prev_pos[v]
                cpp = Cloth.closest_point_on_triangle(
                    prev_pos[i0], prev_pos[i1], prev_pos[i2], xpv)
                if wp.length(xpv - cpp) - d <= repulsionApproachEps:
                    corr = corr * repulsionSepGain  # not approaching: soften
                    if corr <= 0.0:
                        continue
            # Split the separation between the two sides; if the triangle side
            # is fully pinned its share is undeliverable -- reassign it to the
            # vertex (same rule as the c<0 push).
            lmbd = 0.5
            if inv_mass[i0] == 0.0 and inv_mass[i1] == 0.0 and inv_mass[i2] == 0.0:
                lmbd = 0.0
            wp.atomic_add(deltas, v, ((1.0 - lmbd) * corr) * n)
            tp = lmbd * corr / 3.0
            if inv_mass[i0] != 0.0:
                wp.atomic_add(deltas, i0, -tp * n)
            if inv_mass[i1] != 0.0:
                wp.atomic_add(deltas, i1, -tp * n)
            if inv_mass[i2] != 0.0:
                wp.atomic_add(deltas, i2, -tp * n)
            if frictionMu > 0.0 and d < frictionEngage * d_offset \
                    and d > frictionFloor * d_offset:
                # Coulomb grip on PRESSED contacts only: damp the relative
                # tangential slip over the detection window, budgeted by the
                # normal correction (cap = mu * corr). Barycentric weights of
                # the CURRENT closest point applied to prev positions estimate
                # the face's motion -- no second triangle solve.
                e0 = pos[i1] - pos[i0]
                e1 = pos[i2] - pos[i0]
                cv = cp - pos[i0]
                d00 = wp.dot(e0, e0)
                d01 = wp.dot(e0, e1)
                d11 = wp.dot(e1, e1)
                den = d00 * d11 - d01 * d01
                w1 = 0.0
                w2 = 0.0
                if den > epsilon:
                    w1 = (d11 * wp.dot(cv, e0) - d01 * wp.dot(cv, e1)) / den
                    w2 = (d00 * wp.dot(cv, e1) - d01 * wp.dot(cv, e0)) / den
                w0 = 1.0 - w1 - w2
                face_dv = w0 * (pos[i0] - prev_pos[i0]) \
                    + w1 * (pos[i1] - prev_pos[i1]) \
                    + w2 * (pos[i2] - prev_pos[i2])
                rel = (xv - prev_pos[v]) - face_dv
                rel_t = rel - wp.dot(rel, n) * n
                st = wp.length(rel_t)
                if st > epsilon:
                    fmag = wp.min(st, frictionMu * corr)
                    fvec = rel_t * (fmag / st)
                    wp.atomic_add(deltas, v, -(1.0 - lmbd) * fvec)
                    fp = lmbd / 3.0
                    if inv_mass[i0] != 0.0:
                        wp.atomic_add(deltas, i0, fp * fvec)
                    if inv_mass[i1] != 0.0:
                        wp.atomic_add(deltas, i1, fp * fvec)
                    if inv_mass[i2] != 0.0:
                        wp.atomic_add(deltas, i2, fp * fvec)

    @staticmethod
    @wp.kernel(launch_bounds=(256, 4))
    def contact_repulsion_edges(
            inv_mass: wp.array(dtype=float),
            pos: wp.array(dtype=wp.vec3),          # CURRENT positions
            prev_pos: wp.array(dtype=wp.vec3),     # frozen reference (approach gate only)
            edge_ids: wp.array2d(dtype=wp.int32),  # [numEdges, 2]
            ee_count: wp.array(dtype=wp.int32),    # cached candidate counts (read-only)
            ee_buf: wp.array(dtype=wp.int32),      # cached candidate EDGE ids (read-only)
            deltas: wp.array(dtype=wp.vec3)):      # shared pre-zeroed accumulator
        # Edge-edge companion of contact_repulsion (the "X" crossing config that
        # has no vertex near either face). Same structure as the truncate_edges
        # c<0 push: each thread displaces only its OWN endpoints (barycentric
        # split), the twin thread's complementary share covers the other edge
        # when discovery is two-sided; one-sided discovery still separates the
        # pair at half rate.
        e = wp.tid()
        va = edge_ids[e, 0]
        vb = edge_ids[e, 1]
        wa = inv_mass[va]
        wb = inv_mass[vb]
        if wa == 0.0 and wb == 0.0:
            return
        xa = pos[va]
        xb = pos[vb]
        engage = repulsionEngage * d_offset
        n_cand = wp.min(ee_count[e], maxEE)
        ebase = e * maxEE
        for ci in range(n_cand):
            f = ee_buf[ebase + ci]
            vc = edge_ids[f, 0]
            vd = edge_ids[f, 1]
            ca, cb, se, sf = Cloth.closest_point_segment_segment(xa, xb, pos[vc], pos[vd])
            n_hat = ca - cb
            d = wp.length(n_hat)
            if d < epsilon or d >= engage:
                continue
            n = n_hat / d
            corr = wp.min(repulsionK * (engage - d), repulsionCap)
            if REPULSION_APPROACH != 0:
                cpa, cpb, spe, spf = Cloth.closest_point_segment_segment(
                    prev_pos[va], prev_pos[vb], prev_pos[vc], prev_pos[vd])
                if wp.length(cpa - cpb) - d <= repulsionApproachEps:
                    corr = corr * repulsionSepGain  # not approaching: soften
                    if corr <= 0.0:
                        continue
            lmbd = 0.5
            if inv_mass[vc] == 0.0 and inv_mass[vd] == 0.0:
                lmbd = 0.0  # opposing side pinned: take the full separation here
            if wa != 0.0:
                wp.atomic_add(deltas, va, ((1.0 - lmbd) * corr * (1.0 - se)) * n)
            if wb != 0.0:
                wp.atomic_add(deltas, vb, ((1.0 - lmbd) * corr * se) * n)
            if frictionMu > 0.0 and d < frictionEngage * d_offset \
                    and d > frictionFloor * d_offset:
                # Coulomb slip damping at PRESSED contact points, own endpoints
                # only (barycentric split; twin thread covers the other edge).
                own = xa + se * (xb - xa)
                own_p = prev_pos[va] + se * (prev_pos[vb] - prev_pos[va])
                oth = pos[vc] + sf * (pos[vd] - pos[vc])
                oth_p = prev_pos[vc] + sf * (prev_pos[vd] - prev_pos[vc])
                rel = (own - own_p) - (oth - oth_p)
                rel_t = rel - wp.dot(rel, n) * n
                st = wp.length(rel_t)
                if st > epsilon:
                    fmag = wp.min(st, frictionMu * corr)
                    fvec = rel_t * (fmag / st)
                    if wa != 0.0:
                        wp.atomic_add(deltas, va, -((1.0 - lmbd) * (1.0 - se)) * fvec)
                    if wb != 0.0:
                        wp.atomic_add(deltas, vb, -((1.0 - lmbd) * se) * fvec)

    @staticmethod
    @wp.kernel
    def pinch_extrude(
            inv_mass: wp.array(dtype=float),
            pos: wp.array(dtype=wp.vec3),          # CURRENT positions
            center: wp.array(dtype=wp.vec3),       # CURRENT substep sphere center
            radius: wp.array(dtype=float),
            dc_arr: wp.array(dtype=wp.vec3),       # PER-SUBSTEP sphere translation
            dr_arr: wp.array(dtype=float),         # PER-SUBSTEP radius change
            vt_count: wp.array(dtype=wp.int32),    # candidate-face counts (stack pressure proxy)
            deltas: wp.array(dtype=wp.vec3)):      # pre-zeroed, applied by add_deltas
        # Pinch-wedge lateral extrusion (see the extrudeGain constants note):
        # fabric squeezed under a descending or floor-parked shell has only
        # near-vertical contact normals -- no subsystem transports it sideways,
        # so an overfull stack (more layers than the closing shell/pool or
        # shell/floor wedge can hold) pancakes to ~zero gap, where every
        # post-PDT writer's step exceeds the layer spacing and the layer order
        # scrambles (the permanent knots). Give particles under REAL stacking
        # pressure (vt_count from this substep's detect pass) whose overhead
        # shell clearance is closing a bounded horizontal step away from the
        # sphere axis -- out of the wedge, toward where the shell curves up.
        # Runs pre-narrowphase so the PDT planes veto any step that would
        # cross a sheet (fabric stops at fold walls instead of punching
        # through). Inert when the sphere hovers, rests high (hammock), or the
        # local stack is 1-2 layers.
        i = wp.tid()
        if COLLIDER_KIND != 0:
            return  # sphere-crush-specific geometry: compiled out for the rod
        if inv_mass[i] == 0.0:
            return
        if vt_count[i] < extrudePressure:
            return
        c_end = center[0] + dc_arr[0]
        r_end = radius[0] + dr_arr[0] + thickness
        # Engage only while the wedge is CLOSING: the shell descends, grows, or
        # is already parked at its floor clamp. A sphere resting statically in
        # a draped hammock keeps its wrap.
        if (dc_arr[0][1] >= 0.0 and dr_arr[0] <= 0.0
                and c_end[1] - r_end >= thickness + extrudeBand):
            return
        x = pos[i]
        if x[1] >= c_end[1]:
            return  # above the sphere equator: not in a closing wedge
        # Engage in a radial band around the LOWER shell surface: covers both
        # the floor corridor (fabric under the shell bottom) and the flank
        # wedge (fabric pressed between the diagonal shell and the pool during
        # the descent -- where the crush wads actually form). The horizontal
        # outward step is ~tangent on the flank, pointing out of the wedge.
        dvs = x - c_end
        d_sh = wp.length(dvs) - r_end
        if d_sh < -2.0 * d_offset or d_sh > extrudeBand:
            return
        hx = dvs[0]
        hz = dvs[2]
        hd2 = hx * hx + hz * hz
        if hd2 <= epsilon:
            return  # exactly on the axis: no exit azimuth
        s_ex = extrudeGain * projClamp / wp.sqrt(hd2)
        wp.atomic_add(deltas, i, wp.vec3(s_ex * hx, 0.0, s_ex * hz))

    @staticmethod
    @wp.kernel
    def count_self_contacts(
            mesh: wp.uint64,
            pos: wp.array(dtype=wp.vec3),
            grid_rc: wp.array2d(dtype=wp.int32),
            thresh: float,
            min_dist: wp.array(dtype=float),   # size 1, atomic_min (pre-filled large)
            n_violations: wp.array(dtype=wp.int32)):  # size 1, atomic_add (pre-zeroed)
        # DIAGNOSTIC ONLY (not part of the solve): vertex-triangle self-proximity on
        # the CURRENT geometry. Reports the smallest non-neighbor vertex-triangle gap
        # and how many are closer than `thresh` (separation violations). A correct
        # self-collision keeps this gap ~d_offset and n_violations ~0.
        v = wp.tid()
        xv = pos[v]
        r = thresh + d_offset
        lo = xv - wp.vec3(r, r, r)
        hi = xv + wp.vec3(r, r, r)
        query = wp.mesh_query_aabb(mesh, lo, hi)
        face = wp.int32(0)
        while wp.mesh_query_aabb_next(query, face):
            i0 = wp.mesh_get_index(mesh, 3 * face + 0)
            i1 = wp.mesh_get_index(mesh, 3 * face + 1)
            i2 = wp.mesh_get_index(mesh, 3 * face + 2)
            if (Cloth.within_ring_rc(grid_rc, v, i0)
                    or Cloth.within_ring_rc(grid_rc, v, i1)
                    or Cloth.within_ring_rc(grid_rc, v, i2)):
                continue
            cp = Cloth.closest_point_on_triangle(pos[i0], pos[i1], pos[i2], xv)
            dist = wp.length(xv - cp)
            wp.atomic_min(min_dist, 0, dist)
            if dist < thresh:
                wp.atomic_add(n_violations, 0, 1)

    @staticmethod
    @wp.kernel
    def strain_limit(
            inv_mass: wp.array(dtype=float),
            pos: wp.array(dtype=wp.vec3),
            edge_ids: wp.array2d(dtype=wp.int32),   # [numEdges, 2]
            rest_len: wp.array(dtype=float),
            deltas: wp.array(dtype=wp.vec3)):       # pre-zeroed, applied by add_deltas
        # Hard strain cap (Jacobi): pull the endpoints of any edge longer than
        # maxStrain * rest back toward the cap, mass-weighted, under-relaxed for
        # the shared-vertex Jacobi sum. Engages only past maxStrain (normal solve
        # strain is ~1-2%), so it is inert while draping/resting; under a violent
        # sphere drag it stops the fabric from stretching into the nonphysical
        # 25-100x "wads" that exploded self-collision density and frame time.
        e = wp.tid()
        va = edge_ids[e, 0]
        vb = edge_ids[e, 1]
        wa = inv_mass[va]
        wb = inv_mass[vb]
        wsum = wa + wb
        if wsum == 0.0:
            return
        d = pos[vb] - pos[va]
        L = wp.length(d)
        rl = rest_len[e]
        if L < epsilon:
            return  # fully collapsed: no direction; neighbors resolve it
        Lmax = maxStrain * rl
        Lmin = minStrain * rl
        if L <= Lmax and L >= Lmin:
            return
        n = d / L
        # Step cap: with huge violations (a collider ejection can split an edge
        # 0.5 m across the sphere in one substep), an uncapped Jacobi pull on a
        # hub vertex (up to 8 edges) overshoots and diverges. Capping each edge's
        # per-iteration correction to a fraction of its rest length keeps the sum
        # bounded (~8 * 0.5 * rest * relax per iteration); convergence then comes
        # from iterations x substeps instead of step size.
        if L > Lmax:
            corr = wp.min((L - Lmax) * strainRelax, 0.5 * rl) / wsum
        else:
            # Compression floor: in-plane fabric buckles rather than compresses.
            # Under crushing (dragged grab plowing, sphere folding cloth against
            # the ground) edges collapsed to ~0.1x rest and triangles degenerated
            # -- the "inverted vertices" look -- and a collapsed region has
            # everything within contact range (a self-collision density bomb).
            # Pushing back to Lmin forces a buckle instead. Negative corr flips
            # the displacement direction below.
            corr = -wp.min((Lmin - L) * strainRelax, 0.5 * rl) / wsum
        if wa != 0.0:
            wp.atomic_add(deltas, va, n * (corr * wa))
        if wb != 0.0:
            wp.atomic_add(deltas, vb, -n * (corr * wb))

    @staticmethod
    @wp.kernel
    def ring_floor(
            inv_mass: wp.array(dtype=float),
            pos: wp.array(dtype=wp.vec3),
            pair_ids: wp.array2d(dtype=wp.int32),  # [numRingPairs, 2] static list
            deltas: wp.array(dtype=wp.vec3)):      # shared with strain_limit's pass
        # Anti-fold-through: self-collision deliberately excludes the 2-ring
        # topological neighborhood (it would freeze bending), leaving a blind zone
        # where fabric can fold THROUGH itself at 1-2 cell scale -- flipped
        # triangles ("yellow face over red") and interpenetrated micro-regions
        # whose contact density hammers the broadphase. Enforce the SAME d_offset
        # separation the PDT applies everywhere else, on every non-edge vertex
        # pair within Chebyshev ring distance <= 2 (static list built in __init__;
        # rest distances are 0.015-0.042, all above d_offset, so this only fires
        # at fold-through; mesh edges are excluded -- the strain limiter's
        # compression floor governs those).
        t = wp.tid()
        i = pair_ids[t, 0]
        j = pair_ids[t, 1]
        wa = inv_mass[i]
        wb = inv_mass[j]
        wsum = wa + wb
        if wsum == 0.0:
            return
        d = pos[j] - pos[i]
        L = wp.length(d)
        if L >= d_offset or L < epsilon:
            return
        n = d / L
        corr = -wp.min((d_offset - L) * strainRelax, 0.5 * d_offset) / wsum
        if wa != 0.0:
            wp.atomic_add(deltas, i, n * (corr * wa))
        if wb != 0.0:
            wp.atomic_add(deltas, j, -n * (corr * wb))

    @staticmethod
    @wp.kernel
    def self_contact_gaps(
            mesh: wp.uint64,
            pos: wp.array(dtype=wp.vec3),
            grid_rc: wp.array2d(dtype=wp.int32),
            gaps: wp.array(dtype=float)):
        # DIAGNOSTIC ONLY: per-vertex nearest non-ring vertex-triangle gap on the
        # CURRENT geometry (host analysis of where separations are violated).
        v = wp.tid()
        xv = pos[v]
        r = 2.0 * d_offset
        query = wp.mesh_query_aabb(mesh, xv - wp.vec3(r, r, r), xv + wp.vec3(r, r, r))
        best = float(1.0e30)
        face = wp.int32(0)
        while wp.mesh_query_aabb_next(query, face):
            i0 = wp.mesh_get_index(mesh, 3 * face + 0)
            i1 = wp.mesh_get_index(mesh, 3 * face + 1)
            i2 = wp.mesh_get_index(mesh, 3 * face + 2)
            if (Cloth.within_ring_rc(grid_rc, v, i0)
                    or Cloth.within_ring_rc(grid_rc, v, i1)
                    or Cloth.within_ring_rc(grid_rc, v, i2)):
                continue
            cp = Cloth.closest_point_on_triangle(pos[i0], pos[i1], pos[i2], xv)
            best = wp.min(best, wp.length(xv - cp))
        gaps[v] = best

    @staticmethod
    @wp.kernel
    def detect_crossings(
            grid: wp.uint64,                       # hash grid built on CURRENT pos
            pos: wp.array(dtype=wp.vec3),
            edge_ids: wp.array2d(dtype=wp.int32),  # [numEdges, 2]
            tri_ids: wp.array2d(dtype=wp.int32),   # [numTris, 3]
            grid_rc: wp.array2d(dtype=wp.int32),
            vert_face_off: wp.array(dtype=wp.int32),
            vert_face_ids: wp.array(dtype=wp.int32),
            bounds: wp.array(dtype=float),         # [0] = longest current edge
            pairs: wp.array2d(dtype=wp.int32),     # out: crossed (edge, face)
            count: wp.array(dtype=wp.int32)):      # out: atomic append counter
        # Crossing-resolver DETECTION (once per frame, outside the graph): exact
        # segment-triangle intersections on the CURRENT geometry. A face
        # intersected by this edge has its intersection point q within half the
        # edge length of the edge midpoint, and some face vertex within the
        # longest-edge bound L of q, so a grid walk of half + L from the
        # midpoint reaches a vertex of every intersected face. Each face is
        # tested once: only the minimum-id face vertex IN RANGE expands it.
        # Faces sharing a vertex with the edge are skipped (adjacent geometry
        # cannot legitimately "cross" its own edge), and so are 2-ring
        # material neighbors (see the ring-cull note below -- those belong to
        # SOLVER_RING prevention, not to flipping). Oversized (stretch-wad)
        # primitives are skipped; the resolver targets settled/locked states,
        # not mid-wad transients.
        e = wp.tid()
        va = edge_ids[e, 0]
        vb = edge_ids[e, 1]
        pa = pos[va]
        pb = pos[vb]
        d = pb - pa
        half = 0.5 * wp.length(d)
        if not (half < 0.5 * edgeLenCap):  # oversized or NaN edge: skip
            return
        L = wp.min(bounds[0], edgeLenCap)
        mid = 0.5 * (pa + pb)
        r = half + L + 1.0e-4
        query = wp.hash_grid_query(grid, mid, r)
        u = wp.int32(0)
        while wp.hash_grid_query_next(query, u):
            if wp.length(pos[u] - mid) > r:
                continue
            for k in range(vert_face_off[u], vert_face_off[u + 1]):
                f = vert_face_ids[k]
                i0 = tri_ids[f, 0]
                i1 = tri_ids[f, 1]
                i2 = tri_ids[f, 2]
                if i0 == va or i0 == vb or i1 == va or i1 == vb \
                        or i2 == va or i2 == vb:
                    continue  # shares a vertex with the edge
                # Ring cull (2-ring, matching the metric kernels): a crossing
                # whose partner face is a material neighbor lives inside the
                # constraint skeleton (distance edges, ring_floor pairs) --
                # flipping there fights the constraints and oscillates
                # (measured: WORSE 3003 residue with ring-local flips than
                # without). Those are prevented by SOLVER_RING=1 and relaxed
                # by bending/ring_floor; the resolver handles only true
                # SHEET crossings (ring > 2), the class it reliably fixes.
                if (Cloth.within_ring_rc(grid_rc, va, i0)
                        or Cloth.within_ring_rc(grid_rc, va, i1)
                        or Cloth.within_ring_rc(grid_rc, va, i2)
                        or Cloth.within_ring_rc(grid_rc, vb, i0)
                        or Cloth.within_ring_rc(grid_rc, vb, i1)
                        or Cloth.within_ring_rc(grid_rc, vb, i2)):
                    continue
                # dedup: only the min-id face vertex IN RANGE processes f
                m = u
                if i0 != u and i0 < m and wp.length(pos[i0] - mid) <= r:
                    m = i0
                if i1 != u and i1 < m and wp.length(pos[i1] - mid) <= r:
                    m = i1
                if i2 != u and i2 < m and wp.length(pos[i2] - mid) <= r:
                    m = i2
                if m != u:
                    continue
                # exact Moller-Trumbore segment-triangle test
                a0 = pos[i0]
                e1 = pos[i1] - a0
                e2 = pos[i2] - a0
                h = wp.cross(d, e2)
                det = wp.dot(e1, h)
                if wp.abs(det) < 1.0e-14:
                    continue
                inv = 1.0 / det
                s = pa - a0
                bu = wp.dot(s, h) * inv
                if bu < 0.0 or bu > 1.0:
                    continue
                q = wp.cross(s, e1)
                bv = wp.dot(d, q) * inv
                if bv < 0.0 or bu + bv > 1.0:
                    continue
                t = wp.dot(e2, q) * inv
                if t <= 0.0 or t >= 1.0:
                    continue
                idx = wp.atomic_add(count, 0, 1)
                if idx < maxCross:
                    pairs[idx, 0] = e
                    pairs[idx, 1] = f

    @staticmethod
    @wp.kernel
    def scatter_crossed_flags(
            pairs: wp.array2d(dtype=wp.int32),     # detect_crossings output
            count: wp.array(dtype=wp.int32),
            edge_ids: wp.array2d(dtype=wp.int32),
            tri_ids: wp.array2d(dtype=wp.int32),
            flags: wp.array(dtype=wp.int32)):      # pre-zeroed, 1 = crossed
        # Crossed-flag scatter (see FLAG_GUARD): mark every vertex of every
        # exactly-crossed (edge, face) pair. Plain racing writes of the same
        # value -- order-independent. Runs on device right after
        # detect_crossings so the frame needs no host sync for the flags.
        k = wp.tid()
        if k >= wp.min(count[0], maxCross):
            return
        e = pairs[k, 0]
        f = pairs[k, 1]
        flags[edge_ids[e, 0]] = 1
        flags[edge_ids[e, 1]] = 1
        flags[tri_ids[f, 0]] = 1
        flags[tri_ids[f, 1]] = 1
        flags[tri_ids[f, 2]] = 1

    @staticmethod
    @wp.kernel
    def apply_uncross(
            ids: wp.array(dtype=wp.int32),
            disp: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3)):
        # Crossing-resolver APPLY: per-frame, host-voted uncross displacements
        # (unique vertex ids, magnitudes capped per vertex at the depth-complete
        # bound -- max(uncrossStep, own deepest need), <= uncrossStepMax).
        # Velocity is NOT touched: the next substep's integrate freezes the
        # corrected position into prev_pos, so the flip does not inject
        # kinetic energy.
        k = wp.tid()
        pos[ids[k]] = pos[ids[k]] + disp[k]

    @staticmethod
    @wp.kernel
    def clamp_displacement(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),  # frozen penetration-free reference X
            pos: wp.array(dtype=wp.vec3),
            truncation_ts: wp.array(dtype=float),
            push: wp.array(dtype=wp.vec3),
            push_limit: wp.array(dtype=float)):
        # Displacement governor: bound each free particle's trial displacement
        # |pos - prev_pos| to maxDisplacement, after the XPBD solve and before
        # self-collision. A contract-valid substep travels far below it, so it is a
        # no-op in normal operation; it only sanitizes NaN/Inf and tames an instability
        # spike (XPBD overshoot at ke=1e9, or an accumulated push) so the swept
        # broadphase AABB cannot balloon and the query cannot degenerate to O(tris).
        # prev_pos (the frozen reference) is never modified, so the PDT contract holds.
        i = wp.tid()
        # Prologue: reset the self-collision accumulators for this substep here
        # (unconditional, before the anchor early-out) instead of paying two more
        # fill/zero launches -- clamp_displacement runs before any writer of
        # truncation_ts/push.
        truncation_ts[i] = 1.0
        push[i] = wp.vec3()
        push_limit[i] = 1.0e6  # unbounded until a narrowphase pair tightens it
        if inv_mass[i] == 0.0:
            return  # anchors are host-driven
        dx = pos[i] - prev_pos[i]
        L = wp.length(dx)
        if not (L < 1.0e6):  # NaN/Inf -> snap back to the finite reference state
            pos[i] = prev_pos[i]
        elif L > maxDisplacement:
            pos[i] = prev_pos[i] + dx * (maxDisplacement / L)

    @staticmethod
    @wp.kernel
    def apply_truncation(
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            truncation_ts: wp.array(dtype=float),
            push: wp.array(dtype=wp.vec3),
            push_limit: wp.array(dtype=float),
            center: wp.array(dtype=wp.vec3),   # CURRENT substep sphere center
            radius: wp.array(dtype=float),
            dc_arr: wp.array(dtype=wp.vec3),   # PER-SUBSTEP sphere translation
            dr_arr: wp.array(dtype=float)):    # PER-SUBSTEP radius change
        i = wp.tid()
        if inv_mass[i] == 0.0:
            return  # leave host-driven anchors untouched
        # Bound the accumulated recovery push (see pushClamp): keeps a dense-overlap
        # substep from flinging a particle and pumping unbounded energy. With
        # FAR_GUARD, additionally bound it by the crossing budget (kappa * the
        # tightest near pair's remaining frozen gap): a net push composed
        # across many pairs in a squeeze can no longer carry this vertex
        # across its tightest gap. The leftover budget is handed to
        # fingertip_project's eviction cap (same-substep stream order).
        p = push[i]
        lp = wp.length(p)
        cap = pushClamp
        if FAR_GUARD != 0 and FAR_BUDGET_PUSH != 0:
            # FLOORED budget on the recovery push: the raw cap doubled fuzz
            # fails and stalled pressed piles (recovery starved), while a full
            # exemption let the push-composition creator class (10/74) leak
            # small B-bursts back. Never throttle recovery below half its
            # normal strength; the eviction (45/74 of creators) still gets
            # the fully decremented budget.
            cap = wp.min(cap, wp.max(push_limit[i], 0.5 * pushClamp))
        if lp > cap:
            p = p * (cap / lp)
            lp = cap
        if FAR_GUARD != 0:
            push_limit[i] = wp.max(push_limit[i] - lp, 0.0)
        base = prev_pos[i] + (pos[i] - prev_pos[i]) * truncation_ts[i]
        x = base + p
        # Floor invariant (same rule as add_deltas): the C<0 recovery push may
        # not expel a particle through the ground, nor deepen one already below.
        fb = wp.min(base[1], thickness)
        if x[1] < fb:
            x = wp.vec3(x[0], fb, x[2])
        # Shell invariant: the push may not drive a particle DEEPER into the
        # sphere either. In a crush knot wrapped on the shell, the wad's c<0
        # pushes shove an inside particle inward at up to pushClamp/substep --
        # exactly matching the collider's bounded ejection, a permanent
        # stalemate (measured as persistent post-release sphere penetration).
        # Cancel only the radial-inward component, and only when the result
        # ends up inside the shell and deeper than the pre-push base.
        ce = center[0] + dc_arr[0]
        re = radius[0] + dr_arr[0] + thickness
        dvx = Cloth.radial(x - ce)
        dx2 = wp.dot(dvx, dvx)
        if dx2 < re * re:
            dvb = Cloth.radial(base - ce)
            db = wp.length(dvb)
            if db > epsilon and dx2 < db * db:
                nrad = dvb / db
                inward = wp.dot(x - base, nrad)
                if inward < 0.0:
                    x -= inward * nrad
        pos[i] = x

    @staticmethod
    @wp.kernel
    def collider_project(
            dt: float,
            inv_mass: wp.array(dtype=float),
            prev_pos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            center: wp.array(dtype=wp.vec3),  # CURRENT substep sphere center (advanced in-graph)
            radius: wp.array(dtype=float),    # CURRENT substep sphere radius
            dc_arr: wp.array(dtype=wp.vec3),  # PER-SUBSTEP sphere translation (frame dc / numSubsteps)
            dr_arr: wp.array(dtype=float),    # PER-SUBSTEP radius change (frame dr / numSubsteps)
            dq_arr: wp.array(dtype=wp.quat),  # PER-SUBSTEP rotation (frame dq ^ (1/numSubsteps))
            vt_count: wp.array(dtype=wp.int32)):  # candidate-face counts (stack-pressure proxy)
        # Swept CCD against the moving / growing sphere: the analytic time-of-impact
        # catches approaching particles on the swept surface (no tunneling at any
        # drag speed); a static end-pose contact then resolves already-inside /
        # resting particles and applies depth-based Coulomb friction. Runs after the
        # constraint solve so its penetration-free result has the final say.
        # dc/dr/dq are per-substep slices: the sphere moves the frame delta over the
        # whole frame, so the swept velocity vc = dc/dt is the TRUE sphere speed
        # rather than numSubsteps x too fast (which flung hit particles and blew up).
        # They live in device arrays (set per frame by simulate) so the captured
        # graph stays frame-invariant.
        i = wp.tid()
        if inv_mass[i] == 0.0:
            return

        dc = dc_arr[0]
        dr = dr_arr[0]
        dq = dq_arr[0]

        sc = center[0]   # sphere center/radius at the START of this substep
        sr = radius[0]

        x = pos[i]
        vel_eff = (x - prev_pos[i]) / dt  # cloth substep velocity, before collision

        # Pass 1: swept sphere -> snap approaching particles onto the swept surface.
        hit, c = Cloth.swept_sphere_ccd(prev_pos[i], vel_eff, sc, sr, dc, dr, dt)
        if hit:
            n = wp.normalize(Cloth.radial(c - sc - dc))
            x = c + thickness * n

        # Pass 2: contact at this substep's end pose. The solved position's penetration
        # depth is the normal force signal for friction (rest contacts, where the
        # swept test does not fire because the per-substep travel is sub-margin).
        c_end = sc + dc
        r_end = sr + dr + thickness
        dv = Cloth.radial(x - c_end)
        d = wp.length(dv)
        if d > epsilon and d < r_end:
            n = dv / d
            lambda_n = d - r_end  # < 0 when penetrating
            arm = r_end * n
            # dc/dq are per-substep, so the surface sweeps them over one substep dt.
            v_surf = (dc + wp.quat_rotate(dq, arm) - arm) / dt
            vrel = vel_eff - v_surf
            vt = vrel - wp.dot(n, vrel) * n
            lvt = wp.length(vt)
            # non-penetration: step toward the offset surface, velocity-bounded
            # (projClamp) so a squeezed particle cannot be ejected through a
            # fabric sheet resting above it in a single substep
            x_t = c_end + r_end * n + Cloth.axial(x - c_end)
            # Pile-crush pinch: a mostly-DOWNWARD radial ejection of a particle
            # under real stacking pressure (vt_count) drives it through every
            # pancaked layer between the shell bottom and the floor, one
            # projClamp step per substep -- the descending shell "eats" the top
            # of the stack and spits it through the rest (the crossing pump
            # behind permanent post-crush entanglement). Exit horizontally at
            # the particle's own height instead: the lateral shell boundary at
            # that layer -- parallel to the squeezed sheets, so the ejection
            # cannot cross them, and it doubles as the pinch extrusion pump.
            # Depth gate: only the shallow, per-substep "eaten" band exits
            # laterally; a deeply contained particle takes the radial route so
            # sphere non-penetration always converges (a pure-lateral exit can
            # be meters long and starves it -- measured as persistent
            # post-release sphere penetration).
            # (sphere-crush machinery: compiled out for the cylinder kind, whose
            # scenarios never park the collider on a floor pile)
            if COLLIDER_KIND == 0 and n[1] < -0.5 and -lambda_n < 2.0 * d_offset \
                    and vt_count[i] >= extrudePressure:
                dyv = c_end[1] - x[1]
                rr = r_end * r_end - dyv * dyv
                hx = x[0] - c_end[0]
                hz = x[2] - c_end[2]
                hd2 = hx * hx + hz * hz
                if rr > 0.0 and hd2 > epsilon:
                    rho = wp.sqrt(rr / hd2)
                    x_t = wp.vec3(c_end[0] + rho * hx, x[1], c_end[2] + rho * hz)
            step = x_t - x
            sl = wp.length(step)
            if sl > projClamp:
                step *= projClamp / sl
            x = x + step
            if lvt > epsilon:
                lambda_f = wp.max(0.4 * lambda_n, -lvt * dt)
                x += (vt / lvt) * lambda_f

        # Ground plane at y = thickness, with friction (bounded snap: same
        # through-fabric teleport hazard as the sphere ejection, at the pile).
        if x[1] < thickness:
            n = Ground.NORMAL
            lambda_n = x[1] - thickness  # < 0 below ground
            vt = vel_eff - wp.dot(n, vel_eff) * n
            lvt = wp.length(vt)
            x = wp.vec3(x[0], wp.min(thickness, x[1] + projClamp), x[2])
            if lvt > epsilon:
                lambda_f = wp.max(0.65 * lambda_n, -lvt * dt)
                x += (vt / lvt) * lambda_f

        pos[i] = x

    @staticmethod
    @wp.kernel
    def collider_project_edges(
            inv_mass: wp.array(dtype=float),
            pos: wp.array(dtype=wp.vec3),
            edge_ids: wp.array2d(dtype=wp.int32),  # [numEdges, 2]
            center: wp.array(dtype=wp.vec3),       # CURRENT substep sphere center
            radius: wp.array(dtype=float),
            dc_arr: wp.array(dtype=wp.vec3),  # PER-SUBSTEP sphere translation
            dr_arr: wp.array(dtype=float),    # PER-SUBSTEP radius change
            deltas: wp.array(dtype=wp.vec3)):      # pre-zeroed, applied by add_deltas
        # Edge-vs-sphere non-penetration: the vertex pass (collider_project) keeps every
        # PARTICLE out of the sphere, but under load the cloth can stretch until edges
        # span many times the particle spacing -- then the sphere passes BETWEEN
        # particles, through a stretched edge, with every vertex clear (fast lateral
        # drag through a large draped cloth: edges reached 100x rest length and the
        # sphere tunneled through the fabric). Project the closest point of each EDGE
        # segment out of the sphere, pushing the endpoints by their barycentric weights.
        # At rest this only trims the ~1e-4 chord sag between projected vertices.
        e = wp.tid()
        va = edge_ids[e, 0]
        vb = edge_ids[e, 1]
        wa = inv_mass[va]
        wb = inv_mass[vb]
        if wa == 0.0 and wb == 0.0:
            return

        dc = dc_arr[0]
        dr = dr_arr[0]
        c_end = center[0] + dc            # same end pose as collider_project's Pass 2
        r_end = radius[0] + dr + thickness

        pa = pos[va]
        pb = pos[vb]
        ab = pb - pa
        # Closest point of the segment in the collider's RADIAL metric (identity
        # for the sphere; xy for the cylinder, where an axis-parallel edge keeps
        # t = 0 and the vertex pass covers its endpoints).
        abr = Cloth.radial(ab)
        ab2 = wp.dot(abr, abr)
        t = 0.0
        if ab2 > epsilon:
            t = wp.clamp(wp.dot(Cloth.radial(c_end - pa), abr) / ab2, 0.0, 1.0)
        cp = pa + t * ab
        dv = Cloth.radial(cp - c_end)
        d = wp.length(dv)
        if d <= epsilon or d >= r_end:
            return
        n = dv / d
        depth = r_end - d
        # Pile-crush pinch (same redirect as collider_project pass 2): when the
        # shell is parked near the floor, a downward radial ejection drives the
        # edge through the pancaked stack below. Exit laterally at the contact
        # point's own height instead.
        # (sphere-crush machinery: compiled out for the cylinder kind)
        if COLLIDER_KIND == 0 and n[1] < -0.5 and depth < 2.0 * d_offset \
                and c_end[1] - r_end < thickness + extrudeBand:
            dyv = c_end[1] - cp[1]
            rr = r_end * r_end - dyv * dyv
            hx = cp[0] - c_end[0]
            hz = cp[2] - c_end[2]
            hd2 = hx * hx + hz * hz
            if rr > 0.0 and hd2 > epsilon:
                rho = wp.sqrt(rr / hd2)
                x_t = wp.vec3(c_end[0] + rho * hx, cp[1], c_end[2] + rho * hz)
                dl = x_t - cp
                dll = wp.length(dl)
                if dll > epsilon:
                    n = dl / dll
                    depth = dll

        # Move the contact point out by `depth` along n with the minimum-norm endpoint
        # displacements (weights 1-t and t), skipping pinned endpoints. Accumulated via
        # deltas + add_deltas so concurrent edges sharing a vertex read consistent pos;
        # any over-push from summing neighbors points AWAY from the sphere (safe) and
        # only occurs in the already-pathological stretched state.
        w = 0.0
        if wa != 0.0:
            w += (1.0 - t) * (1.0 - t)
        if wb != 0.0:
            w += t * t
        if w < epsilon:
            return
        # w >= 0.5 with both endpoints free; it only approaches 0 when the contact sits
        # next to a pinned endpoint -- clamp so the free endpoint's push stays bounded
        # (the remaining gap resolves over subsequent substeps).
        s = wp.min(depth, projClamp) / wp.max(w, 0.1)
        if wa != 0.0:
            wp.atomic_add(deltas, va, (1.0 - t) * s * n)
        if wb != 0.0:
            wp.atomic_add(deltas, vb, t * s * n)

    @staticmethod
    @wp.kernel
    def advance_grab_anchors(anchor_pos: wp.array(dtype=wp.vec3),
                             anchor_delta: wp.array(dtype=wp.vec3)):
        # One substep slice of each grab's frame motion (runs BEFORE apply, seeded
        # at the previous frame's committed anchor point, so substep k places the
        # patch at prev + (k+1)*delta and the last substep lands on the target).
        i = wp.tid()
        anchor_pos[i] = anchor_pos[i] + anchor_delta[i]

    @staticmethod
    @wp.kernel
    def apply_grab_anchors(count: wp.array(dtype=wp.int32),
                           member_ids: wp.array(dtype=wp.int32),
                           member_ax: wp.array(dtype=wp.int32),
                           member_off: wp.array(dtype=wp.vec3),
                           anchor_pos: wp.array(dtype=wp.vec3),
                           pos: wp.array(dtype=wp.vec3)):
        # Place every grabbed (pinned) particle at its grab's CURRENT substep
        # anchor point + offset. Runs right after integrate: prev_pos froze the
        # pre-move position, so this substep's patch motion is a proper swept
        # displacement that the self-collision broadphase and truncation see --
        # the patch plows fabric instead of teleporting through it.
        t = wp.tid()
        if t >= count[0]:
            return
        pos[member_ids[t]] = anchor_pos[member_ax[t]] + member_off[t]

    @staticmethod
    @wp.kernel
    def fingertip_project(inv_mass: wp.array(dtype=float),
                          n_grabs: wp.array(dtype=wp.int32),
                          anchor_pos: wp.array(dtype=wp.vec3),
                          anchor_delta: wp.array(dtype=wp.vec3),
                          prev_pos: wp.array(dtype=wp.vec3),
                          pos: wp.array(dtype=wp.vec3),
                          push_limit: wp.array(dtype=float)):
        # Project free fabric out of each active grab's fingertip ball (see
        # FINGER_COLLIDER). Velocity-bounded (projClamp) like the sphere's
        # pass 2: a fast grip indents fabric transiently instead of
        # teleporting it through neighboring sheets.
        if FINGER_COLLIDER == 0:
            return
        i = wp.tid()
        if inv_mass[i] == 0.0:
            return
        rr = fingerColliderR * fingerRadius
        rc = fingerCcdR * fingerRadius
        x = pos[i]
        for g in range(n_grabs[0]):
            b1 = anchor_pos[g]
            if FINGER_CCD != 0:
                # time-of-impact of the particle's substep segment vs the
                # moving ball (both start-of-substep poses reconstructed);
                # place a hit on the END-pose shell at the contact normal.
                # Radius rc (>= the volumetric rr): entering trajectories only
                # -- taut sheets that cannot comply with the bounded eviction
                # are caught at CCD scale without a steady-state guard volume.
                b0 = b1 - anchor_delta[g]
                p0 = prev_pos[i]
                sv = p0 - b0
                c = wp.dot(sv, sv) - rc * rc
                if c > 0.0:
                    v = (x - p0) - anchor_delta[g]
                    a = wp.dot(v, v)
                    b = wp.dot(v, sv)
                    if a > 1.0e-12 and b < 0.0:
                        disc = b * b - a * c
                        if disc > 0.0:
                            t = (-b - wp.sqrt(disc)) / a
                            if 0.0 <= t and t <= 1.0:
                                bc = b0 + t * anchor_delta[g]
                                n = wp.normalize((p0 + t * (x - p0)) - bc)
                                x = b1 + n * rc
            r = x - b1
            d = wp.length(r)
            if d < 1.0e-9 or d >= rr:
                continue
            step = wp.min(rr - d, fingerColliderPush * projClamp)
            if FAR_GUARD != 0:
                # Crossing budget (see FAR_GUARD): the eviction step may not
                # exceed the leftover budget of this vertex's tightest near
                # pair -- an eviction can no longer punch through a sheet
                # closer than the step. Vertices with no near pair keep the
                # full eviction speed (budget resets to 1e6 each substep), so
                # onset grip clearance is unchanged. The CCD placement above
                # stays uncapped (taut-sheet anti-tunnel).
                step = wp.min(step, push_limit[i])
            x = x + (r / d) * step
        pos[i] = x

    @staticmethod
    @wp.kernel
    def accumulate_grab_pressure(count: wp.array(dtype=wp.int32),
                                 member_ids: wp.array(dtype=wp.int32),
                                 member_ax: wp.array(dtype=wp.int32),
                                 push: wp.array(dtype=wp.vec3),
                                 pressure: wp.array(dtype=wp.vec3),
                                 pressure_mag: wp.array(dtype=float)):
        # Load-yielding grip: sum the pressure shares the narrowphase recorded on
        # this substep's grabbed members (their push[] slots are otherwise unused
        # -- apply_truncation skips inv_mass==0) into per-grab accumulators. Runs
        # inside the captured graph right after the narrowphase, before next
        # substep's clamp_displacement re-zeroes push.
        t = wp.tid()
        if t >= count[0]:
            return
        p = push[member_ids[t]]
        m = wp.length(p)
        if m > 0.0:
            g = member_ax[t]
            wp.atomic_add(pressure, g, p)
            wp.atomic_add(pressure_mag, g, m)

    @staticmethod
    @wp.kernel
    def advance_sphere(center: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float),
                       dc_arr: wp.array(dtype=wp.vec3), dr_arr: wp.array(dtype=float)):
        # Advance the collider pose by one substep slice. Runs (in the captured graph)
        # AFTER collider_project each substep, so replay k sees center_0 + k*dc and the
        # last substep lands exactly on the rendered end pose center_0 + numSubsteps*dc.
        center[0] = center[0] + dc_arr[0]
        radius[0] = radius[0] + dr_arr[0]

    @staticmethod
    @wp.kernel
    # copied from https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/xpbd/kernels.py
    def distance_constraints(
            dt: float,
            ke: float,
            kd: float,
            relaxation: float,
            offset: wp.int32,
            pos: wp.array(dtype=wp.vec3),
            prev_pos: wp.array(dtype=wp.vec3),
            inv_mass: wp.array(dtype=float),
            indices: wp.array2d(dtype=int),
            rest_lengths: wp.array(dtype=float),
            lambdas: wp.array(dtype=float),
            deltas: wp.array(dtype=wp.vec3),
    ):
        tid = offset + wp.tid()

        i = indices[tid, 0]
        j = indices[tid, 1]

        rest = rest_lengths[tid]

        xi = pos[i]
        xj = pos[j]

        pi = prev_pos[i]
        pj = prev_pos[j]

        xij = xi - xj
        # Substep displacement difference: the XPBD damping term is
        # gamma * grad_c . (x - x^n) (Macklin et al., eq. 26; newton uses
        # dt * grad_c . (vi - vj), identical since v = (x - x^n)/dt). Using
        # previous POSITIONS here (the old `vij = pi - pj`) is not a damping
        # term at all -- it is a constant ~rest-length bias scaled by
        # gamma = kd/(ke*dt), i.e. a spurious dt-DEPENDENT compression offset
        # with zero dissipation.
        vij = (xi - pi) - (xj - pj)

        l = wp.length(xij)
        if l < epsilon:
            return

        n = xij / l

        c = l - rest
        grad_c_xi = n
        grad_c_xj = -1.0 * n

        wi = inv_mass[i]
        wj = inv_mass[j]

        denom = wi + wj
        if denom == 0.0:
            return

        alpha = 1.0 / (ke * dt * dt)
        gamma = kd / (ke * dt)

        grad_c_dot_v = wp.dot(grad_c_xi, vij)
        dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_c_dot_v) / ((1.0 + gamma) * denom + alpha)

        dxi = wi * dlambda * grad_c_xi
        dxj = wj * dlambda * grad_c_xj

        lambdas[tid] = lambdas[tid] + dlambda

        wp.atomic_add(deltas, i, dxi * relaxation)
        wp.atomic_add(deltas, j, dxj * relaxation)

    @staticmethod
    @wp.kernel
    # copied from https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/xpbd/kernels.py
    def bending_constraints(
            dt: float,
            ke: float,
            kd: float,
            relaxation: float,
            offset: wp.int32,
            pos: wp.array(dtype=wp.vec3),
            prev_pos: wp.array(dtype=wp.vec3),
            inv_mass: wp.array(dtype=float),
            indices: wp.array2d(dtype=int),
            rest_angles: wp.array(dtype=float),
            lambdas: wp.array(dtype=float),
            deltas: wp.array(dtype=wp.vec3),
    ):
        tid = offset + wp.tid()

        # The edge lies between the particles indexed by 'k' and 'l',
        # and the two connected triangles with counter-clockwise winding: (i, k, l), (j, l, k).
        i = indices[tid, 0]
        j = indices[tid, 1]
        k = indices[tid, 2]
        l = indices[tid, 3]

        rest_angle = rest_angles[tid]

        x1 = pos[i]
        x2 = pos[j]
        x3 = pos[k]
        x4 = pos[l]

        p1 = prev_pos[i]
        p2 = prev_pos[j]
        p3 = prev_pos[k]
        p4 = prev_pos[l]

        w1 = inv_mass[i]
        w2 = inv_mass[j]
        w3 = inv_mass[k]
        w4 = inv_mass[l]

        n1 = wp.cross(x3 - x1, x4 - x1)  # normal to face 1
        n2 = wp.cross(x4 - x2, x3 - x2)  # normal to face 2

        n1_length = wp.length(n1)
        n2_length = wp.length(n2)

        if n1_length < epsilon or n2_length < epsilon:
            return

        n1 /= n1_length
        n2 /= n2_length

        # Clamp to [-1, 1]: dot of two unit normals can round just past 1.0 for
        # near-coplanar faces (creases during crumpling), and CUDA acosf(|x|>1)
        # returns NaN, which then spreads through deltas -> pos -> everything.
        cos_theta = wp.clamp(wp.dot(n1, n2), -1.0, 1.0)

        e = x4 - x3
        e_hat = wp.normalize(e)
        e_length = wp.length(e)

        derivative_flip = wp.sign(wp.dot(wp.cross(n1, n2), e))
        derivative_flip *= -1.0
        angle = wp.acos(cos_theta)

        grad_x1 = n1 * e_length * derivative_flip
        grad_x2 = n2 * e_length * derivative_flip
        grad_x3 = (n1 * wp.dot(x1 - x4, e_hat) + n2 * wp.dot(x2 - x4, e_hat)) * derivative_flip
        grad_x4 = (n1 * wp.dot(x3 - x1, e_hat) + n2 * wp.dot(x3 - x2, e_hat)) * derivative_flip
        c = angle - rest_angle
        denominator = (
                w1 * wp.length_sq(grad_x1)
                + w2 * wp.length_sq(grad_x2)
                + w3 * wp.length_sq(grad_x3)
                + w4 * wp.length_sq(grad_x4)
        )

        if denominator <= epsilon:
            return

        alpha = 1.0 / (ke * dt * dt)
        gamma = kd / (ke * dt)

        # XPBD damping: gamma * grad . (x - x^n), the substep DISPLACEMENT
        # (newton: dt * grad . v). The old form dotted the gradients with the
        # absolute previous POSITIONS -- translation-variant, zero actual
        # dissipation, and (scaled by gamma = kd/(ke*dt)) a bias that grows as
        # substeps drop. With the real term, kd damps the dihedral-angle rate:
        # this is what suppresses the residual bending oscillation ("undulation")
        # at lower substep counts, with the physically correct dt scaling.
        grad_dot_v = (wp.dot(grad_x1, x1 - p1) + wp.dot(grad_x2, x2 - p2)
                      + wp.dot(grad_x3, x3 - p3) + wp.dot(grad_x4, x4 - p4))
        dlambda = -1.0 * (c + alpha * lambdas[tid] + gamma * grad_dot_v) / ((1.0 + gamma) * denominator + alpha)

        delta0 = w1 * dlambda * grad_x1
        delta1 = w2 * dlambda * grad_x2
        delta2 = w3 * dlambda * grad_x3
        delta3 = w4 * dlambda * grad_x4

        lambdas[tid] = lambdas[tid] + dlambda

        wp.atomic_add(deltas, i, delta0 * relaxation)
        wp.atomic_add(deltas, j, delta1 * relaxation)
        wp.atomic_add(deltas, k, delta2 * relaxation)
        wp.atomic_add(deltas, l, delta3 * relaxation)

    @staticmethod
    @wp.kernel
    def add_deltas(
            pos: wp.array(dtype=wp.vec3),
            deltas: wp.array(dtype=wp.vec3)):
        # Applies AND re-zeroes: deltas is allocated zero and every producer only
        # atomic_adds, with add_deltas the sole consumer after each producer -- so
        # re-zeroing here keeps the all-zero-between-uses invariant and removes a
        # separate deltas.zero_() launch per iteration (19 launches/substep saved
        # across the solve, strain-limit and collider-edge blocks).
        tid = wp.tid()
        x = pos[tid] + deltas[tid]
        # Floor invariant: internal corrections may not push a particle through
        # the ground (from above y=thickness to below), nor deepen one already
        # below (a falling-cloth dip from integrate). Under a sphere-onto-pile
        # crush the summed downward corrections otherwise out-shove the single
        # bounded ground snap and expel the bottom layers below the floor,
        # crossing every layer on the way (permanent entanglement).
        fb = wp.min(pos[tid][1], thickness)
        if x[1] < fb:
            x = wp.vec3(x[0], fb, x[2])
        pos[tid] = x
        deltas[tid] = wp.vec3()

    @staticmethod
    @wp.kernel
    def update_velocity(
            dt: float,
            pos: wp.array(dtype=wp.vec3),
            prev_pos: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3)):
        tid = wp.tid()

        # pos is already collision-resolved (truncation + collider projection),
        # so velocity simply reflects the committed displacement.
        v = (pos[tid] - prev_pos[tid]) / dt
        mag = wp.length(v)
        # Sanitize first: the old one-sided `mag > maxVelocity` let NaN/Inf through
        # (IEEE comparisons with NaN are false), latching a dead particle forever.
        # Zeroing the velocity lets the sim self-heal (pos itself is repaired by the
        # displacement governor's isfinite reset before the next substep's solve).
        if not (mag < 1.0e6):
            v = wp.vec3(0.0, 0.0, 0.0)
        elif mag > maxVelocity:
            v *= maxVelocity / mag
        vel[tid] = v

    @staticmethod
    @wp.kernel
    def cast_ray(origin: wp.vec3,
                 direction: wp.vec3,
                 pos: wp.array(dtype=wp.vec3),
                 tri_ids: wp.array2d(dtype=wp.int32),
                 min_dist: wp.array(dtype=float)):
        # Picking, pass 1: brute-force Moller-Trumbore over all triangles,
        # atomic_min of the nearest hit distance. Runs once per CLICK (not per
        # frame), so a spatial structure is unnecessary -- this replaced the
        # wp.Mesh ray query and with it the last reason to keep (and refit) an
        # LBVH in the simulation loop.
        f = wp.tid()
        a = pos[tri_ids[f, 0]]
        e1 = pos[tri_ids[f, 1]] - a
        e2 = pos[tri_ids[f, 2]] - a
        h = wp.cross(direction, e2)
        det = wp.dot(e1, h)
        if wp.abs(det) < 1.0e-12:
            return
        inv = 1.0 / det
        s = origin - a
        u = wp.dot(s, h) * inv
        if u < 0.0 or u > 1.0:
            return
        q = wp.cross(s, e1)
        v = wp.dot(direction, q) * inv
        if v < 0.0 or u + v > 1.0:
            return
        t = wp.dot(e2, q) * inv
        if t > 1.0e-6:
            wp.atomic_min(min_dist, 0, t)

    @staticmethod
    @wp.kernel
    def cast_ray_face(origin: wp.vec3,
                      direction: wp.vec3,
                      pos: wp.array(dtype=wp.vec3),
                      tri_ids: wp.array2d(dtype=wp.int32),
                      min_dist: wp.array(dtype=float),
                      face_out: wp.array(dtype=wp.int32)):
        # Picking, pass 2: re-test and claim the face matching the winning
        # distance (ties race benignly -- any coincident face is a valid pick).
        f = wp.tid()
        d = min_dist[0]
        if d >= 1.0e6:
            return
        a = pos[tri_ids[f, 0]]
        e1 = pos[tri_ids[f, 1]] - a
        e2 = pos[tri_ids[f, 2]] - a
        h = wp.cross(direction, e2)
        det = wp.dot(e1, h)
        if wp.abs(det) < 1.0e-12:
            return
        inv = 1.0 / det
        s = origin - a
        u = wp.dot(s, h) * inv
        if u < 0.0 or u > 1.0:
            return
        q = wp.cross(s, e1)
        v = wp.dot(direction, q) * inv
        if v < 0.0 or u + v > 1.0:
            return
        t = wp.dot(e2, q) * inv
        if wp.abs(t - d) < 1.0e-6:
            face_out[0] = f

    def drag_anchor(self, screen_x, screen_y) -> Optional[Particle]:
        origin, direction = ray_from_screen(screen_x, screen_y)

        # Find the closest locked anchor hit by the ray
        near_anchor: Optional[Particle] = None
        min_anchor_dist = None
        host_pos = self.hostPos.numpy()
        for anchor in filter(lambda a: a.flags & AnchorFlag.LOCKED, self.anchors):
            dist = ray_to_sphere(origin, direction, wp.vec3f(host_pos[anchor.id]), 0.08)
            if dist and (not min_anchor_dist or dist[0] < min_anchor_dist):
                min_anchor_dist = dist[0]
                near_anchor = anchor

        # Find the closest triangle hit by the ray (brute force, click-time only)
        self._pickDist.fill_(1.0e6)
        self._pickFace.fill_(-1)
        wp.launch(kernel=Cloth.cast_ray,
                  dim=self.numTris,
                  inputs=[origin, direction, self.pos, self.triIds],
                  outputs=[self._pickDist])
        wp.launch(kernel=Cloth.cast_ray_face,
                  dim=self.numTris,
                  inputs=[origin, direction, self.pos, self.triIds, self._pickDist],
                  outputs=[self._pickFace])
        tri_id = int(self._pickFace.numpy()[0])
        min_tri_dist = float(self._pickDist.numpy()[0])
        hit = tri_id >= 0

        # Ids already claimed by another anchor (primary or patch member) are off
        # limits: their hostInvMass currently reads 0.0, so grabbing one would
        # record mass=0.0 and release would restore it as a permanent invisible pin.
        claimed = {a.id for a in self.anchors}
        for a in self.anchors:
            claimed.update(mid for mid, _, _ in a.group)

        # Resolve the pressed particle + its ray depth. A triangle hit grabs the
        # first unclaimed vertex of the hit face. Otherwise (or if the whole face
        # is claimed) fall back to the FINGERTIP-TOLERANT pick: the pixel ray is
        # infinitesimal, so a press on the visible EDGE of fabric -- the corner
        # of a floor pile, exactly what a user grabs to flatten it -- can graze
        # past every triangle by ~2 mm and returned no anchor ("the drag is
        # inoperant"). A finger pad has area: take the frontmost unclaimed
        # particle whose distance to the ray is within fingerRadius (smallest
        # ray depth, with the ray distance as a mild tiebreak toward what sits
        # under the press point). Occluded deeper layers lie further ALONG the
        # ray, so the frontmost candidate is the visible fabric.
        particle_id = -1
        grab_depth = 0.0
        if hit:
            for k in range(3):
                cand = self.hostTriIds[tri_id, k].item()
                if cand not in claimed:
                    particle_id = cand
                    grab_depth = min_tri_dist
                    break
        if particle_id < 0 and grabPickTolerant:
            o = np.array([origin[0], origin[1], origin[2]])
            dn = np.array([direction[0], direction[1], direction[2]])
            rel = host_pos - o
            t = rel @ dn
            perp2 = np.einsum('ij,ij->i', rel, rel) - t * t
            cand = (t > 0.05) & (perp2 < fingerRadius * fingerRadius)
            if claimed:
                cand[np.fromiter(claimed, dtype=np.int64)] = False
            idx = np.nonzero(cand)[0]
            if len(idx):
                score = t[idx] + 2.0 * np.sqrt(np.maximum(perp2[idx], 0.0))
                j = int(idx[np.argmin(score)])
                particle_id = j
                grab_depth = float(t[j])
        if particle_id < 0 and not min_anchor_dist:
            return None

        # Check if the collider is hit by the ray before any hit anchor or fabric
        dist = ray_to_collider(origin, direction, sphere.center, sphere.radius)
        if dist and (not near_anchor or dist[0] < min_anchor_dist) \
                and (particle_id < 0 or dist[0] < grab_depth):
            return None

        # Check if the hit locked anchor is closer than the hit fabric
        if near_anchor and (particle_id < 0 or min_anchor_dist <= grab_depth):
            near_anchor.flags |= AnchorFlag.ACTIVE
            near_anchor.depth = min_anchor_dist
            near_anchor.screen = wp.vec2(screen_x, screen_y)
            return near_anchor.drag()
        if particle_id < 0:
            return None

        inv_mass = self.hostInvMass.numpy()
        anchor = Particle(id=particle_id,
                          screen=wp.vec2(screen_x, screen_y),
                          mass=inv_mass[particle_id].item(),
                          depth=grab_depth)

        # Fingertip grab: pin every particle within fingerRadius of the picked one
        # and drag them as a conforming patch (world offsets from the primary at
        # grab time; see update_anchors), skipping claimed ids.
        anchor.group = self._fingertip_patch(particle_id, host_pos, inv_mass, claimed)

        # Onset brake pre-arm (see GRAB_BRAKE / grabBrakeOnset): the committed
        # first target can sit a patch-width away from the raw pick (depth /
        # shell / stack rules), so a newborn grab otherwise sweeps at the full
        # 0.45*d_offset/substep clamp through whatever is interleaved in the
        # grip -- at 400^2 pile density that closure sweep IS the onset
        # crossing burst (80-110 pairs in the first 2 frames, before the
        # crossing brake has any signal). Start the brake hold non-zero: the
        # first frames ramp in gently, and the hold decays to free authority
        # within ~10 frames unless the sweep reports real crossings.
        anchor.brake_hold = grabBrakeOnset

        self.anchors.append(anchor)
        return anchor.drag()

    def _fingertip_patch(self, particle_id, host_pos, inv_mass, claimed):
        """Membership of a fingertip grab centered on particle_id: every particle
        within fingerRadius, restricted to the SAME SHEET. A Euclidean ball also
        captures occluded layers folded behind the visible one (grabbing through
        the fabric); the grid ring distance <= fingerRadius/spacing filter is
        topological (same layer) by construction. Returns the group list of
        (member id, original inv-mass, world offset from the primary)."""
        center = host_pos[particle_id]
        d2 = np.einsum('ij,ij->i', host_pos - center, host_pos - center)
        near = d2 < fingerRadius * fingerRadius
        ring = int(np.ceil(fingerRadius / self.spacing))
        rows = np.arange(self.numParticles) // self.numCols
        cols = np.arange(self.numParticles) % self.numCols
        pr, pc = particle_id // self.numCols, particle_id % self.numCols
        near &= (np.abs(rows - pr) <= ring) & (np.abs(cols - pc) <= ring)
        group = []
        for mid in np.nonzero(near)[0]:
            mid = int(mid)
            if mid == particle_id or mid in claimed:
                continue
            off = host_pos[mid] - center
            group.append((mid, inv_mass[mid].item(),
                          wp.vec3f(off[0], off[1], off[2])))
        return group

    def _build_rim_pairs(self, member_lists, inv_mass):
        """Rim pair lists for the rim_truncate_* kernels (see RIM_SOLVER):
        for each active grab's member set M on the grid, with S1 = free ring-1
        skirt and zone = ring<=2 dilation of M,
          VT: (v, face) with v in M|S1, face inside the zone touching M|S1,
              v not in face, grid ring(v, face) == 1, >= 1 free vertex;
          EE: unordered edge pairs, both edges inside the zone touching M|S1,
              disjoint, ring(e1, e2) == 1, >= 1 free endpoint.
        Rest ring-1 gaps are >= spacing*sqrt(2)/2 (3.3x rimDOffset), so these
        pairs are inert until a cell kinks to sub-half-fabric-thickness.
        Vectorized numpy; runs only when grab membership changes."""
        nr = self.numParticles // self.numCols
        nc = self.numCols
        T = self.hostTriIds
        E = self.hostEdgeIds
        vt_chunks, ee_chunks = [], []
        for mem in member_lists:
            mask = np.zeros((nr, nc), dtype=bool)
            mask[mem // nc, mem % nc] = True

            def dilate(m):
                # 3x3 (Chebyshev ring-1) dilation; every |= reads from a copy,
                # never an overlapping view (in-place shifted ORs can cascade)
                row = m.copy()
                row[:-1] |= m[1:]; row[1:] |= m[:-1]
                out = row.copy()
                out[:, :-1] |= row[:, 1:]; out[:, 1:] |= row[:, :-1]
                return out

            d1 = dilate(mask)   # 3x3 dilate: ring <= 1
            d2 = dilate(d1)     # 5x5 dilate: ring <= 2
            core = d1.reshape(-1)                  # M | S1 (ring <= 1)
            zone = d2.reshape(-1)                  # ring <= 2
            core_ids = np.nonzero(core)[0].astype(np.int64)
            rcv = np.stack([core_ids // nc, core_ids % nc], axis=1)
            free = inv_mass > 0.0
            # --- VT: core vertices x zone faces at ring exactly 1 ---
            fsel = np.nonzero(zone[T].all(axis=1) & core[T].any(axis=1))[0]
            if len(fsel) and len(core_ids):
                FT = T[fsel].astype(np.int64)               # [F, 3]
                rcf = np.stack([FT // nc, FT % nc], axis=2)  # [F, 3, 2]
                ring = np.abs(rcv[:, None, None, :] - rcf[None, :, :, :]) \
                    .max(axis=3).min(axis=2)                 # [K, F]
                contains = (FT[None, :, :] == core_ids[:, None, None]).any(axis=2)
                anyfree = free[core_ids][:, None] | free[FT].any(axis=1)[None, :]
                ki, fi = np.nonzero((ring == 1) & ~contains & anyfree)
                if len(ki):
                    vt_chunks.append(np.column_stack(
                        [core_ids[ki], FT[fi]]).astype(np.int32))
            # --- EE: zone edge pairs at ring exactly 1 ---
            esel = np.nonzero(zone[E].all(axis=1) & core[E].any(axis=1))[0]
            if len(esel) > 1:
                EZ = E[esel].astype(np.int64)                # [n, 2]
                rce = np.stack([EZ // nc, EZ % nc], axis=2)  # [n, 2, 2]
                ring = np.abs(rce[:, None, :, None, :] - rce[None, :, None, :, :]) \
                    .max(axis=4).reshape(len(EZ), len(EZ), 4).min(axis=2)
                shared = (EZ[:, None, :, None] == EZ[None, :, None, :]).any(axis=(2, 3))
                anyfree = free[EZ].any(axis=1)
                pair_ok = (ring == 1) & ~shared \
                    & (anyfree[:, None] | anyfree[None, :])
                ii, jj = np.nonzero(np.triu(pair_ok, k=1))   # each unordered pair once
                if len(ii):
                    ee_chunks.append(np.column_stack(
                        [EZ[ii], EZ[jj]]).astype(np.int32))
        vt = np.concatenate(vt_chunks, axis=0) if vt_chunks \
            else np.zeros((0, 4), np.int32)
        ee = np.concatenate(ee_chunks, axis=0) if ee_chunks \
            else np.zeros((0, 4), np.int32)
        # overlapping grabs could stage the same pair twice (double-counted
        # atomic pushes) -- dedup rows
        if len(member_lists) > 1:
            if len(vt):
                vt = np.unique(vt, axis=0)
            if len(ee):
                ee = np.unique(ee, axis=0)
        return vt, ee

    def _slide_grab(self, particle, pointer_np, pos, inv_mass, origin, direction):
        """Sliding grab: the committed anchor target has chronically lagged the
        commanded pointer-ray target (yield stall against a snagged patch, or
        any obstruction), so slip the grip over the fabric like a finger pad:
        release the current members (inv-mass restored through the normal
        bookkeeping, exactly once), re-grab a fingertip patch around the vertex
        nearest to a point stepped from the current anchor toward the pointer,
        refresh the anchor's id/group/depth, and continue the same stroke.
        prev_target seeds at the NEW grab's own current position -- never the
        old anchor's -- so the per-substep sweep sees no teleport."""
        # 1) release the old patch: restore primary + member inv-mass
        inv_mass[particle.id] = particle.mass
        for mid, mmass, _ in particle.group:
            inv_mass[mid] = mmass
        # 2) step one fingertip from the committed anchor toward the pointer
        dvec = pointer_np - particle.prev_target
        dlen = float(np.linalg.norm(dvec))
        probe = particle.prev_target + (dvec * (min(grabSlideStep, dlen) / dlen)
                                        if dlen > 1e-9 else 0.0)
        # 3) re-pick: nearest particle to the probe, other anchors' claims excluded
        #    (our own just-released members are fair game -- the grip may slide
        #    only partway off its old patch)
        claimed = set()
        for a in self.anchors:
            if a is particle:
                continue
            claimed.add(a.id)
            claimed.update(mid for mid, _, _ in a.group)
        d2 = np.einsum('ij,ij->i', pos - probe, pos - probe)
        if claimed:
            d2[np.fromiter(claimed, dtype=np.int64)] = np.inf
        new_id = int(np.argmin(d2))
        # 4) refresh identity + fingertip patch around the new primary
        particle.id = new_id
        particle.mass = inv_mass[new_id].item()
        particle.group = self._fingertip_patch(new_id, pos, inv_mass, claimed)
        inv_mass[new_id] = 0.0
        # 5) stroke continuity: pointer-ray depth of the new grab point, and the
        #    sweep origin at the new grab's OWN position (no teleport)
        center = np.asarray(pos[new_id], dtype=np.float64)
        o = np.array([origin[0], origin[1], origin[2]], dtype=np.float64)
        dn = np.array([direction[0], direction[1], direction[2]], dtype=np.float64)
        particle.depth = max(0.05, float((center - o) @ dn))
        particle.prev_target = center.copy()
        if os.environ.get("GRAB_DEBUG"):
            print(f"[slide] regrab -> id={new_id} members={len(particle.group)} "
                  f"depth={particle.depth:.3f}", flush=True)

    def _record_frame(self):
        rec = {"f": getattr(self, "_rec_frame", 0), "anchors": [], "pins": []}
        self._rec_frame = rec["f"] + 1
        for a in self.anchors:
            if a.flags & AnchorFlag.ACTIVE:
                o, d = ray_from_screen(a.screen[0], a.screen[1])
                rec["anchors"].append(dict(
                    id=int(a.id), screen=[float(a.screen[0]), float(a.screen[1])],
                    origin=[float(o[0]), float(o[1]), float(o[2])],
                    dir=[float(d[0]), float(d[1]), float(d[2])],
                    depth=float(a.depth)))
            elif a.flags & AnchorFlag.LOCKED:
                rec["pins"].append(int(a.id))
        rec["sphere"] = [float(sphere.center[0]), float(sphere.center[1]),
                         float(sphere.center[2]), float(sphere.radius),
                         float(sphere.dc[0]), float(sphere.dc[1]),
                         float(sphere.dc[2]), float(sphere.dr)]
        _clothRecordFile.write(json.dumps(rec) + "\n")
        _clothRecordFile.flush()

    def update_anchors(self):
        inv_mass = self.hostInvMass.numpy()
        pos = self.hostPos.numpy()
        self.activeGrabs = []
        if _clothRecordFile is not None:
            self._record_frame()

        for particle in self.anchors:
            if not particle.flags & (AnchorFlag.ACTIVE | AnchorFlag.LOCKED):
                inv_mass[particle.id] = particle.mass
                for mid, mmass, _ in particle.group:
                    inv_mass[mid] = mmass
                continue

            if not particle.flags & AnchorFlag.ACTIVE:
                # host-side sphere push-out for released anchors (whole grab patch);
                # push depth |d| along the UNIT normal (r is not unit length), against
                # the sphere's committed-this-frame pose (center+dc, radius+dr) like
                # the ACTIVE branch below
                for mid in [particle.id] + [m for m, _, _ in particle.group]:
                    r = wp.vec3f(pos[mid]) - (sphere.center + sphere.dc)
                    if colliderKind == 1:
                        r = wp.vec3f(r[0], r[1], 0.0)  # rod: radial = xy only
                    d = wp.length(r) - (sphere.radius + sphere.dr) - thickness - particleRadius
                    if d < 0:
                        pos[mid] -= d * wp.normalize(r)
                continue

            inv_mass[particle.id] = 0.0
            screen_x, screen_y = particle.screen
            origin, direction = ray_from_screen(screen_x, screen_y)

            # Keep the anchor on the CAMERA side of the sphere. The old rule
            # snapped to whichever boundary was nearer to the current depth
            # (mid < depth < far -> far), but fabric hanging BESIDE the sphere
            # sits at the sphere-center's depth, so dragging it across the
            # silhouette flung the anchor to the FAR shell -- the patch orbited
            # behind the sphere while its sheet wrapped the front face (reads as
            # interpenetration), and the depth then stuck at the far value after
            # leaving the silhouette (cloth left hanging in the air). Clamping to
            # the near boundary drags fabric OVER the visible face, which is what
            # the gesture means; a grab genuinely behind the sphere (depth beyond
            # the far boundary) is left alone.
            dist = ray_to_collider(origin, direction, sphere.center, sphere.radius + thickness)
            if dist and dist[0] < particle.depth < dist[1]:
                particle.depth = dist[0]

            # Check intersection with the ground
            d = wp.dot(Ground.NORMAL, direction)
            if wp.abs(d) > 1e-3:
                depth = -wp.dot(Ground.NORMAL, origin) / d
                if camera.pos[1] >= 0.0 and 0.5 < depth < particle.depth:
                    particle.depth = depth
                elif (grabGroundFollow and camera.pos[1] >= 0.0
                      and depth > particle.depth > 0.5
                      and pos[particle.id][1] <= grabGroundBand):
                    # Fabric held ON the floor, pointer receding: the old rule
                    # only ever DECREASED depth above ground, so dragging floor
                    # fabric AWAY from the camera left the target hovering at
                    # the grab depth, short of the pointer -- far-side floor
                    # drags moved ~1/3 of the commanded travel ("the drag is
                    # inoperant"). Let the target FOLLOW the receding ground
                    # intersection while the grabbed fabric is at floor level.
                    # RATE-LIMITED (2x the anchor's own speed clamp): near the
                    # horizon the ground depth diverges, and an uncapped follow
                    # would park the depth at a huge value that outlives the
                    # gesture (the sticky-depth bug class) and spoof the
                    # sliding-grab's chronic-lag trigger.
                    particle.depth = min(depth, particle.depth
                                         + 0.9 * d_offset * numSubsteps)
                elif (camera.pos[1] < 0.0
                      and wp.abs(pos[particle.id][1]) <= 2.0 * thickness
                      and depth > particle.depth):
                    particle.depth = depth

            target = origin + direction * particle.depth
            # Stage the grab for the PER-SUBSTEP sweep instead of teleporting the
            # patch here: a once-per-frame host-side position write moves the
            # pinned fabric in one invisible jump (several times d_offset on a
            # normal drag), so collision never sees the motion and the reference
            # state can land already interpenetrated inside other cloth --
            # measured as persistent self-collision violations and rising frame
            # times when dragging one flank across into the other. simulate()
            # packs activeGrabs into device buffers; advance/apply_grab_anchors
            # move the patch one slice per substep inside the captured graph,
            # exactly like the sphere collider's pose.
            # The grip stays CONFORMING: each frame the sphere push-out of each
            # member's TARGET position is half-folded into its stored offset, so
            # fabric slips around the sphere instead of scrubbing the grab-time
            # shape against it (only the push-out conforms -- anchor motion never
            # leaks in, so the grip stays firm away from the sphere).
            if particle.prev_target is None:
                particle.prev_target = np.array(pos[particle.id], dtype=np.float64)
            # SLIDING GRAB: when the committed anchor chronically lags the
            # commanded pointer-ray target, the grip is snagged (yield stall
            # against an anchored fold, or any obstruction) -- a real finger
            # would SLIP over the fabric rather than stay glued to a stuck
            # patch. Trigger only on CHRONIC lag: several consecutive frames
            # beyond the threshold AND not shrinking (a post-flick anchor
            # catches up at ~max_step per frame, so its lag falls fast and
            # resets the counter; a snagged grip's lag holds or grows).
            raw_np = np.array([target[0], target[1], target[2]], dtype=np.float64)
            if grabSlide and not particle.flags & AnchorFlag.LOCKED:
                # (LOCKED anchors keep their identity: sliding would silently
                # move a user-placed pin to a different vertex.)
                lag = float(np.linalg.norm(raw_np - particle.prev_target))
                max_step_f = 0.45 * d_offset * numSubsteps
                if lag > grabSlideLag and lag > particle.last_lag - 0.5 * max_step_f:
                    particle.slide_frames += 1
                else:
                    particle.slide_frames = 0
                particle.last_lag = lag
                if particle.slide_frames >= grabSlideFrames:
                    self._slide_grab(particle, raw_np, pos, inv_mass,
                                     origin, direction)
                    particle.slide_frames = 0
                    particle.last_lag = 0.0
                    # same stroke, new grip: re-derive the pointer target at
                    # the refreshed depth
                    target = origin + direction * particle.depth
            # Clamp the grab's effective speed: the free fabric a pinned patch
            # plows can yield at most ~pushClamp (0.5*d_offset) per substep, so a
            # patch advancing faster than that MUST cross through it -- pinned
            # motion has no final-say projection like the sphere collider. Cap the
            # per-substep advance at 0.45*d_offset (~4 m/s at 30 substeps): violent
            # flicks rubber-band (the anchor lags the pointer and catches up when
            # it slows), which is also what dragging real cloth through real cloth
            # feels like. prev_target commits the CLAMPED point so the device
            # sweep and the host state never disagree.
            delta = np.array([target[0], target[1], target[2]]) - particle.prev_target
            dist = float(np.linalg.norm(delta))
            max_step = 0.45 * d_offset * numSubsteps
            if dist > max_step:
                clamped = particle.prev_target + delta * (max_step / dist)
                target = wp.vec3f(clamped[0], clamped[1], clamped[2])
            # Load-yielding grip: yield under the plow pressure LAST frame's
            # narrowphase recorded on this grab's members (one frame of latency;
            # see the grabYield* notes near pushClamp). STALL shrinks the advance
            # toward the pointer as the mean member load rises; RETREAT backs the
            # anchor off along the net separation direction, capped at max_step
            # per frame so the grip lags (rubber-bands) but never detaches from
            # the patch -- members always sit exactly at anchor + offset.
            yld = self.grabPressureHost.get(particle.id)
            if yld is not None and (grabYieldStallK > 0.0 or grabYieldGain > 0.0):
                pvec, pmag, n_mem = yld
                mean_mag = pmag / float(n_mem * numSubsteps)  # per member-substep
                t_np = np.array([target[0], target[1], target[2]], dtype=np.float64)
                pv = np.asarray(pvec, dtype=np.float64)
                pnorm = float(np.linalg.norm(pv))
                # DIRECTION-AWARE yield (see grabYieldDirectional): pressure
                # magnitude alone stalls EXTRACTION exactly like plowing --
                # pulling members OUT along the net direction the contacts
                # push them (pvec, the escape direction) relieves the load,
                # yet the isotropic stall brakes it all the same. Split the
                # advance: ONLY the RELIEVING component (along +pvec) passes
                # unstalled; the remainder -- tangential AND anti-parallel --
                # keeps the validated isotropic stall + retreat. (A first cut
                # also passed the tangential component "because plowing
                # rotates pvec"; that is true in a pile but NOT on the sphere
                # shell, where pvec stays radial while a folded stack is
                # plowed tangentially across it -- fold_slide bad_streak went
                # 3 -> 30. Relief-pass-only restores the old behavior there.)
                # Requires a coherent net direction; a symmetric squeeze
                # (|pvec| << sum of member push magnitudes) has no meaningful
                # escape direction and stays fully isotropic.
                directional = (grabYieldDirectional and pnorm > 1e-12
                               and pnorm > grabYieldDirCoherence * pmag)
                adv = t_np - particle.prev_target
                relief = np.zeros(3)
                if directional:
                    ph = pv / pnorm
                    a_par = float(adv @ ph)
                    if a_par > 0.0:
                        relief = a_par * ph  # passes the stall untouched
                        adv = adv - relief
                # Pressure-budget: track the SUSTAINED pressure with an EMA and
                # stall only on the excess over beta*ema (see grabYieldSustain*).
                # The EMA updates every yielded frame, including mean_mag==0, so
                # the budget decays when contact clears.
                m_eff = mean_mag
                if grabYieldSustainBeta > 0.0:
                    p_ema = getattr(particle, "pressure_ema", 0.0)
                    m_eff = max(0.0, mean_mag - grabYieldSustainBeta * p_ema)
                    particle.pressure_ema = (
                        (1.0 - grabYieldSustainEma) * p_ema
                        + grabYieldSustainEma * mean_mag)
                scale = 1.0
                if grabYieldStallK > 0.0 and m_eff > 0.0:
                    scale = 1.0 / (1.0 + grabYieldStallK * m_eff / d_offset)
                minScale = grabYieldMinScale
                if minScale > 0.0:
                    # Minimum drag authority (see grabYieldMinScale).
                    if grabYieldSat:
                        # saturating shape: floor + (1-floor)/(1+K*m) -- keeps
                        # a smooth curve but weakens mid-range braking (A/B'd
                        # against the clip; see the ship notes).
                        scale = minScale + (1.0 - minScale) * scale
                    else:
                        scale = max(scale, minScale)
                if os.environ.get("GRAB_DEBUG"):
                    print(f"[yield] dir={directional} coh={pnorm / max(pmag, 1e-12):.2f} "
                          f"mean_mag={mean_mag:.5f} scale={scale:.3f} "
                          f"relief={np.linalg.norm(relief):.4f} "
                          f"|adv|={np.linalg.norm(adv):.4f}", flush=True)
                adv = adv * scale
                if grabYieldGain > 0.0:
                    retreat = grabYieldGain * pv / float(n_mem)
                    rmag = float(np.linalg.norm(retreat))
                    if rmag > max_step:
                        retreat *= max_step / rmag
                    # Low-pass the retreat: the raw pressure signal is a
                    # sawtooth (a retreat relieves the contact, so next
                    # frame's pressure collapses, the advance lunges back in,
                    # pressure spikes again...) -- applied raw it limit-cycles
                    # the anchor at a few Hz, felt as shakiness when dragging
                    # one flank against another (measured tv_ratio 1.5-1.7 vs
                    # 0.2 stall-only). An EMA turns the sawtooth into a steady
                    # partial back-off; the stall term is smooth already and
                    # stays unfiltered.
                    ema = getattr(particle, "retreat_ema", None)
                    if ema is None:
                        ema = np.zeros(3)
                    retreat = (1.0 - grabYieldRetreatEma) * ema \
                        + grabYieldRetreatEma * retreat
                    particle.retreat_ema = retreat
                    adv = adv + retreat
                t_np = particle.prev_target + relief + adv
                target = wp.vec3f(t_np[0], t_np[1], t_np[2])
            # CROSSING BRAKE (see GRAB_BRAKE): while the exact sweep reports
            # actual edge-through-face crossings within grabBrakeR of this
            # anchor, throttle the whole advance (relief included -- a rip is
            # not a legitimate contact, so the pressure stall's directional
            # bypass and authority floor do not apply) by 1/(1 + K*n), with
            # its own much lower floor. Self-releasing: the count comes back
            # from each frame's sweep, so as the recovery machinery clears
            # the plow front the brake fades and full authority returns.
            if grabBrakeK > 0.0 and particle.prev_target is not None:
                n_near = getattr(self, "crossNearHost", {}).get(particle.id, 0)
                # Peak-hold with decay (see grabBrakeDecay): brake on the max
                # of the current count and the decaying recent peak.
                hold = max(float(n_near),
                           getattr(particle, "brake_hold", 0.0) * grabBrakeDecay)
                particle.brake_hold = hold
                if hold > 0.5:
                    t_np = np.array([target[0], target[1], target[2]],
                                    dtype=np.float64)
                    adv_b = t_np - particle.prev_target
                    bscale = max(1.0 / (1.0 + grabBrakeK * hold),
                                 grabBrakeFloor)
                    t_np = particle.prev_target + adv_b * bscale
                    target = wp.vec3f(t_np[0], t_np[1], t_np[2])
                    if os.environ.get("GRAB_DEBUG"):
                        print(f"[brake] id={particle.id} n_near={n_near} "
                              f"hold={hold:.1f} scale={bscale:.3f}", flush=True)
            # The grab patch is pinned (inv_mass 0) and thus INVISIBLE to the
            # sphere collider: if the sphere overruns the anchor -- or the
            # anchor is dragged into the ball -- the patch parks inside the
            # shell and tows its whole fabric span in with it (measured 0.47
            # deep on drift-drag fuzz sessions). Keep the anchor point itself
            # outside the committed sphere pose, at the same contact distance
            # the member push-out below uses. Geometric, not rate-limited: the
            # ejected anchor rides the advancing shell like fabric on its
            # surface, so it must not lag behind a fast drift.
            sc = sphere.center + sphere.dc
            rvec = wp.vec3f(target[0] - sc[0], target[1] - sc[1], target[2] - sc[2])
            if colliderKind == 1:
                rvec = wp.vec3f(rvec[0], rvec[1], 0.0)  # rod: radial = xy only
            rlen = wp.length(rvec)
            rmin = sphere.radius + sphere.dr + thickness + particleRadius
            # Stack-aware clamp: the bare contact distance above assumes the
            # grabbed fabric sits DIRECTLY on the shell, but when the grab
            # rides a folded stack (drag the top layer of 2-3 folds across
            # the sphere) the trapped layers under the patch are squeezed
            # between two hard constraints -- the pinned patch pressed to
            # bare-shell distance and the final-say sphere projection.
            # Measured (fold_slide diagnostics): every big transient
            # violation cluster sits exactly under the grab ON the shell,
            # with 100-500 free particles in the patch column, and appears
            # the moment patch height drops below the free-stack height.
            # Lift the anchor clamp by the stack height in the patch's shadow
            # column, and (below) each MEMBER's push-out by its own local
            # column -- the patch is curved, so its down-slope members reach
            # bare-shell contact while the anchor still hovers. Candidates
            # exclude the patch's own topological skirt (same-sheet fabric
            # around the grab rides at patch height and must not read as a
            # "stack" -- a lone sheet dragged on a bare sphere stays in
            # normal contact); a folded under-layer is topologically far.
            stack_cand = None
            raw_lift = 0.0
            if anchorStack and rlen - rmin < anchorStackCap + 2.0 * fingerRadius:
                scn = np.array([sc[0], sc[1], sc[2]])
                rel_all = pos - scn
                if colliderKind == 1:
                    dist_all = np.linalg.norm(rel_all[:, :2], axis=1)
                else:
                    dist_all = np.linalg.norm(rel_all, axis=1)
                sd_all = dist_all - (sphere.radius + sphere.dr)
                ring = int(np.ceil(fingerRadius / self.spacing)) + 4
                pr, pc = particle.id // self.numCols, particle.id % self.numCols
                rows = np.arange(self.numParticles) // self.numCols
                cols = np.arange(self.numParticles) % self.numCols
                skirt = (np.abs(rows - pr) <= ring) & (np.abs(cols - pc) <= ring)
                cand = ((inv_mass > 0.0) & ~skirt
                        & (sd_all > 0.25 * d_offset) & (sd_all < anchorStackCap))
                if int(cand.sum()) >= 4:
                    stack_cand = (rel_all[cand], sd_all[cand])
                    tv = np.array([target[0], target[1], target[2]]) - scn
                    if colliderKind == 1:
                        tv[2] = 0.0  # rod: shadow column along the xy radial
                    tl = float(np.linalg.norm(tv))
                    if tl > 1e-9:
                        u = tv / tl
                        crel, csd = stack_cand
                        proj = crel @ u
                        perp2 = np.einsum('ij,ij->i', crel, crel) - proj * proj
                        shadow_r = fingerRadius + 2.0 * d_offset
                        under = (proj > 0.0) & (perp2 < shadow_r * shadow_r)
                        if int(under.sum()) >= 6:  # a real stack, not stray noise
                            raw_lift = min(float(np.percentile(csd[under], 90))
                                           + d_offset, anchorStackCap)
            if anchorStack:
                # Smooth the lift state (bounded rise AND bounded decay, see
                # the anchorLiftStep/Decay note) and apply it geometrically.
                prev_l = getattr(particle, 'stack_lift', 0.0)
                lift_a = float(min(max(raw_lift, prev_l - anchorLiftDecay),
                                   prev_l + anchorLiftStep))
                lift_a = max(lift_a, 0.0)
                particle.stack_lift = lift_a
                rmin += lift_a
            if rlen < rmin:
                out = rvec / rlen if rlen > 1e-9 else wp.vec3f(0.0, 1.0, 0.0)
                if colliderKind == 1:
                    # rod: eject in the xy plane, keep the free z coordinate
                    target = wp.vec3f(sc[0] + out[0] * rmin,
                                      sc[1] + out[1] * rmin,
                                      target[2])
                else:
                    target = wp.vec3f(sc[0] + out[0] * rmin,
                                      sc[1] + out[1] * rmin,
                                      sc[2] + out[2] * rmin)
            gids = [particle.id]
            goffs = [np.zeros(3)]
            old_group = particle.group
            new_group = []
            # Per-member stack lift: max under-column stack height for each
            # member's own radial column (the patch is curved -- its low-side
            # members reach bare-shell contact while the anchor hovers above
            # its lifted clamp; measured as the residual on-shell violation
            # bursts with viol_pinned ~0.7 after the anchor-level clamp).
            member_lift = None
            if anchorStack and particle.group:
                n_m = len(particle.group)
                raw_ml = np.zeros(n_m)
                if stack_cand is not None:
                    crel, csd = stack_cand
                    offs = np.array([[o[0], o[1], o[2]] for _, _, o in particle.group])
                    mrel = (np.array([target[0], target[1], target[2]]) - scn) + offs
                    if colliderKind == 1:
                        mrel[:, 2] = 0.0  # rod: member columns along xy radials
                    mlen = np.maximum(np.linalg.norm(mrel, axis=1), 1e-9)
                    mdir = mrel / mlen[:, None]
                    proj = mdir @ crel.T                     # [n_mem, n_cand]
                    perp2 = np.einsum('ij,ij->i', crel, crel)[None, :] - proj * proj
                    col_r = 2.0 * d_offset + 0.5 * self.spacing
                    in_col = (proj > 0.0) & (perp2 < col_r * col_r)
                    raw_ml = np.where(
                        in_col.any(axis=1),
                        np.minimum(np.where(in_col, csd[None, :], 0.0).max(axis=1)
                                   + d_offset, anchorStackCap),
                        0.0)
                # Same bounded rise/decay smoothing as the anchor lift.
                prev_ml = getattr(particle, 'member_stack_lift', None)
                if prev_ml is None or len(prev_ml) != n_m:
                    prev_ml = np.zeros(n_m)
                member_lift = np.maximum(
                    np.minimum(np.maximum(raw_ml, prev_ml - anchorLiftDecay),
                               prev_ml + anchorLiftStep), 0.0)
                particle.member_stack_lift = member_lift
                if not member_lift.any():
                    member_lift = None
            for gi, (mid, mmass, off) in enumerate(particle.group):
                inv_mass[mid] = 0.0
                p0 = wp.vec3f(target[0] + off[0], target[1] + off[1], target[2] + off[2])
                r = p0 - (sphere.center + sphere.dc)
                if colliderKind == 1:
                    r = wp.vec3f(r[0], r[1], 0.0)  # rod: radial push-out in xy
                lift = float(member_lift[gi]) if member_lift is not None else 0.0
                bare_d = wp.length(r) - (sphere.radius + sphere.dr + thickness + particleRadius)
                d = bare_d - lift
                off_staged = off
                if d < 0.0:
                    # Geometric push to the lifted contact distance; the lift
                    # itself is already rise/decay-smoothed above, so this
                    # cannot yank the patch nor sawtooth around the stack top.
                    push = -d * wp.normalize(r)
                    # Stage the FULL push-out for this frame's pinned
                    # placement: a pinned member is invisible to the sphere
                    # collider, so a staged offset inside the shell IS
                    # user-visible fabric penetration (fold_slide diagnostics:
                    # every sphere-penetrating edge during a stack slide had a
                    # pinned endpoint; the half-folded offset lagged the
                    # anchor's approach by several frames at ~mem_shell -0.01).
                    # The STORED offset keeps the original half-fold conform
                    # so the grip shape still relaxes gradually and anchor
                    # motion never leaks into the blend.
                    if grabConformFull:
                        off_staged = off + push
                    off = off + 0.5 * push
                    if not grabConformFull:
                        off_staged = off
                gids.append(mid)
                goffs.append(np.array([off_staged[0], off_staged[1], off_staged[2]]))
                new_group.append((mid, mmass, off))
            particle.group = new_group
            # Patch self-crossing veto (see grabVeto/_patch_self_crossed): a
            # staged shape whose rim folds through itself would lock crossings
            # for the life of the grab -- reuse the last clean shape instead
            # (offsets are anchor-relative, so it rides the new target).
            if grabVeto:
                tnp = np.array([target[0], target[1], target[2]])
                staged = [tnp + gof for gof in goffs]
                if self._patch_self_crossed(particle, gids, staged):
                    prev_goffs = getattr(particle, "_last_goffs", None)
                    if prev_goffs is not None and len(prev_goffs) == len(goffs):
                        goffs = prev_goffs
                        particle.group = old_group
                        if os.environ.get("GRAB_DEBUG"):
                            print("[grab-veto] staged patch self-crossed: "
                                  "reusing last clean shape", flush=True)
                else:
                    particle._last_goffs = [np.array(gof) for gof in goffs]
            self.activeGrabs.append((particle.prev_target.copy(),
                                     np.array([target[0], target[1], target[2]]),
                                     np.array(gids, dtype=np.int32),
                                     np.array(goffs, dtype=np.float32)))
            particle.prev_target = np.array([target[0], target[1], target[2]], dtype=np.float64)

        self.anchors[:] = [anchor for anchor in self.anchors if anchor.flags & (AnchorFlag.ACTIVE | AnchorFlag.LOCKED)]

        # Rim pair solver (see RIM_SOLVER): rebuild the ring-1 rim pair lists
        # only when the active grab membership changes (grab / release /
        # sliding re-grab); simulate() uploads the staged lists when dirty.
        if rimSolverEnable:
            grabs = [a for a in self.anchors
                     if a.flags & AnchorFlag.ACTIVE and a.group]
            sig = tuple((a.id, a.group[0][0], len(a.group)) for a in grabs)
            if sig != self._rimSig:
                self._rimSig = sig
                members = [np.array([a.id] + [m for m, _, _ in a.group],
                                    dtype=np.int64) for a in grabs]
                self._rimStage = self._build_rim_pairs(members, inv_mass)
                self._rimDirty = True

    def _patch_self_crossed(self, particle, gids, staged):
        # EXACT self-intersection check of one grab's staged patch surface
        # (member edges vs non-adjacent member faces, vectorized
        # Moller-Trumbore on the staged positions). The pinned patch is a hole
        # in every runtime mechanism -- PDT culls member pairs (ring), the
        # uncross resolver moves free vertices only, and the device CCD treats
        # the patch as rigid -- so a CROSSED shape baked into the offsets by
        # the per-frame conform (curvature-sheared push-outs at the shell
        # silhouette, per-member stack lifts) locks for the life of the grab.
        # Probe attribution on the recorded session: the largest in-drag
        # crossing bursts were mem=5/5 pairs (member edge through member
        # face). update_anchors vetoes a staged shape that self-crosses and
        # re-uses the last clean one instead. Cost: only while grabbing, a
        # few hundred edges x faces, numpy.
        n = len(gids)
        if n < 4:
            return False
        key = (n, gids[0], gids[-1])
        topo = getattr(particle, "_patch_topo", None)
        if topo is None or topo[0] != key:
            ids = np.asarray(gids, dtype=np.int64)
            order = np.argsort(ids)
            sids = ids[order]
            T = self.hostTriIds
            E = self.hostEdgeIds
            def to_idx(arr):
                p = np.searchsorted(sids, arr)
                p = np.clip(p, 0, n - 1)
                ok = sids[p] == arr
                return order[p], ok
            ti, tok = to_idx(T)
            fmask = tok.all(axis=1)
            ei, eok = to_idx(E)
            emask = eok.all(axis=1)
            topo = (key, ei[emask], ti[fmask])
            particle._patch_topo = topo
        _, pe, pf = topo
        if len(pe) == 0 or len(pf) == 0:
            return False
        S = np.asarray(staged, dtype=np.float64)
        # all edge x face pairs, excluding shared-vertex pairs
        share = (pe[:, 0:1, None] == pf[None, :, :]).any(-1) \
            | (pe[:, 1:2, None] == pf[None, :, :]).any(-1)
        epair, fpair = np.nonzero(~share)
        if len(epair) == 0:
            return False
        o = S[pe[epair, 0]]
        dv = S[pe[epair, 1]] - o
        v0 = S[pf[fpair, 0]]
        e1 = S[pf[fpair, 1]] - v0
        e2 = S[pf[fpair, 2]] - v0
        h = np.cross(dv, e2)
        a = np.einsum('ij,ij->i', e1, h)
        ok = np.abs(a) > 1e-14
        f = np.zeros_like(a)
        f[ok] = 1.0 / a[ok]
        s = o - v0
        u = f * np.einsum('ij,ij->i', s, h)
        q = np.cross(s, e1)
        v = f * np.einsum('ij,ij->i', dv, q)
        tt = f * np.einsum('ij,ij->i', e2, q)
        eps = 1e-9
        hit = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps) \
            & (tt > 1e-6) & (tt < 1.0 - 1e-6)
        return bool(hit.any())

    def _uncross_mask(self):
        # Forcing-site mask for in-drag resolver sweeps (see uncrossMaskR):
        # balls around every active grab anchor target, plus the sphere shell
        # while it is being driven. Crossed pairs inside a ball are left to
        # the post-release path; everything else (the trailing wad) resolves.
        m = [(np.asarray(target, dtype=np.float64), uncrossMaskR)
             for _prev, target, _gids, _goffs in self.activeGrabs]
        if sphere.dc[0] != 0.0 or sphere.dc[1] != 0.0 \
                or sphere.dc[2] != 0.0 or sphere.dr != 0.0:
            c = np.array([sphere.center[0], sphere.center[1],
                          sphere.center[2]], dtype=np.float64)
            m.append((c, float(sphere.radius) + uncrossMaskR))
        return m

    def _resolve_crossings(self, max_n=None, mask=None):
        # Crossing resolver (see the uncross* constants): exact intersection
        # sweep on the current positions, then a host-side CLUSTER vote that
        # flips one coherent side of each intersection contour back across
        # the partner surface (free vertices only, veto'd against creating
        # new foreign-layer crossings). Runs outside the captured graph,
        # before the substep replays -- the frame's substeps of PDT/repulsion
        # then separate the un-crossed pairs on the correct side. Returns the
        # first sweep's crossing count (0 = clean, drives the idle backoff).
        # A vertex flipped once this frame is LOCKED against re-flipping by
        # a later iteration (uncrossIters > 1 is experimental -- see the
        # cascade note at uncrossIters).
        im = None
        found = 0        # first-sweep crossing count (returned for idle backoff)
        moved = set()    # flipped this frame: final, never flipped again
        blocked = set()  # veto'd this frame: the pair retries its other endpoint
        rounds = uncrossIters
        for _it in range(max(uncrossIters, uncrossBurstIters)):
            if _it >= rounds:
                break
            self.grid.build(self.pos, gridCellSize)
            self.crossBounds.zero_()
            wp.launch(kernel=Cloth.max_edge_length,
                      dim=boundsReduceThreads,
                      inputs=[self.pos, self.edgeIds],
                      outputs=[self.crossBounds])
            self.crossCount.zero_()
            wp.launch(kernel=Cloth.detect_crossings,
                      dim=self.numEdges,
                      inputs=[self.grid.id, self.pos, self.edgeIds, self.triIds,
                              self.gridRC, self.vertFaceOff, self.vertFaceIds,
                              self.crossBounds],
                      outputs=[self.crossPairs, self.crossCount])
            n = int(self.crossCount.numpy()[0])  # device read: syncs the stream
            if _it == 0:
                found = n
                # Large-wad recovery burst (quiescent path only, see
                # uncrossBurstN): peel several contour rings this frame.
                if max_n is None and n > uncrossBurstN:
                    rounds = max(rounds, uncrossBurstIters)
            if n == 0 or (max_n is not None and n > max_n):
                return found
            if _UNCROSS_DEBUG:
                print(f"[uncross] it{_it}: {n} crossed pairs", flush=True)
            n = min(n, maxCross)
            pairs = self.crossPairs.numpy()[:n]
            P = self.pos.numpy()
            if im is None:
                im = self.hostInvMass.numpy()
            E = self.hostEdgeIds
            T = self.hostTriIds
            disp = {}
            need = {}      # target vertex -> deepest single-pair flip distance
            partners = {}  # target vertex -> intended partner face ids
            recs = []      # (va, vb, f, nf, da, db) per valid crossed pair
            for e, f in pairs:
                va, vb = int(E[e, 0]), int(E[e, 1])
                if mask:
                    mid = 0.5 * (P[va] + P[vb])
                    if any(np.linalg.norm(mid - mc) < mr for mc, mr in mask):
                        continue  # forcing site: leave to post-release
                i0, i1, i2 = int(T[f, 0]), int(T[f, 1]), int(T[f, 2])
                a0 = P[i0]
                nf = np.cross(P[i1] - a0, P[i2] - a0)
                ln = np.linalg.norm(nf)
                if ln < 1.0e-12:
                    continue
                nf /= ln
                da = float(np.dot(P[va] - a0, nf))
                db = float(np.dot(P[vb] - a0, nf))
                if da * db > 0.0:
                    continue  # float-noise mismatch with the exact test: skip
                recs.append((va, vb, int(f), nf, da, db))

            def add_flip(s, ds, nf, f):
                step = -np.sign(ds) * (abs(ds) + uncrossMargin) * nf
                disp[s] = disp.get(s, 0.0) + step
                need[s] = max(need.get(s, 0.0), abs(ds) + uncrossMargin)
                partners.setdefault(s, []).append(f)

            if uncrossVote == "pair":
                # independent per-pair least-motion (fallback policy)
                for va, vb, f, nf, da, db in recs:
                    s, ds = (va, da) if abs(da) <= abs(db) else (vb, db)
                    o, do_ = (vb, db) if s == va else (va, da)
                    if im[s] == 0.0 or s in moved or s in blocked:
                        s, ds = o, do_
                        if im[s] == 0.0 or s in moved or s in blocked:
                            continue
                    add_flip(s, ds, nf, f)
            else:
                # CLUSTER vote: union-find the crossing edges into contour
                # clusters (shared endpoints) and flip ONE coherent side per
                # cluster. Per-pair least-motion picks incoherent directions
                # along a band (each pair flips its own shallow endpoint,
                # which for a deep intrusion ADVANCES the front instead of
                # retracting it -- measured as a growing contour). The side
                # with fewer vertices (the intruded tongue) flips; on a tie,
                # the side with the smaller total depth (least region
                # motion).
                parent = {}

                def find(x):
                    while parent.get(x, x) != x:
                        parent[x] = parent.get(parent[x], parent[x])
                        x = parent[x]
                    return x

                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx != ry:
                        parent[rx] = ry

                vdep = {}
                for va, vb, f, nf, da, db in recs:
                    union(va, vb)
                    vdep.setdefault(va, []).append((da, nf, f))
                    vdep.setdefault(vb, []).append((db, nf, f))
                clusters = {}
                for v in vdep:
                    clusters.setdefault(find(v), []).append(v)
                for members in clusters.values():
                    dsum = {v: sum(d for d, _, _ in vdep[v]) for v in members}
                    plus = [v for v in members if dsum[v] > 0.0]
                    minus = [v for v in members if dsum[v] <= 0.0]
                    if len(plus) != len(minus):
                        flip = plus if len(plus) < len(minus) else minus
                    else:
                        dp = sum(abs(dsum[v]) for v in plus)
                        dm = sum(abs(dsum[v]) for v in minus)
                        flip = plus if dp <= dm else minus
                    for v in flip:
                        if im[v] == 0.0 or v in moved or v in blocked:
                            continue
                        for d, nf, f in vdep[v]:
                            add_flip(v, d, nf, f)
            if not disp:
                return found
            # Regional dilation (see uncrossDilate): drag each flip's grid
            # neighborhood along so the pleat lobe moves bodily instead of
            # having its crossed ring yanked back by interior tension.
            if uncrossDilate > 0:
                nrows = self.numParticles // self.numCols
                frontier = dict(disp)
                for _ring in range(uncrossDilate):
                    acc = {}
                    for v, dvv in frontier.items():
                        r, c = v // self.numCols, v % self.numCols
                        if r > 0:
                            acc.setdefault(v - self.numCols, []).append((v, dvv))
                        if r < nrows - 1:
                            acc.setdefault(v + self.numCols, []).append((v, dvv))
                        if c > 0:
                            acc.setdefault(v - 1, []).append((v, dvv))
                        if c < self.numCols - 1:
                            acc.setdefault(v + 1, []).append((v, dvv))
                    frontier = {}
                    for u, contrib in acc.items():
                        if u in disp or u in moved or u in blocked \
                                or im[u] == 0.0:
                            continue
                        step = uncrossDilateGain \
                            * (sum(d for _, d in contrib) / len(contrib))
                        disp[u] = step
                        need[u] = float(np.linalg.norm(step))
                        # veto allowed-set: inherit the contributing seeds'
                        # intended partner faces
                        pl = partners.setdefault(u, [])
                        for sv, _ in contrib:
                            pl.extend(partners.get(sv, ()))
                        frontier[u] = step
            ids = np.fromiter(disp.keys(), dtype=np.int32, count=len(disp))
            dv = np.stack([disp[int(i)] for i in ids]).astype(np.float32)
            mag = np.linalg.norm(dv, axis=1)
            # DEPTH-COMPLETE step cap (see uncrossStepMax): a flip that cannot
            # reach past the partner plane is worse than useless -- it lands
            # the vertex still-crossed at d in [farBarrierFloor, 1)*d_offset,
            # where the c<0 barrier plane truncates its return motion and the
            # recovery push drives it deeper: the flip is undone within the
            # frame's substeps (measured on the locked 400x400 recorded wad:
            # depth med 0.0069 / p90 0.0089 vs the old fixed 0.00675 cap ->
            # ~180 flips/frame applied, net drain ~2 pairs/frame, reads as a
            # permanently locked knot). Cap each vertex by what it NEEDS to
            # cross its own deepest partner plane (+margin), floored at the
            # legacy uncrossStep, bounded by uncrossStepMax; the no-new-
            # crossing veto below exact-tests the full longer segment, so a
            # deep flip through a third layer is still rejected.
            cap = np.fromiter(
                (min(max(need[int(i)], uncrossStep), uncrossStepMax)
                 for i in ids), dtype=np.float64, count=len(ids))
            over = mag > cap
            dv[over] *= (cap[over] / mag[over])[:, None]
            # NO-NEW-CROSSING VETO: in a multi-layer pile (or a tightly
            # creased fold, where the "layers" are material neighbors),
            # flipping a vertex across its partner sheet B can carry it
            # THROUGH a third layer sitting just behind -- the next frame's
            # sweep then flips it back (oscillation: the knot never resolves,
            # and the churn pumps stretch). Exact-test each flip segment
            # against the nearby faces; only the recorded partner faces and
            # their vertex-adjacent neighbors (the same local surface, in
            # case the segment exits through a coplanar neighbor of f) may be
            # crossed -- ANY other face vetoes the flip. A vetoed vertex is
            # blocked for this frame so the pair retries with its other
            # endpoint next iteration.
            P0 = P[ids]
            P1 = P0 + dv
            lo = np.minimum(P0, P1).min(axis=0) - 0.03
            hi = np.maximum(P0, P1).max(axis=0) + 0.03
            tc = (P[T[:, 0]] + P[T[:, 1]] + P[T[:, 2]]) / 3.0
            tr = np.maximum(np.linalg.norm(P[T[:, 0]] - tc, axis=1),
                            np.maximum(np.linalg.norm(P[T[:, 1]] - tc, axis=1),
                                       np.linalg.norm(P[T[:, 2]] - tc, axis=1)))
            cand = np.nonzero(np.all((tc + tr[:, None] >= lo)
                                     & (tc - tr[:, None] <= hi), axis=1))[0]
            keep = np.ones(len(ids), dtype=bool)
            if len(cand) and uncrossVetoVec:
                # VECTORIZED veto (2026-09): the per-flip loop below tests
                # every flip against EVERY face in the joint bbox -- O(flips
                # x bbox faces), ~5 s per round on a 2k-pair 400^2 wad
                # spread across the pile (10k dilated flips x 100k faces),
                # i.e. minutes per recovery frame with burst rounds and the
                # settle-interleave. Same candidate semantics (every face
                # whose centroid lies within slen + tr + 1e-5 of the segment
                # midpoint), found through a uniform grid on the candidate
                # face centroids and tested with one batched Moller-Trumbore
                # over the (flip, face) pairs. UNCROSS_VETO_VEC=0 restores
                # the loop; UNCROSS_VETO_CHECK=1 runs both and asserts.
                keep = self._veto_flips_vec(ids, dv, P0, P, T, cand, tc, tr,
                                            partners)
                if uncrossVetoCheck:
                    keep_ref = self._veto_flips_loop(ids, dv, P0, P, T, cand,
                                                     tc, tr, partners, set())
                    if not np.array_equal(keep, keep_ref):
                        raise AssertionError(
                            f"[uncross] veto mismatch: vec {int((~keep).sum())} "
                            f"vs loop {int((~keep_ref).sum())} vetoes")
                for s in ids[~keep]:
                    blocked.add(int(s))
            elif len(cand):
                keep = self._veto_flips_loop(ids, dv, P0, P, T, cand, tc, tr,
                                             partners, blocked)
            if _UNCROSS_DEBUG:
                print(f"[uncross]   apply={int(keep.sum())} veto={int((~keep).sum())}",
                      flush=True)
            ids = ids[keep]
            dv = dv[keep]
            if not len(ids):
                continue  # everything veto'd: retry other endpoints next iter
            moved.update(int(i) for i in ids)
            wp.launch(kernel=Cloth.apply_uncross,
                      dim=len(ids),
                      inputs=[wp.array(ids, dtype=wp.int32),
                              wp.array(dv, dtype=wp.vec3), self.pos])
        return found

    def _allowed_codes(self, ids, partners, T, ks):
        # (flip index k, face f) codes of the faces a flip segment MAY cross:
        # its recorded partner faces and their vertex-adjacent neighbors --
        # only for the flips ks that still have an external candidate.
        vfo = self.hostVertFaceOff
        vfi = self.hostVertFaceIds
        nT = T.shape[0]
        codes = []
        cache = {}   # dilated flips inherit their seeds' lists: share the work
        for k in ks:
            k = int(k)
            pl = partners[int(ids[k])]
            if not pl:
                continue
            key = tuple(sorted(set(pl)))
            fs = cache.get(key)
            if fs is None:
                pf = np.asarray(key, dtype=np.int64)
                verts = np.unique(T[pf].ravel())
                parts = [pf]
                for pv in verts:
                    parts.append(vfi[vfo[pv]:vfo[pv + 1]].astype(np.int64))
                fs = cache[key] = np.unique(np.concatenate(parts))
            codes.append(k * nT + fs)
        if not codes:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(codes))

    def _veto_flips_vec(self, ids, dv, P0, P, T, cand, tc, tr, partners):
        n = len(ids)
        keep = np.ones(n, dtype=bool)
        tcc = tc[cand]
        trc = tr[cand]
        smid = P0 + 0.5 * dv
        slen = 0.5 * np.linalg.norm(dv, axis=1)
        reach = slen + float(trc.max()) + 1e-5          # per-flip search radius
        cell = float(reach.max()) + 1e-6
        # uniform grid over the candidate centroids
        origin = np.minimum(tcc.min(axis=0), smid.min(axis=0)) - cell
        fk = np.floor((tcc - origin) / cell).astype(np.int64)
        dims = fk.max(axis=0) + 3
        fkey = (fk[:, 0] * dims[1] + fk[:, 1]) * dims[2] + fk[:, 2]
        order = np.argsort(fkey, kind="stable")
        fkey_s = fkey[order]
        sk = np.floor((smid - origin) / cell).astype(np.int64)
        offs = np.array([(i, j, l) for i in (-1, 0, 1) for j in (-1, 0, 1)
                         for l in (-1, 0, 1)], dtype=np.int64)
        nk = sk[:, None, :] + offs[None, :, :]                     # [n, 27, 3]
        nk = np.clip(nk, 0, dims - 1)
        nkey = ((nk[..., 0] * dims[1] + nk[..., 1]) * dims[2]
                + nk[..., 2]).reshape(-1)                            # [n*27]
        lo_i = np.searchsorted(fkey_s, nkey, side="left")
        hi_i = np.searchsorted(fkey_s, nkey, side="right")
        cnt = hi_i - lo_i
        tot = int(cnt.sum())
        if tot == 0:
            return keep
        # expand the (flip, cell) ranges into (flip, cand-face) pairs
        rep_k = np.repeat(np.arange(n * 27) // 27, cnt)
        starts = np.repeat(lo_i, cnt)
        within = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        pj = order[starts + within]                                  # cand index
        pk = rep_k
        # distance filter (identical to the loop's `near`)
        near = np.linalg.norm(tcc[pj] - smid[pk], axis=1) <= slen[pk] + trc[pj] + 1e-5
        pk = pk[near]; pj = pj[near]
        if not len(pk):
            return keep
        # exclude the flip vertex's own faces and its allowed set
        Tc = T[cand]
        own = (Tc[pj] == ids[pk][:, None]).any(axis=1)
        pk = pk[~own]; pj = pj[~own]
        if not len(pk):
            return keep
        nT = T.shape[0]
        allowed = self._allowed_codes(ids, partners, T, np.unique(pk))
        if len(allowed):
            codes = pk.astype(np.int64) * nT + cand[pj].astype(np.int64)
            ext = ~np.isin(codes, allowed, assume_unique=False)
            pk = pk[ext]; pj = pj[ext]
            if not len(pk):
                return keep
        # batched Moller-Trumbore, same tolerances as the loop
        A = P[Tc[pj, 0]]
        E1 = P[Tc[pj, 1]] - A
        E2 = P[Tc[pj, 2]] - A
        d = dv[pk]
        h = np.cross(d, E2)
        det = np.einsum('ij,ij->i', E1, h)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sv = P0[pk] - A
        u = np.einsum('ij,ij->i', sv, h) * inv
        q = np.cross(sv, E1)
        vpar = np.einsum('ij,ij->i', d, q) * inv
        tpar = np.einsum('ij,ij->i', E2, q) * inv
        hit = ok & (u >= 0) & (vpar >= 0) & (u + vpar <= 1) & (tpar > 0) & (tpar < 1)
        if hit.any():
            keep[np.unique(pk[hit])] = False
        return keep

    def _veto_flips_loop(self, ids, dv, P0, P, T, cand, tc, tr, partners, blocked):
        keep = np.ones(len(ids), dtype=bool)
        if len(cand):
            Tc = T[cand]
            tcc = tc[cand]
            trc = tr[cand]
            A = P[Tc[:, 0]]
            E1 = P[Tc[:, 1]] - A
            E2 = P[Tc[:, 2]] - A
            vfo = self.hostVertFaceOff
            vfi = self.hostVertFaceIds
            for k in range(len(ids)):
                s = int(ids[k])
                d = dv[k]
                smid = P0[k] + 0.5 * d
                slen = 0.5 * np.linalg.norm(d)
                near = np.linalg.norm(tcc - smid, axis=1) <= slen + trc + 1e-5
                if not near.any():
                    continue
                j = np.nonzero(near)[0]
                # allowed set: the intended partner faces + their
                # vertex-adjacent neighbors (same local surface patch)
                allowed = set()
                for pf in partners[s]:
                    allowed.add(pf)
                    for pv in T[pf]:
                        allowed.update(
                            int(x) for x in vfi[vfo[pv]:vfo[pv + 1]])
                own = (Tc[j] == s).any(axis=1)
                ext = np.array([int(cand[x]) not in allowed
                                for x in j], dtype=bool)
                j = j[~own & ext]
                if not len(j):
                    continue
                h = np.cross(d, E2[j])
                det = np.einsum('ij,ij->i', E1[j], h)
                ok = np.abs(det) > 1e-14
                inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
                sv = P0[k] - A[j]
                u = np.einsum('ij,ij->i', sv, h) * inv
                q = np.cross(sv, E1[j])
                vpar = np.einsum('j,ij->i', d, q) * inv
                tpar = np.einsum('ij,ij->i', E2[j], q) * inv
                hit = ok & (u >= 0) & (vpar >= 0) & (u + vpar <= 1) \
                    & (tpar > 0) & (tpar < 1)
                if hit.any():
                    keep[k] = False
                    blocked.add(s)
        return keep

    def simulate(self, steps=numSubsteps, iterations=numIterations, integrate=True, self_collision=True, solve_constraints=True):
        dt = timeStep / numSubsteps

        wp.copy(self.pos, self.hostPos)
        wp.copy(self.invMass, self.hostInvMass)

        # The resolver only acts on QUIESCENT-CONTROL frames (no active grab,
        # sphere not being driven): during sustained forcing the 30 substeps
        # of plow re-cross whatever one host-side pass uncrosses, and the
        # fight churns the pressed pile (measured: worse outcomes and stretch
        # blow-ups with in-drag resolution). Post-release -- where persistent
        # knots actually matter -- it untangles within a few frames.
        quiescent = (not self.activeGrabs
                     and sphere.dc[0] == 0.0 and sphere.dc[1] == 0.0
                     and sphere.dc[2] == 0.0 and sphere.dr == 0.0)
        self._uncrossFrame = getattr(self, "_uncrossFrame", 0) + 1
        recovery_found = 0  # this frame's quiescent sweep count (interleave gate)
        if uncrossEnable and self_collision:
            # Idle backoff: a clean sweep costs ~3.5 ms/frame at 400x400 (grid
            # build + per-edge walk + sync) and a settled scene stays clean,
            # so after each empty sweep the cadence decays 1 -> 2 -> 4 frames;
            # any hit (or an interaction ending) restores full rate.
            skip = getattr(self, "_uncrossSkip", 0)
            if quiescent or not uncrossGated:
                if skip > 0:
                    self._uncrossSkip = skip - 1
                else:
                    found = recovery_found = self._resolve_crossings()
                    back = getattr(self, "_uncrossBackoff", 0)
                    if found == 0:
                        self._uncrossBackoff = min(max(back, 1) * 2, 4)
                        self._uncrossSkip = self._uncrossBackoff - 1
                    else:
                        self._uncrossBackoff = 0
                        self._uncrossSkip = 0
            elif (uncrossForcedEvery > 0
                  and self._uncrossFrame % uncrossForcedEvery == 0):
                # In-drag sweep with the forcing sites masked out (see the
                # uncrossForcedEvery block): prunes the trailing wad while
                # the user still drags, so release starts near-clean.
                # (Default OFF: unmasked-unbraked variants measured harmful.)
                self._resolve_crossings(max_n=uncrossForcedMaxN,
                                        mask=self._uncross_mask())
                self._uncrossSkip = 0
                self._uncrossBackoff = 0
            elif (uncrossDrag and grabBrakeK > 0.0 and self.activeGrabs
                    and getattr(self, "_lastCrossTotal", 0) > 0
                    and self._uncrossFrame % uncrossDragEvery == 0):
                # In-drag resolution (see UNCROSS_DRAG): fires only while the
                # brake's sweep is reporting actual crossings under an active
                # grab, i.e. exactly when the anchor is throttled -- the brake
                # holds creation below the resolver's drain rate (in-drag
                # resolution WITHOUT the brake loses to the plow's creation
                # rate; measured 96->745).
                self._resolve_crossings(max_n=uncrossForcedMaxN)
                self._uncrossSkip = 0
                self._uncrossBackoff = 0

        # Crossed-vertex flags for the side-aware barrier (see FLAG_GUARD) and
        # the crossing-brake signal (see GRAB_BRAKE): while a grab is active,
        # run the resolver's exact intersection sweep on the frame-start
        # positions and scatter the involved vertices into crossedFlag --
        # entirely on-device, no host readback here, so it adds no sync point
        # (the brake reads the count after the frame's existing sync). The
        # narrowphase consults the flags only while n_grabs > 0, so the
        # buffer being stale outside grabs is harmless.
        self._crossSwept = ((flagGuardEnable or grabBrakeK > 0.0)
                            and self_collision and bool(self.activeGrabs))
        if self._crossSwept:
            self.grid.build(self.pos, gridCellSize)
            self.crossBounds.zero_()
            wp.launch(kernel=Cloth.max_edge_length,
                      dim=boundsReduceThreads,
                      inputs=[self.pos, self.edgeIds],
                      outputs=[self.crossBounds])
            self.crossCount.zero_()
            wp.launch(kernel=Cloth.detect_crossings,
                      dim=self.numEdges,
                      inputs=[self.grid.id, self.pos, self.edgeIds, self.triIds,
                              self.gridRC, self.vertFaceOff, self.vertFaceIds,
                              self.crossBounds],
                      outputs=[self.crossPairs, self.crossCount])
            self.crossedFlag.zero_()
            wp.launch(kernel=Cloth.scatter_crossed_flags,
                      dim=maxCross,
                      inputs=[self.crossPairs, self.crossCount, self.edgeIds,
                              self.triIds],
                      outputs=[self.crossedFlag])

        # Capture one substep ONCE per flag combination and replay it across frames.
        # Everything per-frame (collider pose and delta slices) lives in device
        # arrays, so the graph is frame-invariant. This matters doubly with the
        # hash-grid build captured in-graph: the build's mempool alloc nodes make
        # graph INSTANTIATION cost ~0.5 s (a one-time hitch per flag combination),
        # while replaying it is ~0.06 ms -- and issuing the build on the stream
        # between replays instead serializes against the graph launches (~+2 ms per
        # substep), so in-graph + capture-once is the only fast arrangement.
        key = (iterations, integrate, self_collision, solve_constraints)
        graph = self._graphs.get(key)
        if graph is None:
            with wp.ScopedCapture() as capture:
                self.step(dt, iterations, integrate, self_collision, solve_constraints)
            graph = self._graphs[key] = capture.graph
        # Seed the collider pose at the frame START (outside the graph); advance_sphere
        # then walks it forward one substep slice per replay, landing on sphere.center+dc.
        # The delta slices feed the collider kernels through device memory.
        # Slice by the ACTUAL replay count: debug step modes call simulate(steps=1)
        # while Sphere.post_render commits the full frame delta -- slicing by
        # numSubsteps there would sweep only 1/numSubsteps of the sphere motion.
        inv_sub = 1.0 / float(steps)
        self.colliderDeltaC.fill_(sphere.dc * inv_sub)
        self.colliderDeltaR.fill_(sphere.dr * inv_sub)
        self.colliderDeltaQ.fill_(quat_fraction(sphere.dq, inv_sub))
        self.colliderCenter.fill_(sphere.center)
        self.colliderRadius.fill_(sphere.radius)
        self.selfCollisionOverflow.zero_()  # per-frame; accumulates over the replayed substeps
        self.grabPressure.zero_()           # per-frame grip-load accumulators
        self.grabPressureMag.zero_()

        # Pack the grab anchors for the per-substep sweep. activeGrabs entries are
        # (prev, target, ids, offs): prev = last frame's committed anchor point,
        # target = where this frame should end; each substep moves the patch by
        # (target - prev) / steps (see advance/apply_grab_anchors in step()).
        n_members = 0
        if self.activeGrabs:
            ids_np = np.zeros(maxGrabMembers, dtype=np.int32)
            ax_np = np.zeros(maxGrabMembers, dtype=np.int32)
            off_np = np.zeros((maxGrabMembers, 3), dtype=np.float32)
            pos_np = np.zeros((maxGrabs, 3), dtype=np.float32)
            dlt_np = np.zeros((maxGrabs, 3), dtype=np.float32)
            for gi, (prev, target, gids, goffs) in enumerate(self.activeGrabs[:maxGrabs]):
                n = min(len(gids), maxGrabMembers - n_members)
                if n < len(gids):
                    print(f"[grab] member buffer full: dropping {len(gids) - n}", flush=True)
                ids_np[n_members:n_members + n] = gids[:n]
                ax_np[n_members:n_members + n] = gi
                off_np[n_members:n_members + n] = goffs[:n]
                pos_np[gi] = prev
                dlt_np[gi] = (np.asarray(target) - np.asarray(prev)) / float(steps)
                n_members += n
            wp.copy(self.anchorMemberIds, wp.array(ids_np, dtype=wp.int32))
            wp.copy(self.anchorMemberAx, wp.array(ax_np, dtype=wp.int32))
            wp.copy(self.anchorMemberOff, wp.array(off_np, dtype=wp.vec3))
            wp.copy(self.anchorPos, wp.array(pos_np, dtype=wp.vec3))
            wp.copy(self.anchorDelta, wp.array(dlt_np, dtype=wp.vec3))
        self.anchorMemberCount.fill_(n_members)
        self.anchorCount.fill_(min(len(self.activeGrabs), maxGrabs))

        # Rim pair solver: upload the staged pair lists when membership changed
        # (see update_anchors); counts live on device so the graph is invariant.
        if rimSolverEnable and self._rimDirty:
            vt_np, ee_np = self._rimStage
            n_vt = min(len(vt_np), maxRimVT)
            n_ee = min(len(ee_np), maxRimEE)
            if len(vt_np) > maxRimVT or len(ee_np) > maxRimEE:
                print(f"[rim] pair buffer full: dropping "
                      f"{len(vt_np) - n_vt} VT / {len(ee_np) - n_ee} EE", flush=True)
            buf = np.zeros((maxRimVT, 4), dtype=np.int32)
            buf[:n_vt] = vt_np[:n_vt]
            wp.copy(self.rimVTPairs, wp.array(buf, dtype=wp.int32))
            buf = np.zeros((maxRimEE, 4), dtype=np.int32)
            buf[:n_ee] = ee_np[:n_ee]
            wp.copy(self.rimEEPairs, wp.array(buf, dtype=wp.int32))
            self.rimVTCount.fill_(n_vt)
            self.rimEECount.fill_(n_ee)
            self._rimDirty = False

        # Settle-interleaved recovery (see uncrossInterleave): on a quiescent
        # frame whose sweep found a large wad, resolve again between substep
        # chunks -- each pass then acts on constraint-relaxed geometry.
        # Moving pos between chunks is exactly as safe as between frames: the
        # next substep's integrate freezes the flipped positions into
        # prev_pos, so no kinetic energy is injected.
        if uncrossInterleave > 1 and recovery_found > uncrossBurstN:
            inter = uncrossInterleaveBig \
                if recovery_found > uncrossInterleaveBigN else uncrossInterleave
            per = (steps + inter - 1) // inter
            done = 0
            while done < steps:
                cnt = min(per, steps - done)
                for _ in range(cnt):
                    wp.capture_launch(graph)
                done += cnt
                if done < steps:
                    self._resolve_crossings()
        else:
            for _ in range(steps):
                wp.capture_launch(graph)

        wp.copy(self.hostPos, self.pos)
        # hostPos is a PINNED cpu array, so the D2H copy above is issued as an
        # async cudaMemcpyAsync on the stream; .numpy() on a cpu array does NOT
        # sync the stream, so host readers (harnesses, update_anchors/drag_anchor)
        # would otherwise read the pinned buffer while the copy is still in flight,
        # yielding stale positions (looks like deep collider penetration during a
        # fast sphere drag). Sync so the buffer is complete before any host read.
        wp.synchronize_stream()

        # Load-yielding grip: read back this frame's per-grab plow pressure, keyed
        # by each grab's primary particle id, for next update_anchors to yield on.
        if self.activeGrabs and n_members > 0:
            pv = self.grabPressure.numpy()
            pm = self.grabPressureMag.numpy()
            self.grabPressureHost = {
                int(g[2][0]): (pv[gi].copy(), float(pm[gi]), max(len(g[2]), 1))
                for gi, g in enumerate(self.activeGrabs[:maxGrabs])}
        else:
            self.grabPressureHost = {}

        # Crossing-brake signal (see GRAB_BRAKE): per-grab count of exact
        # crossings near the anchor, from this frame's sweep (the stream is
        # already synced above, so the reads are cheap and coherent). Keyed by
        # primary particle id like grabPressureHost; consumed by the next
        # update_anchors.
        self.crossNearHost = {}
        self._lastCrossTotal = 0
        if getattr(self, "_crossSwept", False) and grabBrakeK > 0.0 \
                and self.activeGrabs:
            n_cross = int(self.crossCount.numpy()[0])
            self._lastCrossTotal = n_cross
            if n_cross > 0:
                pairs = self.crossPairs.numpy()[:min(n_cross, maxCross)]
                P = self.hostPos.numpy()
                mids = 0.5 * (P[self.hostEdgeIds[pairs[:, 0], 0]]
                              + P[self.hostEdgeIds[pairs[:, 0], 1]])
                for g in self.activeGrabs[:maxGrabs]:
                    anch = np.asarray(g[1], dtype=np.float64)
                    self.crossNearHost[int(g[2][0])] = int(
                        (np.linalg.norm(mids - anch, axis=1) < grabBrakeR).sum())

    def step(self, dt: float, iterations=numIterations, integrate=True, self_collision=True, solve_constraints=True,
             build_grid=True):
        # Phase 0: rebuild the hash grid over the start-of-substep points. Before
        # integrate runs, self.pos still holds the previous substep's end position,
        # which integrate is about to freeze into self.prevPos -> the grid bins the
        # EXACT reference positions the detect kernels distance-check against. A
        # build is ~0.07 ms of GPU work vs the ~1.2 ms the LBVH broadphase queries
        # used to take. HashGrid.build is graph-capturable on this stack (mempool
        # allocator; verified to rebuild correctly on every replay); the capture-
        # once structure in simulate() amortizes the expensive instantiation of
        # its alloc nodes. build_grid=False lets a caller that already built the
        # grid for this state skip it. The LBVH is no longer refit here -- it only
        # serves rendering/raycast/diagnostics (update_mesh / count_self_contacts
        # refit it on demand).
        if self_collision and build_grid:
            self.grid.build(self.pos, gridCellSize)

        if integrate:
            wp.launch(kernel=Cloth.integrate,
                      dim=self.numParticles,
                      inputs=[
                          dt,
                          self.invMass,
                          self.prevPos,
                          self.pos,
                          self.vel,
                      ])

        # Sweep grabbed patches by one substep slice (no-ops when nothing is
        # grabbed: anchorMemberCount is 0). After integrate so prev_pos holds the
        # pre-move state and the move is a proper swept displacement.
        wp.launch(kernel=Cloth.advance_grab_anchors,
                  dim=maxGrabs,
                  inputs=[self.anchorPos, self.anchorDelta])
        wp.launch(kernel=Cloth.apply_grab_anchors,
                  dim=maxGrabMembers,
                  inputs=[self.anchorMemberCount, self.anchorMemberIds,
                          self.anchorMemberAx, self.anchorMemberOff,
                          self.anchorPos, self.pos])

        if solve_constraints:
            self.distConstraints.lambdas.zero_()
            self.bendConstraints.lambdas.zero_()
            # Bending stability guard. The bending constraint runs in XPBD's
            # compliance-dominated regime (alpha = 1/(ke*dt^2) ~ 8-32 >> the gradient
            # denominator ~ 1e-3), so its per-substep correction behaves like an
            # EXPLICIT force impulse whose stability product scales as
            # relaxation * ke * dt^2. The frame-level bending impulse is
            # dt-invariant (correction/substep ~ dt^2, substeps/frame ~ 1/dt), but at
            # dt > bendStabilityDt the per-substep overshoot crosses the Jacobi
            # stability boundary: a crumpled cloth then never settles -- it "undulates"
            # indefinitely (measured: settled drape frame-to-frame motion 2-5e-2 m at
            # 30 substeps vs 3e-4 at 60; with this guard 30 settles to ~1e-4). Scaling
            # the bending relaxation by (bendStabilityDt/dt)^2 keeps that stability
            # product exactly at its tuned-and-validated 60-substep value: a strict
            # no-op (factor 1.0) at >= 60 substeps, and the distance groups --
            # projection-dominated (denominator >> alpha), unconditionally stable --
            # are never touched.
            bend_relax_scale = min(1.0, (bendStabilityDt / dt) ** 2)
            for iteration in range(iterations):
                for offset, count, kernel, indices, rests, lambdas, ke, kd, parallel, relaxation in self.constraints:
                    if kernel is Cloth.bending_constraints:
                        relaxation = relaxation * bend_relax_scale
                    if parallel:
                        wp.launch(kernel=kernel,
                                  dim=count,
                                  inputs=[
                                      dt,
                                      ke,
                                      kd,
                                      relaxation,
                                      offset,
                                      self.pos,
                                      self.prevPos,
                                      self.invMass,
                                      indices,
                                      rests,
                                      lambdas,
                                      self.deltas,
                                  ])
                        wp.launch(kernel=Cloth.add_deltas,
                                  dim=self.numParticles,
                                  inputs=[self.pos, self.deltas])
                    else:
                        wp.launch(kernel=kernel,
                                  dim=count,
                                  inputs=[
                                      dt,
                                      ke,
                                      kd,
                                      relaxation,
                                      offset,
                                      self.pos,
                                      self.prevPos,
                                      self.invMass,
                                      indices,
                                      rests,
                                      lambdas,
                                      self.pos,
                                  ])

        # Strain limiter: hard-cap edge elongation at maxStrain before anything
        # downstream reads pos. Inert in normal states (solve strain ~1-2%); under
        # violent drags it prevents the 25-100x stretch "wads" that explode
        # self-collision candidate density (the >1s frame cliff) and read as the
        # cloth diverging. deltas is free here (solve batches re-zero before use).
        for _sl_it in range(strainLimitIters):
            wp.launch(kernel=Cloth.strain_limit,
                      dim=self.numEdges,
                      inputs=[self.invMass, self.pos, self.edgeIds, self.edgeRestLen],
                      outputs=[self.deltas])
            if _sl_it == 0 and ringFloorEnable:
                # ring_floor is a rare-fire guard (fold-through only): once per
                # block converges across substeps; 12x/substep cost ~3 ms/frame.
                wp.launch(kernel=Cloth.ring_floor,
                          dim=self.numRingPairs,
                          inputs=[self.invMass, self.pos, self.ringPairs],
                          outputs=[self.deltas])
            wp.launch(kernel=Cloth.add_deltas,
                      dim=self.numParticles,
                      inputs=[self.pos, self.deltas])

        # Displacement governor: bound the post-solve trial displacement (and
        # sanitize NaN/Inf) before the self-collision broadphase reads pos, so an
        # instability spike cannot balloon the swept query AABB into an O(tris)
        # (multi-second) frame. No-op on contract-valid substeps.
        wp.launch(kernel=Cloth.clamp_displacement,
                  dim=self.numParticles,
                  inputs=[self.invMass, self.prevPos, self.pos,
                          self.truncation_ts, self.push, self.pushLimit])

        # Planar Divide-and-Truncate: clamp the net per-vertex displacement so no
        # vertex crosses a plane that separated it from a nearby triangle (vertex-
        # triangle) or a nearby edge (edge-edge) at the frozen reference state,
        # recovering feasibility where a pair already overlaps. Both passes write
        # the same truncation_ts (atomic_min) and push (atomic_add) buffers.
        if self_collision:
            # Global frozen-geometry bound for this substep (longest edge): the
            # grid query radii add min(bounds[0], edgeLenCap) -- moderate
            # stretch widens the search, while primitives stretched past the
            # cap go through the oversized fallback instead of inflating every
            # radius.
            self.detectBounds.zero_()
            wp.launch(kernel=Cloth.max_edge_length,
                      dim=boundsReduceThreads,
                      inputs=[self.prevPos, self.edgeIds],
                      outputs=[self.detectBounds])
            # Per-primitive frozen edge lengths (detect's per-candidate slack)
            # and the oversized partition for this substep's frozen state:
            # edges/faces with a frozen edge length > edgeLenCap are EXCLUDED
            # from the capped-radius vertex walk (gather/expand skip them) and
            # handled by the oversized fallback below instead.
            self.oversizedCount.zero_()
            wp.launch(kernel=Cloth.collect_oversized,
                      dim=self.numEdges,
                      inputs=[self.invMass, self.prevPos, self.pos,
                              self.edgeIds, self.triIds, self.edgeFaceIds],
                      outputs=[self.edgeLen, self.oversizedIds,
                               self.oversizedCount,
                               self.ovF0, self.ovF1, self.ovApex,
                               self.ovSweep, self.ovPinned])
            wp.launch(kernel=Cloth.face_longest_edges,
                      dim=self.numTris,
                      inputs=[self.prevPos, self.triIds],
                      outputs=[self.faceLongest])
            # DETECT (two stages; see the kernels for the radius derivations):
            # gather = one LEAN hash-grid walk per vertex -> neighbor cache;
            # expand = query-free expansion of the cache into both candidate
            # sets. vtCount/nbrCount are written for every vertex; eeCount is
            # filled by scattered atomic appends so it needs the pre-zero.
            self.eeCount.zero_()
            self.vtCount.zero_()
            wp.launch(kernel=Cloth.detect_gather,
                      dim=self.numParticles,
                      inputs=[self.grid.id, self.invMass, self.prevPos, self.pos,
                              self.gridRC, self.edgeIds,
                              self.vertEdgeOff, self.vertEdgeIds,
                              self.edgeLen, self.detectBounds],
                      outputs=[self.nbrCount, self.nbrBuf, self.selfCollisionOverflow])
            wp.launch(kernel=Cloth.detect_expand,
                      dim=(self.numParticles, expandK),
                      inputs=[self.invMass, self.prevPos, self.pos,
                              self.gridRC, self.triIds, self.edgeIds,
                              self.vertFaceOff, self.vertFaceIds,
                              self.vertEdgeOff, self.vertEdgeIds,
                              self.faceLongest, self.edgeLen,
                              self.detectBounds, self.nbrCount, self.nbrBuf],
                      outputs=[self.vtCount, self.vtBuf, self.eeCount, self.eeBuf,
                               self.selfCollisionOverflow])
            # FALLBACK: every vertex scans the compact oversized list (the
            # primitives the capped walk excluded) into the same candidate
            # buffers; a second edge-parallel pass pairs the oversized set
            # against itself (mid-span crossings). Both kernels append
            # atomically to the pre-zeroed counters (expand runs first in
            # stream order). Both early-out to ~one load when no edge is
            # stretched past edgeLenCap.
            wp.launch(kernel=Cloth.scatter_oversized,
                      dim=self.numParticles,
                      inputs=[self.invMass, self.prevPos, self.pos,
                              self.gridRC, self.triIds, self.edgeIds,
                              self.vertEdgeOff, self.vertEdgeIds,
                              self.edgeLen, self.oversizedIds,
                              self.oversizedCount,
                              self.ovF0, self.ovF1, self.ovApex,
                              self.ovSweep, self.ovPinned],
                      outputs=[self.vtCount, self.vtBuf, self.eeCount, self.eeBuf,
                               self.selfCollisionOverflow])
            wp.launch(kernel=Cloth.oversized_pairs,
                      dim=self.numEdges,
                      inputs=[self.invMass, self.prevPos, self.pos,
                              self.gridRC, self.edgeIds, self.edgeLen,
                              self.oversizedIds, self.oversizedCount,
                              self.ovSweep],
                      outputs=[self.eeCount, self.eeBuf,
                               self.selfCollisionOverflow])
            if farDebug:
                wp.copy(self.dbgPosDet, self.pos)
            # CONTACT REPULSION: soft unilateral pressure over the cached
            # candidate pairs, on CURRENT positions, sandwiched between the
            # candidate build and the truncation narrowphase so the PDT planes
            # (built from the frozen reference, applied to the NET displacement
            # including these deltas) veto any repulsion overshoot. deltas is
            # free here (the strain-limit block re-zeroed it via add_deltas).
            for _rp_it in range(repulsionIters):
                wp.launch(kernel=Cloth.contact_repulsion,
                          dim=self.numParticles,
                          inputs=[self.triIds, self.invMass, self.pos,
                                  self.prevPos, self.vtCount, self.vtBuf],
                          outputs=[self.deltas])
                # EE pass on the first repulsionEE iterations only: eeBuf is
                # ~3x vtBuf, so the edge pass dominates the repulsion cost;
                # one EE pass keeps the drape frame time at the control mean
                # while two blew the 10% budget (+19%).
                if repulsionEE > _rp_it:
                    wp.launch(kernel=Cloth.contact_repulsion_edges,
                              dim=self.numEdges,
                              inputs=[self.invMass, self.pos, self.prevPos,
                                      self.edgeIds, self.eeCount, self.eeBuf],
                              outputs=[self.deltas])
                wp.launch(kernel=Cloth.add_deltas,
                          dim=self.numParticles,
                          inputs=[self.pos, self.deltas])
            # PINCH EXTRUSION: bounded tangential escape for fabric squeezed
            # between the shell and the floor (pressure-gated by vtCount). Runs
            # before the narrowphase so the PDT planes veto any extrusion step
            # that would cross a neighboring sheet.
            wp.launch(kernel=Cloth.pinch_extrude,
                      dim=self.numParticles,
                      inputs=[self.invMass, self.pos,
                              self.colliderCenter, self.colliderRadius,
                              self.colliderDeltaC, self.colliderDeltaR,
                              self.vtCount],
                      outputs=[self.deltas])
            wp.launch(kernel=Cloth.add_deltas,
                      dim=self.numParticles,
                      inputs=[self.pos, self.deltas])
            if farDebug:
                wp.copy(self.dbgPosNar, self.pos)
            # NARROWPHASE (query-free -> high occupancy). Reads the caches, does the
            # DIVIDE/TRUNCATE; both passes write the same truncation_ts (atomic_min) and
            # push (atomic_add) buffers as before.
            wp.launch(kernel=Cloth.self_collision_truncate,
                      dim=self.numParticles,
                      inputs=[
                          self.triIds,
                          self.invMass,
                          self.prevPos,
                          self.pos,
                          self.vtCount,
                          self.vtBuf,
                          self.anchorCount,
                          self.crossedFlag,
                      ],
                      outputs=[
                          self.truncation_ts,
                          self.push,
                          self.pushLimit,
                      ])
            wp.launch(kernel=Cloth.self_collision_truncate_edges,
                      dim=self.numEdges,
                      inputs=[
                          self.invMass,
                          self.prevPos,
                          self.pos,
                          self.edgeIds,
                          self.eeCount,
                          self.eeBuf,
                          self.anchorCount,
                          self.crossedFlag,
                      ],
                      outputs=[
                          self.truncation_ts,
                          self.push,
                          self.pushLimit,
                      ])
            # Rim pair solver (see RIM_SOLVER): the grab rim's ring-1 pairs --
            # culled from every pass above by design -- get their own reduced-
            # offset DIVIDE/TRUNCATE into the same truncation_ts/push buffers.
            # No-op (early-out on count) when nothing is grabbed.
            if rimSolverEnable:
                wp.launch(kernel=Cloth.rim_truncate_vt,
                          dim=maxRimVT,
                          inputs=[self.invMass, self.prevPos, self.pos,
                                  self.rimVTPairs, self.rimVTCount],
                          outputs=[self.truncation_ts, self.push])
                wp.launch(kernel=Cloth.rim_truncate_ee,
                          dim=maxRimEE,
                          inputs=[self.invMass, self.prevPos, self.pos,
                                  self.rimEEPairs, self.rimEECount],
                          outputs=[self.truncation_ts, self.push])
            wp.launch(kernel=Cloth.apply_truncation,
                      dim=self.numParticles,
                      inputs=[self.invMass, self.prevPos, self.pos, self.truncation_ts, self.push,
                              self.pushLimit,
                              self.colliderCenter, self.colliderRadius,
                              self.colliderDeltaC, self.colliderDeltaR])
            if farDebug:
                wp.copy(self.dbgPosPDT, self.pos)
            # Load-yielding grip: fold this substep's grabbed-member pressure
            # shares into the per-grab frame accumulators (no-op when nothing is
            # grabbed: anchorMemberCount is 0).
            wp.launch(kernel=Cloth.accumulate_grab_pressure,
                      dim=maxGrabMembers,
                      inputs=[self.anchorMemberCount, self.anchorMemberIds,
                              self.anchorMemberAx, self.push,
                              self.grabPressure, self.grabPressureMag])

        # Swept-CCD sphere projection + ground contact run last so their
        # penetration-free result has final say over the substep. The sphere's
        # per-frame delta (dc/dr/dq) is sliced to PER-SUBSTEP so the swept test sees
        # the true sphere velocity (dc/frame_dt) instead of numSubsteps x too fast
        # (which flung hit particles). collider_project reads the CURRENT substep
        # center/radius from self.collider* device arrays, which advance_sphere then
        # increments so each substep's end pose walks from the frame-start pose to the
        # rendered end pose (center+dc) -- no tunneling, no lag/penetration. The
        # per-substep delta slices are read from self.colliderDelta* device arrays
        # (set per frame by simulate) so the captured graph is frame-invariant.
        wp.launch(kernel=Cloth.collider_project,
                  dim=self.numParticles,
                  inputs=[
                      dt,
                      self.invMass,
                      self.prevPos,
                      self.pos,
                      self.colliderCenter,
                      self.colliderRadius,
                      self.colliderDeltaC,
                      self.colliderDeltaR,
                      self.colliderDeltaQ,
                      self.vtCount,
                  ])
        # Edge-vs-sphere pass: keeps the FABRIC (not just its vertices) out of the
        # sphere -- under load the cloth stretches until the sphere fits between
        # particles, and the vertex pass alone lets it tunnel through stretched edges.
        # Jacobi iterations: applying one edge's push can rotate a neighboring edge
        # into the sphere, and later passes catch it. The pass count is dt-scaled:
        # at the 60-substep reference dt, 2 passes left a 0.014 residual on one
        # extreme-stretch fast-drag trajectory and 3 measured 0 -- but a MORE
        # extreme chaotic rerun of the same gate (30 m/s, max edge 2.36 m) later
        # left 0.0003 with 3 passes, so the 60-substep floor is 5 (measured
        # 0.0000 there); at 30 substeps the per-substep sphere/cloth motion
        # doubles and convergence needs more passes (measured on the 9 m/s
        # 400x400 drag gate: 4 passes left 0.0001-0.0010, 10 measured 0.0000
        # across reruns). Each pass only pushes edges AWAY from the sphere, so
        # extra passes are safe; one pass is ~0.005 ms. deltas is free at this
        # point in the substep (only the constraint solve uses it).
        dt_ratio = max(1, int(math.ceil(dt / bendStabilityDt - 1.0e-6)))
        edge_passes = max(5, 8 * dt_ratio - 6)
        for _ in range(edge_passes):
            wp.launch(kernel=Cloth.collider_project_edges,
                      dim=self.numEdges,
                      inputs=[
                          self.invMass,
                          self.pos,
                          self.edgeIds,
                          self.colliderCenter,
                          self.colliderRadius,
                          self.colliderDeltaC,
                          self.colliderDeltaR,
                      ],
                      outputs=[self.deltas])
            wp.launch(kernel=Cloth.add_deltas,
                      dim=self.numParticles,
                      inputs=[self.pos, self.deltas])
        # Post-collider strain limiting: the collider's nearest-surface ejection can
        # split an edge across the sphere within this substep; cap that stretch
        # BEFORE update_velocity derives velocities from it, or (pos-prev)/dt bakes
        # the stretch into vel and integrate re-creates it next substep (the runaway
        # that made drag frames explode).
        for _sl_it in range(strainLimitPostIters):
            wp.launch(kernel=Cloth.strain_limit,
                      dim=self.numEdges,
                      inputs=[self.invMass, self.pos, self.edgeIds, self.edgeRestLen],
                      outputs=[self.deltas])
            if _sl_it == 0 and ringFloorEnable:
                wp.launch(kernel=Cloth.ring_floor,
                          dim=self.numRingPairs,
                          inputs=[self.invMass, self.pos, self.ringPairs],
                          outputs=[self.deltas])
            wp.launch(kernel=Cloth.add_deltas,
                      dim=self.numParticles,
                      inputs=[self.pos, self.deltas])
        # ...then let the edge-vs-sphere collider have FINAL say: the limiter's
        # step-capped pulls (<= ~0.5*rest per edge per iteration) can drag an edge
        # a few mm back into the sphere; without this closing block the frame ends
        # with fabric penetration (measured 0.003-0.027 at 9-30 m/s drags). The
        # re-ejection re-adds only that mm-scale stretch, which the next substep's
        # limiter absorbs -- alternating projections onto two compatible sets.
        for _ in range(3):
            wp.launch(kernel=Cloth.collider_project_edges,
                      dim=self.numEdges,
                      inputs=[
                          self.invMass,
                          self.pos,
                          self.edgeIds,
                          self.colliderCenter,
                          self.colliderRadius,
                          self.colliderDeltaC,
                          self.colliderDeltaR,
                      ],
                      outputs=[self.deltas])
            wp.launch(kernel=Cloth.add_deltas,
                      dim=self.numParticles,
                      inputs=[self.pos, self.deltas])

        # Advance the collider pose only after every consumer of this substep's
        # pose (vertex pass, edge passes, post-limit closing block) has run.
        # Fingertip collider: after the shell/ground have final-said, evict
        # free fabric from the grip volume (no-op when nothing is grabbed).
        if farDebug:
            wp.copy(self.dbgPosCol, self.pos)
        wp.launch(kernel=Cloth.fingertip_project,
                  dim=self.numParticles,
                  inputs=[self.invMass, self.anchorCount, self.anchorPos,
                          self.anchorDelta, self.prevPos, self.pos,
                          self.pushLimit])
        wp.launch(kernel=Cloth.advance_sphere,
                  dim=1,
                  inputs=[self.colliderCenter, self.colliderRadius,
                          self.colliderDeltaC, self.colliderDeltaR])

        wp.launch(kernel=Cloth.update_velocity,
                  dim=self.numParticles,
                  inputs=[
                      dt,
                      self.pos,
                      self.prevPos,
                  ],
                  outputs=[self.vel])

    def update_mesh(self):
        self.normals.zero_()
        wp.launch(kernel=Cloth.add_normals,
                  dim=self.numTris,
                  inputs=[self.pos, self.triIds, self.normals])
        wp.launch(kernel=Cloth.normalize_normals,
                  dim=self.numParticles,
                  inputs=[self.normals])

    def init(self, **kwargs):
        self._quad = gluNewQuadric()

        # Simulation state lives in PLAIN device arrays (identical to
        # init_headless): the captured graphs keep stable pointers, and CUDA
        # never holds a GL buffer mapped across the frame. Previously pos /
        # normals / triIds were permanently-mapped RegisteredGLBuffers that
        # glDrawElements sourced while still CUDA-mapped -- a pattern that can
        # force driver-level serialization between compute and raster. The GL
        # buffers are now refreshed in render() via a brief map -> copy -> unmap
        # (two ~2 MB device-device copies, ~0.1 ms).
        self.pos = wp.clone(self.restPos)
        self.normals = wp.zeros(self.numParticles, dtype=wp.vec3)
        self.triIds = wp.array(self.hostTriIds, dtype=wp.int32)

        host_pos = self.hostPos.numpy()
        glGenBuffers(1, ctypes.pointer(self.pos_gl_buffer))
        glBindBuffer(GL_ARRAY_BUFFER, self.pos_gl_buffer)
        glBufferData(GL_ARRAY_BUFFER, host_pos.nbytes, host_pos, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._pos_gl = wp.RegisteredGLBuffer(int(self.pos_gl_buffer.value),
                                             flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                                             fallback_to_copy=False)

        normals = np.zeros((self.numParticles, 3), dtype=np.float32)
        glGenBuffers(1, ctypes.pointer(self.normals_gl_buffer))
        glBindBuffer(GL_ARRAY_BUFFER, self.normals_gl_buffer)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._normals_gl = wp.RegisteredGLBuffer(int(self.normals_gl_buffer.value),
                                                 flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                                                 fallback_to_copy=False)

        # The index buffer never changes: plain static GL upload, no interop.
        tri_ids = self.hostTriIds
        glGenBuffers(1, ctypes.pointer(self.triIds_gl_buffer))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.triIds_gl_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, tri_ids.nbytes, tri_ids, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        # Warm up the hash grid: the first build allocates the cell tables and
        # sort scratch (must happen outside CUDA-graph capture).
        self.grid.build(self.pos, gridCellSize)

        self.colliderCenter.fill_(sphere.center)
        self.colliderRadius.fill_(sphere.radius)

        wp.launch(kernel=Cloth.rest_distances,
                  dim=(self.distConstraints.count,),
                  inputs=[self.pos, self.distConstraints.indices, self.distConstraints.rests])

    def init_headless(self):
        # GL-free counterpart to init(): allocate pos/normals/triIds as plain Warp
        # arrays (no OpenGL interop) so the simulation can run headless for testing
        # and benchmarking. Simulation state and kernels are otherwise identical.
        self.pos = wp.clone(self.restPos)
        self.normals = wp.zeros(self.numParticles, dtype=wp.vec3)
        self.triIds = wp.array(self.hostTriIds, dtype=wp.int32)  # (numTris, 3)

        # Warm up the hash grid (first build allocates; matches init()).
        self.grid.build(self.pos, gridCellSize)

        self.colliderCenter.fill_(sphere.center)
        self.colliderRadius.fill_(sphere.radius)

        wp.launch(kernel=Cloth.rest_distances,
                  dim=(self.distConstraints.count,),
                  inputs=[self.pos, self.distConstraints.indices, self.distConstraints.rests])

    def pre_render(self, **kwargs):
        now = time.perf_counter()
        if _perf.t_last is not None:
            _perf.total += now - _perf.t_last
        _perf.t_last = now

        self.update_anchors()

        t0 = time.perf_counter()
        if state & (State.RUN | State.STEP):
            self.simulate(steps=numSubsteps if State.FRAME_STEP in state else 1,
                          integrate=State.SOLVER_STEP not in state,
                          self_collision=State.SELF_COLLISION in state and State.SOLVER_STEP not in state,
                          solve_constraints=State.CONTACT_STEP not in state)
        _perf.sim += time.perf_counter() - t0  # simulate() ends with a stream sync

        self.update_mesh()

    def render(self, **kwargs):
        t0 = time.perf_counter()
        # Upload the sim results to the GL buffers: brief map -> device copy ->
        # unmap (unmap orders the copy against subsequent GL reads).
        dst = self._pos_gl.map(dtype=wp.vec3, shape=(self.numParticles,))
        wp.copy(dst, self.pos)
        self._pos_gl.unmap()
        dst = self._normals_gl.map(dtype=wp.vec3, shape=(self.numParticles,))
        wp.copy(dst, self.normals)
        self._normals_gl.unmap()
        # Make sure all the CUDA operations have completed before calling OpenGL
        wp.synchronize_stream()

        glColor3f(1.0, 0.0, 0.0)
        glNormal3f(0.0, 0.0, -1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if State.WIREFRAME in state else GL_FILL)
        glLineWidth(1.0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.pos_gl_buffer)
        glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ARRAY_BUFFER, self.normals_gl_buffer)
        glNormalPointer(GL_FLOAT, 0, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.triIds_gl_buffer)
        if State.CULL_FACE in state:
            glCullFace(GL_FRONT)
            glColor3f(1.0, 0.0, 0.0)
            glDrawElements(GL_TRIANGLES, 3 * self.numTris, GL_UNSIGNED_INT, None)
            glCullFace(GL_BACK)
            glColor3f(1.0, 1.0, 0.0)
            glDrawElements(GL_TRIANGLES, 3 * self.numTris, GL_UNSIGNED_INT, None)
        else:
            glDisable(GL_CULL_FACE)
            glColor3f(1.0, 0.0, 0.0)
            glDrawElements(GL_TRIANGLES, 3 * self.numTris, GL_UNSIGNED_INT, None)
            glEnable(GL_CULL_FACE)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

        # kinematic particles / anchors
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        host_pos = self.hostPos.numpy()
        for anchor in filter(lambda a: a.flags & (AnchorFlag.ACTIVE | AnchorFlag.LOCKED), self.anchors):
            if anchor.flags & AnchorFlag.SELECTED:
                glColor3f(0 / 255, 145 / 255, 255 / 255)
            else:
                glColor3f(1.0, 1.0, 1.0)
            glPushMatrix()
            pos = host_pos[anchor.id]
            glTranslatef(pos[0], pos[1], pos[2])
            gluSphere(self._quad, 0.02, 40, 40)
            glPopMatrix()

        # Frame-time breakdown, printed once per _perf.PERIOD frames: `sim` is the
        # captured-graph replay (wall, incl. its stream sync), `draw` this method
        # (buffer upload + GL command issue), `other` = everything else in the
        # frame (update_mesh kernels, GL rasterization + page flip, input stack).
        _perf.draw += time.perf_counter() - t0
        _perf.frames += 1
        if _perf.frames == _perf.PERIOD:
            f = float(_perf.frames)
            total = _perf.total / f * 1e3
            sim = _perf.sim / f * 1e3
            draw = _perf.draw / f * 1e3
            ovf = int(self.selfCollisionOverflow.numpy()[0])  # last frame's drops
            print(f"[perf] fps={1e3 / total:5.1f}  frame={total:6.1f} ms  "
                  f"sim={sim:6.1f}  draw={draw:5.1f}  other={total - sim - draw:6.1f}"
                  + (f"  <<< candidate overflow={ovf} (contacts dropped)" if ovf else ""),
                  flush=True)
            _perf.frames = 0
            _perf.total = _perf.sim = _perf.draw = 0.0

    def reset(self):
        self.vel.zero_()
        wp.copy(self.pos, self.restPos)
        wp.copy(self.prevPos, self.restPos)
        wp.copy(self.hostPos, self.restPos)
        for anchor in self.anchors:
            anchor.flags &= ~(AnchorFlag.ACTIVE | AnchorFlag.LOCKED)
        # self.update_anchors()


class Ground(Input):

    NORMAL = wp.vec3(0.0, 1.0, 0.0)

    def __init__(self):
        super().__init__("ground")
        num_tiles = 30
        tile_size = 0.5
        vertices = np.zeros(3 * 4 * num_tiles * num_tiles, dtype=float)
        colors = np.zeros(3 * 4 * num_tiles * num_tiles, dtype=float)
        square = [[0, 0], [0, 1], [1, 1], [1, 0]]
        r = num_tiles / 2.0 * tile_size
        for xi in range(num_tiles):
            for zi in range(num_tiles):
                x = (-num_tiles / 2.0 + xi) * tile_size
                z = (-num_tiles / 2.0 + zi) * tile_size
                p = xi * num_tiles + zi
                for i in range(4):
                    q = 4 * p + i
                    px = x + square[i][0] * tile_size
                    pz = z + square[i][1] * tile_size
                    vertices[3 * q] = px
                    vertices[3 * q + 2] = pz
                    col = 0.4
                    if (xi + zi) % 2 == 1:
                        col = 0.8
                    pr = math.sqrt(px * px + pz * pz)
                    d = max(0.0, 1.0 - pr / r)
                    col = col * d
                    for j in range(3):
                        colors[3 * q + j] = col
        self.colors = colors
        self.vertices = vertices

    def render(self, frame, time):
        glColor3f(1.0, 1.0, 1.0)
        glNormal3f(0.0, 1.0, 0.0)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glColorPointer(3, GL_FLOAT, 0, self.colors)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDrawArrays(GL_QUADS, 0, math.floor(len(self.vertices) / 3))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)


class Sphere(Input):

    def __init__(self, center: wp.vec3, radius: float):
        super().__init__("sphere")
        # TODO: use a wp.transformation
        self.center = center
        self.quat = quat_from_unit_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 1.0, 0.0))
        self.radius = radius
        self.dc = wp.vec3()
        self.dq = wp.quat_identity()
        self.dr = 0.0
        self._quad = None

    def init(self, **kwargs):
        self._quad = gluNewQuadric()

    def translate(self, dc: wp.vec3):
        c = self.center + dc
        # Rest ON the floor, never through it: leave room for one cloth layer
        # under the shell so squeezed fabric keeps an escape corridor. A center
        # clamped at y=0 buries half the sphere and crushes cloth into negative
        # space (unrecoverable entanglement).
        floor_y = self.radius + self.dr + 2.0 * (thickness + particleRadius) + d_offset
        if c[1] < floor_y:
            c[1] = floor_y
            dc = c - self.center
        self.dc += dc
        # Cap the per-frame sphere motion at what the (velocity-bounded) contact
        # response can absorb: with bounded projections the fabric escapes a
        # plowing sphere cleanly up to ~18 m/s (measured); beyond that it is
        # transiently run over. 0.5/frame = 15 m/s at 30 fps -- an extreme flick
        # already; faster gestures rubber-band.
        l = wp.length(self.dc)
        if l > 0.5:
            self.dc = self.dc * (0.5 / l)

    def rotate(self, dq: wp.quat):
        self.dq = dq * self.dq

    def resize(self, dr: float):
        self.dr += dr
        # A sphere inflated at low height must not grow through the floor:
        # lift the center along with the shell (same margin as translate).
        floor_y = self.radius + self.dr + 2.0 * (thickness + particleRadius) + d_offset
        if self.center[1] + self.dc[1] < floor_y:
            lift = wp.vec3(0.0, floor_y - self.center[1] - self.dc[1], 0.0)
            self.dc += lift

    def render(self, **kwargs):
        if (not state & (State.RUN | State.STEP)
                and (wp.length(self.dc) > 0.0 or self.dr != 0.0)):
            self.draw(self.center, self.radius, self.quat, fill=False)
        self.draw(self.center + self.dc, self.radius + self.dr, self.dq * self.quat)

    def draw(self, pos: wp.vec3, rad: float, quat: wp.quat, fill=True, line=True):
        rot = wp.quat_to_matrix(quat)
        glPushMatrix()
        glMultMatrixf(wp.mat44(
            rot[0, 0], rot[1, 0], rot[2, 0], 0.0,
            rot[0, 1], rot[1, 1], rot[2, 1], 0.0,
            rot[0, 2], rot[1, 2], rot[2, 2], 0.0,
            pos[0],    pos[1],    pos[2],    1.0,
        ))
        if fill:
            glColor3f(0.8, 0.8, 0.8)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            gluSphere(self._quad, rad, 40, 40)
        if line:
            glColor3f(0.75, 0.75, 0.75)
            glLineWidth(2.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            gluSphere(self._quad, rad, 40, 40)
        glPopMatrix()

    def post_render(self, **kwargs):
        if state & (State.RUN | State.STEP):
            # One simulation step has run,
            # so changes have been integrated and can be reset
            self.center += self.dc
            self.quat = self.dq * self.quat
            self.radius += self.dr
            self.dc = wp.vec3()
            self.dq = wp.quat_identity()
            self.dr = 0.0


class Camera(Input):

    UP = wp.vec3(0.0, 1.0, 0.0)
    RIGHT = wp.vec3(1.0, 0.0, 0.0)
    EPS = 0.000001
    MIN_DISTANCE = 2.0
    MAX_DISTANCE = 100.0

    def __init__(self):
        super().__init__("camera")
        self.pos = wp.vec3(0.0, 1.0, 5.0)
        self.forward = wp.vec3(0.0, 0.0, -1.0)
        self.up = wp.vec3(0.0, 1.0, 0.0)
        self.right = wp.cross(self.forward, self.up)
        self.target = wp.vec3(0.0, 1.0, 0.0)
        self.quat = quat_from_unit_vectors(self.up, Camera.UP)

    def init(self, width, height):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, float(width) / float(height), 0.01, 1000.0)

    def pre_render(self, **kwargs):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            self.pos[0], self.pos[1], self.pos[2],
            self.pos[0] + self.forward[0], self.pos[1] + self.forward[1], self.pos[2] + self.forward[2],
            self.up[0], self.up[1], self.up[2])

    def orbit(self, dx, dy, gain=0.001):
        self.rotate(-wp.TAU * dx, -wp.TAU * dy, gain)

    def rotate(self, dth, dph, gain=1.0):
        vec = self.pos - self.target
        vec = wp.quat_rotate(self.quat, vec)

        radius = wp.length(vec)
        theta = wp.atan2(vec[0], vec[2])
        theta += dth * gain

        phi = wp.acos(wp.clamp(vec[1] / radius, - 1.0, 1.0))
        phi += dph * gain
        phi = wp.max(Camera.EPS, wp.min(wp.pi - Camera.EPS, phi))

        sin_phi_radius = wp.sin(phi) * radius
        vec = wp.vec3(sin_phi_radius * wp.sin(theta), wp.cos(phi) * radius, sin_phi_radius * wp.cos(theta))
        vec = wp.quat_rotate_inv(self.quat, vec)

        self.pos = self.target + vec
        self.forward = self.target - self.pos
        self.forward = wp.normalize(self.forward)
        self.right = wp.cross(self.forward, Camera.UP)
        self.right = wp.normalize(self.right)
        self.up = wp.cross(self.right, self.forward)
        self.up = wp.normalize(self.up)

    def dolly(self, delta, gain=0.1):
        dist = wp.length(self.target - self.pos)
        delta *= gain * dist / Camera.MIN_DISTANCE
        delta = dist - wp.clamp(dist - delta, Camera.MIN_DISTANCE, Camera.MAX_DISTANCE)
        self.pos += delta * self.forward

    def dolly_scale(self, scale):
        dist = wp.length(self.target - self.pos) / scale
        dist = wp.clamp(dist, Camera.MIN_DISTANCE, Camera.MAX_DISTANCE)
        self.pos = self.target - dist * self.forward

    def track(self, dx, dy, gain=0.001):
        gain *= wp.length(self.target - self.pos) / Camera.MIN_DISTANCE
        track_x = gain * dx * self.right
        track_y = gain * dy * self.up
        self.pos -= track_x
        self.pos += track_y
        self.target -= track_x
        self.target += track_y


class Mouse(shaderbang.input.Mouse):

    def __init__(self):
        super().__init__("mouse")
        self.particle: Optional[Particle] = None

    def pre_render(self, **kwargs):
        if self.deltaW != 0:
            camera.dolly(self.deltaW, gain=0.1)
        if self.click and not self.particle:
            if self.button == EV_KEY.BTN_LEFT:
                self.particle = cloth.drag_anchor(self.mouseX, self.mouseY)
        elif self.drag:
            if self.particle:
                self.particle.screen = wp.vec2(self.mouseX, self.mouseY)
            elif self.button == EV_KEY.BTN_LEFT:
                camera.orbit(self.deltaX, self.deltaY, 0.5 / self.resolution[1])
            elif self.button == EV_KEY.BTN_RIGHT:
                camera.track(self.deltaX, self.deltaY, gain=0.001)
        elif self.particle:
            self.particle.flags &= ~AnchorFlag.ACTIVE
            if keyboard.down(any, EV_KEY.KEY_LEFTCTRL, EV_KEY.KEY_RIGHTCTRL):
                self.particle.flags |= AnchorFlag.LOCKED
            if self.particle.click():
                self.particle.flags ^= AnchorFlag.SELECTED
            self.particle = None


class ParticleSlot(TouchSlot):

    def __init__(self):
        super().__init__()
        self.particle: Optional[Particle] = None


class Touchscreen(shaderbang.input.MultiTouch[ParticleSlot]):

    def __init__(self):
        super().__init__("touchscreen", ParticleSlot)

    def holroyd_trackball(self, screen_x, screen_y) -> wp.vec3:
        """
        https://www.khronos.org/opengl/wiki/Object_Mouse_Trackball
        :param screen_x: image plan abscissa
        :param screen_y: image plan ordinate
        :return: the Holroyd's trackball 3D projection
        """
        width, height = self.resolution
        vec = wp.vec3(screen_x / width * 2 - 1.0, - screen_y / height * 2 + 1.0, 0.0)
        len2 = wp.length_sq(vec)
        vec[2] = 0.5 / wp.sqrt(len2) if len2 > 0.5 else wp.sqrt(1.0 - len2)
        return vec

    def pre_render(self, **kwargs):
        slots: list[ParticleSlot] = []
        for slot in self.slots:
            if slot.touch:
                if slot.particle:
                    slot.particle.drop()
                slot.particle = cloth.drag_anchor(slot.touchX, slot.touchY)
            elif slot.drag:
                if slot.particle:
                    slot.particle.screen = wp.vec2(slot.touchX, slot.touchY)
                else:
                    slots.append(slot)
            elif slot.particle:
                slot.particle.flags &= ~AnchorFlag.ACTIVE
                if keyboard.down(any, EV_KEY.KEY_LEFTCTRL, EV_KEY.KEY_RIGHTCTRL):
                    slot.particle.flags |= AnchorFlag.LOCKED
                if slot.particle.click():
                    slot.particle.flags ^= AnchorFlag.SELECTED
                slot.particle = None

        n = len(slots)
        if n == 1:
            slot = slots[0]

            u = quat_from_unit_vectors(Camera.UP, camera.up)
            v = quat_from_unit_vectors(Camera.RIGHT, camera.right)
            quat = wp.mul(u, v)

            vec1 = self.holroyd_trackball(slot.prevX, slot.prevY)
            vec2 = self.holroyd_trackball(slot.touchX, slot.touchY)

            vec1 = wp.quat_rotate(quat, vec1)
            vec2 = wp.quat_rotate(quat, vec2)

            theta = wp.atan2(wp.dot(wp.cross(vec2, vec1), camera.UP), wp.dot(vec2, vec1))
            camera.rotate(wp.PI * theta, - wp.TAU * slots[0].deltaY * 0.5 / self.resolution[1])

        elif n > 1:
            cx = cy = dx = dy = 0.0
            for slot in slots:
                cx += slot.touchX
                dx += slot.deltaX
                cy += slot.touchY
                dy += slot.deltaY
            cx /= n
            cy /= n
            dx /= n
            dy /= n

            for slot in slots:
                slot.prevX += dx
                slot.prevY += dy

            scale, theta, tx, ty = homothety_and_rotation(slots, center=(cx, cy))

            if n < 4:
                camera.track(dx, dy)
                camera.dolly_scale(scale)
                camera.rotate(wp.sign(camera.pos[1]) * theta, 0.0)
            elif n == 4:
                dcx = 0.002 * dx * camera.right
                dcy = -0.002 * dy * camera.up
                dcz = 1.5 * (1.0 - scale) * camera.forward
                sphere.translate(dcx + dcy + dcz)
            else:
                qr = wp.quat_from_axis_angle(camera.forward, wp.PI * theta)
                qx = wp.quat_from_axis_angle(camera.up, wp.TAU * dx / self.resolution[1])
                qy = wp.quat_from_axis_angle(camera.right, wp.TAU * dy / self.resolution[1])
                sphere.rotate(qx * qy * qr)
                dr = wp.clamp(sphere.radius * scale, 0.25, 1.25) - sphere.radius
                sphere.resize(dr)


class Trackpad(shaderbang.input.MultiTouch[TouchSlot]):

    def __init__(self):
        super().__init__("trackpad")

    def pre_render(self, **kwargs):
        slots = [slot for slot in self.slots if slot.drag]
        n = len(slots)
        if n > 1:
            cx = cy = dx = dy = 0.0
            for slot in slots:
                cx += slot.touchX
                dx += slot.deltaX
                cy += slot.touchY
                dy += slot.deltaY
            cx /= n
            cy /= n
            dx /= n
            dy /= n

            for slot in slots:
                slot.prevX += dx
                slot.prevY += dy

            scale, theta, tx, ty = homothety_and_rotation(slots, center=(cx, cy))

            if n == 2:
                camera.orbit(dx, dy, 0.5 / self.resolution[1])
                camera.dolly_scale(scale)
                camera.rotate(wp.sign(camera.pos[1]) * theta, 0.0)
            elif n == 3:
                camera.track(dx, dy)
                camera.dolly_scale(scale)
                camera.rotate(wp.sign(camera.pos[1]) * theta, 0.0)
            elif n == 4:
                dcx = 0.002 * dx * camera.right
                dcy = -0.002 * dy * camera.up
                dcz = 1.5 * (1.0 - scale) * camera.forward
                sphere.translate(dcx + dcy + dcz)
            else:
                qr = wp.quat_from_axis_angle(camera.forward, wp.PI * theta)
                qx = wp.quat_from_axis_angle(camera.up, wp.TAU * dx / self.resolution[1])
                qy = wp.quat_from_axis_angle(camera.right, wp.TAU * dy / self.resolution[1])
                sphere.rotate(qx * qy * qr)
                dr = wp.clamp(sphere.radius * scale, 0.25, 1.25) - sphere.radius
                sphere.resize(dr)


class Keyboard(shaderbang.input.Keyboard):

    def __init__(self):
        super().__init__("keyboard")

    def pre_render(self, **kwargs):
        global state
        if self.pressed(EV_KEY.KEY_C):
            state ^= State.SELF_COLLISION
        if self.pressed(EV_KEY.KEY_P):
            state ^= State.RUN
        if self.pressed(EV_KEY.KEY_R):
            wp.synchronize_stream()
            cloth.reset()
        if self.pressed(EV_KEY.KEY_S):
            match state & STEPS:
                case State.FRAME_STEP:
                    step = State.SMALL_STEP
                case State.SMALL_STEP:
                    step = State.CONTACT_STEP
                case State.CONTACT_STEP | State.SOLVER_STEP:
                    step = State.FRAME_STEP
            state &= ~STEPS
            state |= step
        if self.pressed(EV_KEY.KEY_F):
            state ^= State.CULL_FACE
        if self.pressed(EV_KEY.KEY_W):
            state ^= State.WIREFRAME
        if self.down(any, EV_KEY.KEY_RIGHT, EV_KEY.KEY_SPACE):
            state |= State.STEP
        if self.down(any, EV_KEY.KEY_LEFTCTRL, EV_KEY.KEY_RIGHTCTRL) and self.pressed(EV_KEY.KEY_A):
            for anchor in cloth.anchors:
                anchor.flags |= AnchorFlag.SELECTED
        if self.pressed(any, EV_KEY.KEY_DELETE, EV_KEY.KEY_BACKSPACE):
            for anchor in filter(lambda a: AnchorFlag.SELECTED in a.flags, cloth.anchors):
                anchor.flags &= ~(AnchorFlag.LOCKED | AnchorFlag.SELECTED)

    def post_render(self, **kwargs):
        global state
        if state & (State.RUN | State.STEP) and state & (State.CONTACT_STEP | State.SOLVER_STEP):
            state ^= State.CONTACT_STEP | State.SOLVER_STEP
        state &= ~State.STEP


class Scene(Input):

    def __init__(self):
        super().__init__("scene")

    def init(self, width, height):
        glViewport(0, 0, width, height)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_CULL_FACE)
        glShadeModel(GL_SMOOTH)
        glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 10.0, 10.0, 0.0])

        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)

        glEnable(GL_NORMALIZE)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)

    def render(self, frame, time):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


DistIndex = Callable[[int, int], tuple[int, int] | list[tuple[int, int]]]
BendIndex = Callable[[int, int], tuple[int, int, int, int] | list[tuple[int, int, int, int]]]

T = TypeVar('T', DistIndex, BendIndex)

class Constraint(Generic[T]):

    def __init__(self, *ranges: tuple[range, range, T], ke: float, kd: float, parallel: bool, relaxation = 1.0):
        self.ranges = ranges
        self.ke = ke
        self.kd = kd
        self.parallel = parallel
        self.relaxation = relaxation

        sizes = []
        for xi, yi, index in ranges:
            size = 1
            idx = index(0, 0)
            if isinstance(idx, list):
                size = len(idx)
            sizes.append(len(xi) * len(yi) * size)
        self.sizes = sizes
        self.count = sum(sizes)


class Constraints(Generic[T]):

    class Chain:

        def __init__(self, *constraints):
            self.constraints = constraints

        def __iter__(self):
            for constraints in self.constraints:
                for constraint in constraints:
                    yield constraint

    def __init__(self, *constraints: Constraint[T], dim: int, kernel: Callable):
        self.constraints = constraints
        self.dim = dim
        self.kernel = kernel

        count = 0
        for constraint in constraints:
            count += constraint.count
        self.count = count

        indices = np.zeros((self.count, self.dim), dtype=wp.int32)
        i = 0
        for constraint in constraints:
            for rx, ry, index in constraint.ranges:
                for xi in rx:
                    for yi in ry:
                        idx = index(xi, yi)
                        if isinstance(idx, list):
                            for e in idx:
                                indices[i] = e
                                i += 1
                        else:
                            indices[i] = idx
                            i += 1

        self.indices = wp.array2d(indices, dtype=wp.int32)
        self.rests = wp.zeros((self.count,), dtype=float)
        self.lambdas = wp.zeros((self.count,), dtype=float)

    def __iter__(self):
        i = 0
        for c in self.constraints:
            if c.parallel:
                yield i, c.count, self.kernel, self.indices, self.rests, self.lambdas, c.ke, c.kd, True, c.relaxation
                i += c.count
            else:
                for size in c.sizes:
                    yield i, size, self.kernel, self.indices, self.rests, self.lambdas, c.ke, c.kd, False, c.relaxation
                    i += size


class DistConstraints(Constraints[DistIndex]):

    def __init__(self, *constraints: Constraint[DistIndex]):
        super().__init__(*constraints, dim=2, kernel=Cloth.distance_constraints)


class BendConstraints(Constraints[BendIndex]):

    def __init__(self, *constraints: Constraint[BendIndex]):
        super().__init__(*constraints, dim=4, kernel=Cloth.bending_constraints)


def ray_from_screen(screen_x, screen_y) -> tuple[wp.vec3f, wp.vec3f]:
    viewport = glGetIntegerv(GL_VIEWPORT)
    model_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    proj_matrix = glGetDoublev(GL_PROJECTION_MATRIX)

    screen_y = viewport[3] - screen_y - 1
    p0 = gluUnProject(screen_x, screen_y, 0.0, model_matrix, proj_matrix, viewport)
    p1 = gluUnProject(screen_x, screen_y, 1.0, model_matrix, proj_matrix, viewport)
    origin = wp.vec3(p0[0], p0[1], p0[2])
    direction = wp.vec3(p1[0], p1[1], p1[2]) - origin
    direction = wp.normalize(direction)
    return origin, direction


def ray_to_sphere(origin: wp.vec3, direction: wp.vec3, center: wp.vec3, radius: float) -> Optional[tuple[float, float]]:
    m = origin - center
    b = wp.dot(m, direction)
    c = wp.dot(m, m) - radius * radius
    d = b * b - c
    if d < 0.0:
        return None
    d = wp.sqrt(d)
    return -b - d, -b + d


def ray_to_cylinder(origin: wp.vec3, direction: wp.vec3, center: wp.vec3, radius: float) -> Optional[tuple[float, float]]:
    # Ray vs the INFINITE cylinder along the world-z axis through (center.x,
    # center.y): the sphere quadratic on the xy projection. Rays near-parallel
    # to the axis (a ~ 0) are treated as a miss -- from the app camera they only
    # occur when aiming almost exactly along the rod, where no depth clamp is
    # meaningful.
    mx = origin[0] - center[0]
    my = origin[1] - center[1]
    a = direction[0] * direction[0] + direction[1] * direction[1]
    if a < 1.0e-12:
        return None
    b = mx * direction[0] + my * direction[1]
    c = mx * mx + my * my - radius * radius
    d = b * b - a * c
    if d < 0.0:
        return None
    d = math.sqrt(d)
    return (-b - d) / a, (-b + d) / a


def ray_to_collider(origin: wp.vec3, direction: wp.vec3, center: wp.vec3, radius: float) -> Optional[tuple[float, float]]:
    # Dispatch on the compile-time collider kind (see COLLIDER_KIND).
    if colliderKind == 1:
        return ray_to_cylinder(origin, direction, center, radius)
    return ray_to_sphere(origin, direction, center, radius)


def quat_fraction(q: wp.quat, f: float) -> wp.quat:
    # q^f: the fraction f of the rotation encoded by (normalized) q, i.e. slerp from
    # identity. Used to split a per-frame sphere rotation into per-substep steps so
    # the collider's surface velocity is consistent with the per-substep translation.
    w = max(-1.0, min(1.0, q[3]))
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1.0e-6:
        return wp.quat_identity()
    phi = math.acos(w) * f          # per-substep half-angle
    sn = math.sin(phi) / s
    return wp.quat(q[0] * sn, q[1] * sn, q[2] * sn, math.cos(phi))


def quat_from_unit_vectors(from_vec: wp.vec3, to_vec: wp.vec3) -> wp.quat:
    rot = wp.dot(from_vec, to_vec) + 1.0
    if rot < epsilon:
        rot = 0.0
        if abs(from_vec[0]) > abs(from_vec[2]):
            quat = wp.quat(-from_vec[1], from_vec[0], 0.0, rot)
        else:
            quat = wp.quat(0.0, -from_vec[2], from_vec[1], rot)
    else:
        vec = wp.cross(from_vec, to_vec)
        # quat = wp.quaternion(wp.cross(from_vec, to_vec), rot)
        quat = wp.quat(vec[0], vec[1], vec[2], rot)
    return wp.normalize(quat)


def input_from_device(dev: Device):
    if dev.has(EV_REL) and dev.has(EV_KEY.BTN_LEFT):
        # Mouse
        shaderbang.input.ButtonMouse(dev.name, dev, mouse)
    elif dev.has(EV_KEY) and dev.has(EV_KEY.KEY_A):
        # Keyboard
        shaderbang.input.AsciiKeyboard(dev.name, dev, keyboard)
    elif dev.has(EV_ABS.ABS_MT_SLOT) and dev.has(EV_KEY.BTN_TOUCH) and dev.has_property(INPUT_PROP_DIRECT):
        # Touchscreen
        # Only consider direct input devices, like touchscreens and drawing tablets, see:
        # https://www.kernel.org/doc/Documentation/input/event-codes.txt
        shaderbang.input.Touchscreen(dev.name, dev, touchscreen)
    elif dev.has(EV_ABS.ABS_MT_SLOT) and dev.has(EV_KEY.BTN_TOUCH) and dev.has_property(INPUT_PROP_POINTER):
        # Trackpad
        # https://www.kernel.org/doc/Documentation/input/multi-touch-protocol.txt
        shaderbang.input.Trackpad(dev.name, dev, trackpad, mouse)
    else:
        dev.fd.close()


def hot_plug_devices(devices: ExitStack, inotify: INotify):
    with devices:
        while True:
            for ev in inotify.read():
                p = os.path.join("/dev/input", ev.name)
                if (str.startswith(ev.name, "event")
                        and os.path.exists(p)
                        and os.access(p, os.R_OK)
                        and stat.S_ISCHR(os.stat(p)[stat.ST_MODE])):
                    input_from_device(Device(devices.enter_context(open(p, "rb"))))


if __name__ == "__main__":
    args = parser.parse_args()

    scene = Scene()
    camera = Camera()
    ground = Ground()
    sphere = Sphere(center=wp.vec3(0.0, 1.5, 0.0), radius=0.5)
    cloth = Cloth(y_offset=2.2, num_x=400, num_y=400, spacing=0.015)

    keyboard = Keyboard()
    mouse = Mouse()
    touchscreen = Touchscreen()
    trackpad = Trackpad()

    devices = ExitStack()
    with devices:
        for path in list(filter(lambda p: os.path.exists(p) and stat.S_ISCHR(os.stat(p)[stat.ST_MODE]),
                                glob.glob("{}/event*".format("/dev/input")))):
            input_from_device(Device(devices.enter_context(open(path, "rb"))))
        devices = devices.pop_all()

    inotify = INotify()
    inotify.add_watch("/dev/input", IN_CREATE | IN_ATTRIB)
    Thread(target=hot_plug_devices, args=[devices, inotify], daemon=True).start()

    ret = sb.init(ctypes.byref(options(args)))
    if ret != 0:
        devices.close()
        exit(ret)

    ret = sb.run()
    if ret != 0:
        devices.close()
        exit(ret)

    stopped = threading.Event()
    pthread_sigmask(signal.SIG_BLOCK, [signal.SIGCONT])

    def join():
        sb.join()
        stopped.set()
        pthread_kill(main_thread().ident, signal.SIGCONT)

    Thread(target=join, daemon=True).start()

    if sigwait({signal.SIGINT, signal.SIGCONT}) == signal.SIGINT:
        sb.stop()
        ret = stopped.wait(timeout=5.0)

    inotify.close()
    devices.close()
