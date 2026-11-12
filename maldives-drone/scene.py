"""Maldives one-take drone shot, built and rendered procedurally with Blender (bpy).

Usage:
    python scene.py <out_dir> [start] [end] [step] [samples] [res_percent]

Renders PNG frames (existing files are skipped, so a run can be resumed).
Assemble with:
    ffmpeg -framerate 24 -i <out_dir>/%04d.png -c:v libx264 -pix_fmt yuv420p -crf 17 out.mp4
"""
import math
import os
import random
import sys

import bpy  # noqa: E402 (must precede addon_utils)
import bmesh
import addon_utils
import numpy as np
from mathutils import Euler, Matrix, Vector

FPS = 24
DURATION = 12.5
N_FRAMES = int(FPS * DURATION)
DECK_Z = 1.8

random.seed(7)
rng = np.random.default_rng(7)


# ----------------------------------------------------------------- helpers
def new_obj(name, bm, mats=(), smooth=False):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for m in mats:
        mesh.materials.append(m)
    if smooth:
        mesh.shade_smooth()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def instance(src, matrix):
    obj = bpy.data.objects.new(src.name + "_i", src.data)
    obj.matrix_world = matrix
    bpy.context.scene.collection.objects.link(obj)
    return obj


def trs(loc, rot_z=0.0, scale=(1, 1, 1), rot=None):
    e = rot if rot is not None else Euler((0, 0, rot_z))
    return Matrix.LocRotScale(Vector(loc), e, Vector(scale))


def material(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return m, nt, out


def simple_mat(name, color, rough=0.8, bump_scale=None, bump_strength=0.3,
               emission=None):
    m, nt, out = material(name)
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    if emission:
        b.inputs["Emission Color"].default_value = (*emission[0], 1)
        b.inputs["Emission Strength"].default_value = emission[1]
    if bump_scale:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = bump_scale
        nz.inputs["Detail"].default_value = 4
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = bump_strength
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        nt.links.new(nz.outputs["Fac"], bp.inputs["Height"])
        nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])
        mixc = nt.nodes.new("ShaderNodeMixRGB")
        mixc.blend_type = "MULTIPLY"
        mixc.inputs["Fac"].default_value = 0.5
        mixc.inputs["Color1"].default_value = (*color, 1)
        nt.links.new(nz.outputs["Fac"], mixc.inputs["Color2"])
        nt.links.new(mixc.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    return m


# ------------------------------------------------------------------ layout
def jetty(u):
    """Curved jetty centre line, u in [0, 1] from the beach out into the lagoon."""
    return np.array([58.0 + 142.0 * u, 28.0 * math.sin(math.pi * u)])


def jetty_tangent(u):
    d = jetty(u + 1e-3) - jetty(u - 1e-3)
    return d / np.linalg.norm(d)


def terrain_height(x, y):
    d = np.sqrt((x / 1.3) ** 2 + y ** 2)
    prof_d = [0, 30, 42, 50, 58, 80, 135, 152, 166, 185, 270, 2000]
    prof_h = [3.0, 2.6, 1.3, 0.05, -0.7, -1.6, -1.9, -0.6, -0.5, -4.0, -25, -25]
    h = np.interp(d, prof_d, prof_h)
    n = (np.sin(x * 0.11 + 1.3) * np.cos(y * 0.13) * 0.5
         + np.sin(x * 0.037 - y * 0.051) * 0.6
         + np.sin(x * 0.31 + y * 0.27) * 0.12)
    under = np.clip(-h / 1.5, 0, 1) * np.clip((270 - d) / 60, 0, 1)
    h = h + n * (0.25 + 0.45 * under) * np.clip((270 - d) / 60, 0, 1)
    # sandbank in the lagoon
    h += 2.6 * np.exp(-(((x + 30) / 1.6) ** 2 + (y - 105) ** 2) / (2 * 13.0 ** 2))
    # keep the beach strictly above water near the shore line
    return h


def build_terrain(mat):
    n, size = 321, 960.0
    xs = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Z = terrain_height(X, Y)
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    idx = np.arange(n * n).reshape(n, n)
    faces = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1)
    mesh = bpy.data.meshes.new("terrain")
    mesh.from_pydata(verts.tolist(), [], faces.reshape(-1, 4).tolist())
    mesh.materials.append(mat)
    mesh.shade_smooth()
    obj = bpy.data.objects.new("terrain", mesh)
    bpy.context.scene.collection.objects.link(obj)


def seabed_material():
    m, nt, out = material("seabed")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -25
    mr.inputs["From Max"].default_value = 3
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp

    def pos(z):
        return (z + 25) / 28

    stops = [
        (-25, (0.004, 0.045, 0.13)),
        (-9, (0.008, 0.16, 0.34)),
        (-3.5, (0.02, 0.42, 0.55)),
        (-1.6, (0.07, 0.66, 0.66)),
        (-0.5, (0.36, 0.86, 0.76)),
        (0.0, (0.80, 0.86, 0.72)),
        (0.25, (0.93, 0.86, 0.70)),
        (1.5, (0.90, 0.82, 0.64)),
        (2.2, (0.16, 0.26, 0.07)),
    ]
    cr.elements[0].position = pos(stops[0][0])
    cr.elements[0].color = (*stops[0][1], 1)
    cr.elements[1].position = pos(stops[1][0])
    cr.elements[1].color = (*stops[1][1], 1)
    for z, c in stops[2:]:
        e = cr.elements.new(pos(z))
        e.color = (*c, 1)
    # reef / sand variation
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.06
    nz.inputs["Detail"].default_value = 6
    nz.inputs["Roughness"].default_value = 0.65
    var = nt.nodes.new("ShaderNodeMapRange")
    var.inputs["From Min"].default_value = 0.35
    var.inputs["From Max"].default_value = 0.7
    var.inputs["To Min"].default_value = 1.0
    var.inputs["To Max"].default_value = 0.45
    mul = nt.nodes.new("ShaderNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs["Fac"].default_value = 1.0
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 1.5
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.15
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Roughness"].default_value = 0.9
    L = nt.links.new
    L(geo.outputs["Position"], sep.inputs[0])
    L(sep.outputs["Z"], mr.inputs["Value"])
    L(mr.outputs["Result"], ramp.inputs["Fac"])
    L(tc.outputs["Object"], nz.inputs["Vector"])
    L(tc.outputs["Object"], fine.inputs["Vector"])
    L(nz.outputs["Fac"], var.inputs["Value"])
    L(ramp.outputs["Color"], mul.inputs["Color1"])
    L(var.outputs["Result"], mul.inputs["Color2"])
    L(mul.outputs["Color"], b.inputs["Base Color"])
    # underwater parts glow slightly: cheap stand-in for light scattered in the water
    lt = nt.nodes.new("ShaderNodeMath")
    lt.operation = "LESS_THAN"
    lt.inputs[1].default_value = -0.02
    gs = nt.nodes.new("ShaderNodeMath")
    gs.operation = "MULTIPLY"
    gs.inputs[1].default_value = 1.7
    L(sep.outputs["Z"], lt.inputs[0])
    L(lt.outputs[0], gs.inputs[0])
    L(mul.outputs["Color"], b.inputs["Emission Color"])
    L(gs.outputs[0], b.inputs["Emission Strength"])
    L(fine.outputs["Fac"], bp.inputs["Height"])
    L(bp.outputs["Normal"], b.inputs["Normal"])
    L(b.outputs["BSDF"], out.inputs["Surface"])
    return m


def water_material():
    m, nt, out = material("water")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.noise_dimensions = "4D"
    n1.inputs["Scale"].default_value = 0.22
    n1.inputs["Detail"].default_value = 5
    n1.inputs["Roughness"].default_value = 0.6
    n2 = nt.nodes.new("ShaderNodeTexNoise")
    n2.noise_dimensions = "4D"
    n2.inputs["Scale"].default_value = 1.3
    n2.inputs["Detail"].default_value = 3
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "MULTIPLY_ADD"
    add.inputs[1].default_value = 0.25
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.35
    bp.inputs["Distance"].default_value = 1.0
    fr = nt.nodes.new("ShaderNodeFresnel")
    fr.inputs["IOR"].default_value = 1.33
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.inputs["Color"].default_value = (0.86, 0.97, 0.95, 1)
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl.inputs["Roughness"].default_value = 0.035
    mix = nt.nodes.new("ShaderNodeMixShader")
    L = nt.links.new
    L(tc.outputs["Object"], n1.inputs["Vector"])
    L(tc.outputs["Object"], n2.inputs["Vector"])
    L(n2.outputs["Fac"], add.inputs[0])
    L(n1.outputs["Fac"], add.inputs[2])
    L(add.outputs[0], bp.inputs["Height"])
    L(bp.outputs["Normal"], fr.inputs["Normal"])
    L(bp.outputs["Normal"], gl.inputs["Normal"])
    L(fr.outputs["Fac"], mix.inputs["Fac"])
    L(tr.outputs["BSDF"], mix.inputs[1])
    L(gl.outputs["BSDF"], mix.inputs[2])
    L(mix.outputs["Shader"], out.inputs["Surface"])
    return m, (n1, n2)


def build_water():
    mat, noises = water_material()
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=30000)
    new_obj("water", bm, [mat])
    deep = simple_mat("deep", (0.004, 0.045, 0.13), 1.0, emission=((0.004, 0.045, 0.13), 1.7))
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=30000)
    o = new_obj("deep", bm, [deep])
    o.location.z = -25.5
    return noises


# ------------------------------------------------------------------ palms
def build_palm(trunk_mat, leaf_mat, height, bend, seed):
    r = random.Random(seed)
    bm = bmesh.new()
    rings, sides = 9, 6
    prev = None
    top = None
    for i in range(rings):
        s = i / (rings - 1)
        c = Vector((bend * s * s, 0, height * s))
        rad = 0.30 - 0.14 * s
        ring = [bm.verts.new(c + Vector((math.cos(a) * rad, math.sin(a) * rad, 0)))
                for a in [2 * math.pi * k / sides for k in range(sides)]]
        if prev:
            for k in range(sides):
                f = bm.faces.new((prev[k], prev[(k + 1) % sides],
                                  ring[(k + 1) % sides], ring[k]))
                f.material_index = 0
        prev = ring
        top = c
    n_fronds = 13
    for k in range(n_fronds):
        a = 2 * math.pi * k / n_fronds + r.uniform(-0.2, 0.2)
        length = r.uniform(3.6, 4.8)
        rise = r.uniform(0.3, 1.3)
        droop = r.uniform(0.9, 1.5)
        dirv = Vector((math.cos(a), math.sin(a), 0))
        side = Vector((-math.sin(a), math.cos(a), 0))
        segs = 7
        prev = None
        for i in range(segs + 1):
            s = i / segs
            p = top + dirv * (length * s) + Vector(
                (0, 0, rise * math.sin(s * math.pi * 0.6) * 1.6 - droop * s * s * length * 0.45))
            w = 0.75 * math.sin(math.pi * (s ** 0.7)) + 0.04
            row = [bm.verts.new(p - side * w - Vector((0, 0, 0.22 * w))),
                   bm.verts.new(p),
                   bm.verts.new(p + side * w - Vector((0, 0, 0.22 * w)))]
            if prev:
                for j in range(2):
                    f = bm.faces.new((prev[j], prev[j + 1], row[j + 1], row[j]))
                    f.material_index = 1
            prev = row
    return new_obj("palm", bm, [trunk_mat, leaf_mat], smooth=True)


def scatter_vegetation():
    trunk = simple_mat("trunk", (0.30, 0.22, 0.14), 0.9, bump_scale=6, bump_strength=0.5)
    leaf = simple_mat("leaf", (0.07, 0.24, 0.035), 0.55)
    bush_mat = simple_mat("bush", (0.05, 0.17, 0.03), 0.7, bump_scale=3, bump_strength=0.8)
    palms = [build_palm(trunk, leaf, h, b, i)
             for i, (h, b) in enumerate([(7.5, 1.8), (9.0, 2.6), (6.2, 1.0), (8.3, 3.2)])]
    for p in palms:
        p.location = (0, 0, -500)  # templates hidden far below
        p.hide_render = True
    placed = 0
    tries = 0
    while placed < 150 and tries < 5000:
        tries += 1
        x, y = random.uniform(-62, 62), random.uniform(-48, 48)
        d = math.hypot(x / 1.3, y)
        if d > 46:
            continue
        z = float(terrain_height(np.array(x), np.array(y)))
        if z < 0.7:
            continue
        # keep a flight corridor from the jetty over the island
        if abs(y) < 10 and x > -5:
            continue
        s = random.uniform(0.85, 1.25)
        rot = Euler((random.uniform(-0.08, 0.08), random.uniform(-0.08, 0.08),
                     random.uniform(0, 6.28)))
        instance(random.choice(palms), trs((x, y, z - 0.2), rot=rot, scale=(s, s, s)))
        placed += 1
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    bush = new_obj("bush", bm, [bush_mat], smooth=True)
    bush.location = (0, 0, -500)
    bush.hide_render = True
    for _ in range(170):
        x, y = random.uniform(-52, 52), random.uniform(-40, 40)
        d = math.hypot(x / 1.3, y)
        if d > 39 or (abs(y) < 5 and x > 30):
            continue
        z = float(terrain_height(np.array(x), np.array(y)))
        s = random.uniform(1.2, 2.8)
        instance(bush, trs((x, y, z + s * 0.25), random.uniform(0, 6.28),
                           (s, s * random.uniform(0.8, 1.2), s * random.uniform(0.45, 0.7))))


# ------------------------------------------------------------------ resort
def cube_mesh(name, mat):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    o = new_obj(name, bm, [mat])
    o.location = (0, 0, -500)
    o.hide_render = True
    return o


def build_resort():
    wood = simple_mat("wood", (0.24, 0.14, 0.07), 0.7, bump_scale=9, bump_strength=0.35)
    wall = simple_mat("wall", (0.80, 0.68, 0.50), 0.8)
    thatch = simple_mat("thatch", (0.50, 0.33, 0.13), 0.95, bump_scale=14, bump_strength=0.9)
    glass = simple_mat("glass", (0.03, 0.05, 0.06), 0.08,
                       emission=((1.0, 0.62, 0.30), 0.12))
    plank = cube_mesh("plank", wood)
    wallbox = cube_mesh("wallbox", wall)
    window = cube_mesh("window", glass)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=1.0, radius2=0.0, depth=1.0)
    roof = new_obj("roof", bm, [thatch])
    roof.location = (0, 0, -500)
    roof.hide_render = True
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.17, radius2=0.17, depth=1.0)
    post = new_obj("post", bm, [wood], smooth=True)
    post.location = (0, 0, -500)
    post.hide_render = True

    def villa(cx, cy, ang):
        base = trs((cx, cy, 0), ang)

        def put(src, loc, scale, rz=0.0):
            instance(src, base @ trs(loc, rz, scale))

        put(plank, (0, 0, DECK_Z - 0.13), (11, 11, 0.26))
        put(plank, (6.6, 0, DECK_Z - 0.6), (2.4, 5, 0.2))  # lower sun deck
        put(wallbox, (0, 0, DECK_Z + 1.45), (7.6, 7.6, 2.9))
        put(roof, (0, 0, DECK_Z + 2.75 + 2.3), (7.6, 7.6, 4.6), math.pi / 4)
        for k in range(4):
            a = k * math.pi / 2
            put(window, (3.82 * math.cos(a), 3.82 * math.sin(a), DECK_Z + 1.35),
                (0.08, 4.2, 1.9), a)
        for sx in (-4.6, 4.6):
            for sy in (-4.6, 4.6):
                put(post, (sx, sy, DECK_Z / 2 - 1.6), (1, 1, DECK_Z + 3.2))

    # main walkway
    n_seg = 56
    us = np.linspace(-0.03, 1.0, n_seg + 1)
    for i in range(n_seg):
        a, b = jetty(us[i]), jetty(us[i + 1])
        mid = (a + b) / 2
        d = b - a
        ang = math.atan2(d[1], d[0])
        ln = float(np.linalg.norm(d)) + 0.08
        instance(plank, trs((mid[0], mid[1], DECK_Z - 0.13), ang, (ln, 2.8, 0.26)))
        if i % 2 == 0:
            nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            for s in (-1.2, 1.2):
                p = mid + nrm * s
                instance(post, trs((p[0], p[1], DECK_Z / 2 - 1.6), 0, (1, 1, DECK_Z + 3.2)))
    # end platform
    e = jetty(1.0)
    instance(plank, trs((e[0] + 3.5, e[1], DECK_Z - 0.13),
                        math.atan2(*jetty_tangent(1.0)[::-1]), (8, 8, 0.26)))
    # villas alternate left / right of the walkway
    for k, u in enumerate(np.linspace(0.2, 0.95, 9)):
        c = jetty(u)
        t = jetty_tangent(u)
        nrm = np.array([-t[1], t[0]])
        side = 1 if k % 2 == 0 else -1
        ang = math.atan2(nrm[1] * side, nrm[0] * side)
        vc = c + nrm * side * 15.0
        villa(vc[0], vc[1], ang)
        cc = c + nrm * side * 5.4
        instance(plank, trs((cc[0], cc[1], DECK_Z - 0.13), ang, (8.4, 1.9, 0.26)))


# ------------------------------------------------------------------- world
def build_world():
    w = bpy.data.worlds.new("sky")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    sky = nt.nodes.new("ShaderNodeTexSky")
    avail = [e.identifier for e in sky.bl_rna.properties["sky_type"].enum_items]
    for cand in ("MULTIPLE_SCATTERING", "NISHITA", "SINGLE_SCATTERING"):
        if cand in avail:
            sky.sky_type = cand
            break
    for attr, val in (("sun_elevation", math.radians(21)),
                      ("sun_rotation", math.radians(-28)),
                      ("sun_size", math.radians(1.2)),
                      ("sun_intensity", 0.6),
                      ("air_density", 1.0),
                      ("dust_density", 0.9),
                      ("aerosol_density", 0.9),
                      ("ozone_density", 1.5)):
        if hasattr(sky, attr):
            setattr(sky, attr, val)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.35
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs[0], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    print("sky type:", sky.sky_type, "of", avail)


# ------------------------------------------------------------------ camera
def camera_path():
    def j3(u, z):
        p = jetty(u)
        return [p[0], p[1], z]

    keys = [  # time, position, look-at target
        (0.0, [440, -75, 96], [175, 12, 0]),
        (2.4, [318, -34, 40], [160, 14, 2]),
        (4.2, j3(1.12, 8.5), j3(0.78, 2.5)),
        (5.0, j3(1.0, 4.6), j3(0.72, 3.0)),
    ]
    for k in range(1, 9):
        u = 1.0 - k / 8 * 0.92
        t = 5.0 + k / 8 * 4.2
        ut = u - 0.27
        tgt = j3(ut, 3.2) if ut > 0 else [58 + ut * 260, 0, 3.2 - ut * 30]
        keys.append((t, j3(u, 4.4), tgt))
    keys += [
        (9.9, [42, 1, 10.5], [-60, 0, 18]),
        (10.9, [-8, 2, 28], [-230, 0, 28]),
        (12.5, [-112, 6, 64], [-640, -20, 6]),
    ]
    T = np.array([k[0] for k in keys])
    P = np.array([k[1] for k in keys], float)
    G = np.array([k[2] for k in keys], float)

    def hermite(V, ts):
        m = np.zeros_like(V)
        m[1:-1] = (V[2:] - V[:-2]) / (T[2:] - T[:-2])[:, None]
        m[0] = (V[1] - V[0]) / (T[1] - T[0])
        m[-1] = (V[-1] - V[-2]) / (T[-1] - T[-2])
        out = np.zeros((len(ts), V.shape[1]))
        for n, t in enumerate(ts):
            i = int(np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(T) - 2))
            h = T[i + 1] - T[i]
            s = (t - T[i]) / h
            out[n] = ((2 * s ** 3 - 3 * s ** 2 + 1) * V[i] + (s ** 3 - 2 * s ** 2 + s) * h * m[i]
                      + (-2 * s ** 3 + 3 * s ** 2) * V[i + 1] + (s ** 3 - s ** 2) * h * m[i + 1])
        return out

    def smooth(A, win):
        pad = win // 2
        Ap = np.concatenate([np.repeat(A[:1], pad, 0), A, np.repeat(A[-1:], pad, 0)])
        ker = np.hanning(win + 2)[1:-1]
        ker /= ker.sum()
        return np.stack([np.convolve(Ap[:, c], ker, mode="valid") for c in range(A.shape[1])], 1)

    ts = np.arange(N_FRAMES) / FPS
    return smooth(hermite(P, ts), 15), smooth(hermite(G, ts), 21)


# -------------------------------------------------------------------- main
def setup_render(samples, res_percent):
    sc = bpy.context.scene
    addon_utils.enable("cycles")
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.04
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 5
    sc.cycles.diffuse_bounces = 2
    sc.cycles.glossy_bounces = 2
    sc.cycles.transmission_bounces = 2
    sc.cycles.transparent_max_bounces = 6
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.use_persistent_data = True
    sc.render.threads_mode = "AUTO"
    sc.render.resolution_x = 1280
    sc.render.resolution_y = 720
    sc.render.resolution_percentage = res_percent
    sc.render.fps = FPS
    sc.render.image_settings.file_format = "PNG"
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - High Contrast"
    except Exception as e:  # noqa: BLE001
        print("view transform fallback:", e)
    sc.view_settings.exposure = -0.75


def main():
    args = sys.argv[1:]
    out_dir = args[0]
    start = int(args[1]) if len(args) > 1 else 0
    end = int(args[2]) if len(args) > 2 else N_FRAMES - 1
    step = int(args[3]) if len(args) > 3 else 1
    samples = int(args[4]) if len(args) > 4 else 32
    res = int(args[5]) if len(args) > 5 else 100
    os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_render(samples, res)
    build_world()
    build_terrain(seabed_material())
    noises = build_water()
    scatter_vegetation()
    build_resort()

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 17
    cam_data.clip_start = 0.3
    cam_data.clip_end = 60000
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    pos, tgt = camera_path()
    for f in range(start, end + 1, step):
        path = os.path.join(out_dir, f"{f:04d}.png")
        if os.path.exists(path):
            continue
        p, g = Vector(pos[f]), Vector(tgt[f])
        cam.location = p
        cam.rotation_euler = (g - p).to_track_quat("-Z", "Y").to_euler()
        for n in noises:
            n.inputs["W"].default_value = f / FPS * 0.55
        bpy.context.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print("FRAME_DONE", f, flush=True)


if __name__ == "__main__":
    main()
