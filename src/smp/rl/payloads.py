"""Sim-to-real hardware payloads for the G1: Jetson on back + Dex1-1 grippers.

Vendored/adapted from wbc_mjlab so the SMP getup teacher trains on the *same*
mass/inertia + collision geometry the wbc_mjlab student deploys with. Kept
self-contained (no wbc_mjlab import) so smp keeps its own pinned mjlab rev.

The Jetson is a single box on the torso back. The Dex1-1 grippers are welded
on as the **full mesh assembly** (chassis + rail + two slider jaws + UMI finger
pads) — the same bodies as ``deploy/assets/g1/g1_sim2sim_29dof_gripper.xml`` but
with the prismatic jaw joints dropped, so the gripper is a single rigid fixed
payload that contributes accurate collision geometry (e.g. hands hitting the
ground while getting up). Total per-side gripper mass ≈ 0.55 kg, matching the
old box approximation; this swap is a geometry upgrade at the same mass.

Payloads change dynamics/collision only, not the joint/ee kinematics the frozen
SMP prior + guidance reward score, so the prior stays valid.

On by default; set ``SMP_ATTACH_PAYLOADS=0`` to disable (reproduce the original
payload-free getup task).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco

_MESH_DIR = os.path.join(os.path.dirname(__file__), "assets", "gripper_meshes")

# Rubber-hand mass baked into MJCF wrist_yaw_link.mass (0.254576 kg total).
# The URDF splits this into wrist_yaw_link (0.084576 kg) + rubber_hand (0.170);
# we subtract exactly 0.170 from the MJCF wrist to match.
RUBBER_HAND_MASS_ESTIMATE: float = 0.170


# =============================================================================
# Jetson box payload (unchanged from the box approach).


@dataclass(frozen=True)
class _NewBodyCfg:
  """A rigid child box welded to a parent link (no joint)."""

  name: str
  parent_body: str
  pos: tuple[float, float, float]
  half_size: tuple[float, float, float]  # box geom half-sizes (x, y, z)
  mass: float
  diaginertia: tuple[float, float, float]
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  rgba: tuple[float, float, float, float] = (0.15, 0.15, 0.15, 1.0)


# NVIDIA Jetson + mount + cameras on the robot's back (−X face of torso_link).
JETSON_PAYLOAD = _NewBodyCfg(
  name="jetson_payload",
  parent_body="torso_link",
  pos=(-0.09, 0.0, 0.15),
  half_size=(0.02844, 0.05620, 0.12160),
  mass=2.0,
  diaginertia=(0.011962, 0.010396, 0.002645),
  rgba=(0.0, 0.0, 0.0, 1.0),
)


# =============================================================================
# Dex1-1 gripper assembly (welded mesh tree, joints dropped).

# Mesh assets to register once: (asset_name, filename, scale). Mirrors the
# <asset> block in g1_sim2sim_29dof_gripper.xml.
_GRIPPER_MESHES: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
  ("dex1_base_link", "dex1_base_link.STL", (1.0, 1.0, 1.0)),
  ("dex1_slider", "dex1_slider.STL", (0.5, 1.0, 1.0)),
  ("umi_finger_L", "umi_finger_L.stl", (0.65, 0.65, 0.65)),
  ("umi_finger_R", "umi_finger_R.stl", (0.65, 0.65, 0.65)),
)
_GRIPPER_RGBA = (0.05, 0.05, 0.05, 1.0)


@dataclass(frozen=True)
class _GeomCfg:
  name: str
  pos: tuple[float, float, float]
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  mesh: str | None = None          # mesh asset name, or None for a primitive
  prim_type: int | None = None     # mujoco.mjtGeom.* for primitives
  size: tuple[float, ...] = ()
  collide: bool = True


@dataclass(frozen=True)
class _AsmBodyCfg:
  name: str
  pos: tuple[float, float, float]
  quat: tuple[float, float, float, float]
  mass: float
  ipos: tuple[float, float, float]
  diaginertia: tuple[float, float, float]
  geoms: tuple[_GeomCfg, ...] = ()
  children: tuple["_AsmBodyCfg", ...] = ()


def _gripper_tree(side: str) -> _AsmBodyCfg:
  """Welded Dex1-1 assembly for ``side`` in {"L", "R"}. Joints dropped; masses
  + frames copied verbatim from g1_sim2sim_29dof_gripper.xml."""
  rail_y = 0.003 if side == "L" else -0.003
  jaw1_pad_z = 0.007 if side == "L" else -0.007
  return _AsmBodyCfg(
    name=f"{'left' if side == 'L' else 'right'}_gripper_mount",
    pos=(0.0415, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    mass=0.316,
    ipos=(0.0593, 0.0, 0.0081),
    diaginertia=(6.8944e-05, 4.9533e-05, 3.4443e-05),
    geoms=(
      _GeomCfg(
        name=f"{'left' if side == 'L' else 'right'}_gripper_mount_geom",
        mesh="dex1_base_link",
        pos=(0.03, 0.0, 0.0),
        quat=(0.7071068, 0.0, 0.0, -0.7071068),
      ),
      # Spacer cylinder = the connector between wrist and chassis. The source
      # manipulation XML left it visual-only (contype/conaffinity 0) since it
      # never grasps; for getup the whole gripper bears weight on the ground,
      # so it must collide too.
      _GeomCfg(
        name=f"{'left' if side == 'L' else 'right'}_wrist_gripper_spacer",
        prim_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.025, 0.015),
        pos=(0.015, 0.0, 0.0),
        quat=(0.7071068, 0.0, 0.7071068, 0.0),
      ),
    ),
    children=(
      _AsmBodyCfg(
        name=f"gripper_rail_{side}",
        pos=(0.100, rail_y, 0.0),
        quat=(0.707107, 0.0, 0.707107, 0.0),
        mass=1e-4,
        ipos=(0.0, 0.0, 0.0),
        diaginertia=(1e-7, 1e-7, 1e-7),
        children=(
          _AsmBodyCfg(
            name=f"gripper_1_{side}",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            mass=0.117,
            ipos=(0.0, -0.020, 0.06),
            diaginertia=(1.5e-04, 1.4e-04, 6.0e-05),
            geoms=(
              _GeomCfg(
                name=f"gripper_1_{side}_slider_geom",
                mesh="dex1_slider",
                pos=(-0.012, 0.0, 0.0),
                quat=(0.5, 0.5, -0.5, -0.5),
              ),
              _GeomCfg(
                name=f"gripper_1_{side}_pad_geom",
                mesh="umi_finger_L",
                pos=(-0.008, 0.0, jaw1_pad_z),
                quat=(0.0, 0.7071068, -0.7071068, 0.0),
              ),
            ),
          ),
          _AsmBodyCfg(
            name=f"gripper_2_{side}",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            mass=0.117,
            ipos=(0.0, 0.020, 0.06),
            diaginertia=(1.5e-04, 1.4e-04, 6.0e-05),
            geoms=(
              _GeomCfg(
                name=f"gripper_2_{side}_slider_geom",
                mesh="dex1_slider",
                pos=(0.012, 0.0, 0.0),
                quat=(0.5, 0.5, 0.5, 0.5),
              ),
              _GeomCfg(
                name=f"gripper_2_{side}_pad_geom",
                mesh="umi_finger_R",
                pos=(0.008, 0.0, 0.007),
                quat=(0.0, 0.7071068, 0.7071068, 0.0),
              ),
            ),
          ),
        ),
      ),
    ),
  )


# (wrist_body, rubber_hand_mesh_name, hand_collision_geom_name, side)
_HAND_SWAPS: tuple[tuple[str, str, str, str], ...] = (
  ("left_wrist_yaw_link", "left_rubber_hand", "left_hand_collision", "L"),
  ("right_wrist_yaw_link", "right_rubber_hand", "right_hand_collision", "R"),
)


def _attach_enabled() -> bool:
  return os.environ.get("SMP_ATTACH_PAYLOADS", "1").lower() in (
    "1", "true", "yes", "on",
  )


def _add_box_body(spec: mujoco.MjSpec, p: _NewBodyCfg) -> None:
  parent = spec.body(p.parent_body)
  body = parent.add_body(name=p.name, pos=list(p.pos), quat=list(p.quat))
  body.mass = p.mass
  body.inertia = list(p.diaginertia)
  body.ipos = [0.0, 0.0, 0.0]
  body.iquat = [1.0, 0.0, 0.0, 0.0]
  visual_cls = spec.find_default("visual")
  collision_cls = spec.find_default("collision")
  mat_name = f"{p.name}_mat"
  spec.add_material(
    name=mat_name, rgba=list(p.rgba), specular=0.0, reflectance=0.0,
    shininess=0.0,
  )
  body.add_geom(
    default=visual_cls, name=f"{p.name}_visual",
    type=mujoco.mjtGeom.mjGEOM_BOX, size=list(p.half_size), rgba=list(p.rgba),
    material=mat_name,
  )
  body.add_geom(
    default=collision_cls, name=f"{p.name}_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX, size=list(p.half_size),
  )


def _delete_geoms(spec: mujoco.MjSpec, body_name: str, *, names=(), meshes=()) -> None:
  body = spec.body(body_name)
  victims = [g for g in body.geoms if (g.name in names) or (g.meshname in meshes)]
  for g in victims:
    spec.delete(g)


def _ensure_gripper_meshes(spec: mujoco.MjSpec) -> None:
  existing = {m.name for m in spec.meshes}
  for name, filename, scale in _GRIPPER_MESHES:
    if name in existing:
      continue
    spec.add_mesh(
      name=name, file=os.path.join(_MESH_DIR, filename), scale=list(scale)
    )


def _add_geom(body, g: _GeomCfg, visual_cls, collision_cls) -> None:
  if g.mesh is not None:
    # Visual (rendered) + collision (mjlab's collision default → floor / self
    # collision filtering). Convex hull is used for mesh collision.
    body.add_geom(
      default=visual_cls, name=f"{g.name}_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH, meshname=g.mesh,
      pos=list(g.pos), quat=list(g.quat), rgba=list(_GRIPPER_RGBA),
    )
    if g.collide:
      body.add_geom(
        default=collision_cls, name=g.name,
        type=mujoco.mjtGeom.mjGEOM_MESH, meshname=g.mesh,
        pos=list(g.pos), quat=list(g.quat),
      )
  else:
    # Primitive (e.g. spacer cylinder): visual + collision, like the meshes.
    body.add_geom(
      default=visual_cls, name=f"{g.name}_visual", type=g.prim_type,
      size=list(g.size), pos=list(g.pos), quat=list(g.quat),
      rgba=list(_GRIPPER_RGBA),
    )
    if g.collide:
      body.add_geom(
        default=collision_cls, name=g.name, type=g.prim_type,
        size=list(g.size), pos=list(g.pos), quat=list(g.quat),
      )


def _build_assembly(parent_body, cfg: _AsmBodyCfg, visual_cls, collision_cls) -> None:
  body = parent_body.add_body(name=cfg.name, pos=list(cfg.pos), quat=list(cfg.quat))
  body.mass = cfg.mass
  body.ipos = list(cfg.ipos)
  body.inertia = list(cfg.diaginertia)
  body.iquat = [1.0, 0.0, 0.0, 0.0]
  for g in cfg.geoms:
    _add_geom(body, g, visual_cls, collision_cls)
  for child in cfg.children:
    _build_assembly(body, child, visual_cls, collision_cls)


def apply_payloads_to_spec(
  spec: mujoco.MjSpec,
  *,
  include_jetson: bool = True,
  swap_hands: bool = True,
) -> mujoco.MjSpec:
  """Attach Jetson box + welded Dex1-1 gripper assemblies. No-op when disabled.

  Idempotency is the caller's responsibility (use ``wrap_spec_fn_with_payloads``).
  """
  if not _attach_enabled():
    return spec
  if include_jetson:
    _add_box_body(spec, JETSON_PAYLOAD)
  if swap_hands:
    _ensure_gripper_meshes(spec)
    visual_cls = spec.find_default("visual")
    collision_cls = spec.find_default("collision")
    for wrist_name, rubber_mesh, hand_col, side in _HAND_SWAPS:
      # Drop the stock rubber hand (geom + mesh) and its baked-in wrist mass.
      _delete_geoms(spec, wrist_name, names=(hand_col,), meshes=(rubber_mesh,))
      wrist = spec.body(wrist_name)
      wrist.mass = wrist.mass - RUBBER_HAND_MASS_ESTIMATE
      # Weld the gripper tree under the wrist.
      _build_assembly(wrist, _gripper_tree(side), visual_cls, collision_cls)
      # The two jaws are siblings whose pad/slider geoms cross the centerline;
      # exclude their mutual contact (matches the source XML <exclude>).
      spec.add_exclude(bodyname1=f"gripper_1_{side}", bodyname2=f"gripper_2_{side}")
  return spec


_PAYLOAD_WRAPPED_ATTR = "__smp_payloads_wrapped__"


def wrap_spec_fn_with_payloads(
  base_spec_fn: "callable[[], mujoco.MjSpec]",
  *,
  include_jetson: bool = True,
  swap_hands: bool = True,
) -> "callable[[], mujoco.MjSpec]":
  """Wrap an ``EntityCfg.spec_fn`` so its returned MjSpec carries our hardware.

  The wrapper is tagged so double-wrapping is a no-op.
  """
  if getattr(base_spec_fn, _PAYLOAD_WRAPPED_ATTR, False):
    return base_spec_fn

  def _wrapped() -> mujoco.MjSpec:
    return apply_payloads_to_spec(
      base_spec_fn(), include_jetson=include_jetson, swap_hands=swap_hands
    )

  setattr(_wrapped, _PAYLOAD_WRAPPED_ATTR, True)
  return _wrapped
