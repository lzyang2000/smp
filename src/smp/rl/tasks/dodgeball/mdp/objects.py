"""Projectile (dodgeball) entity for the dodgeball task.

A single floating-base sphere with a freejoint and no actuators — parked far
away until ``DodgeballCommand`` launches it ballistically at the character.
"""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

PROJECTILE_BODY_NAME = "dodgeball"


def get_projectile_cfg(
  radius: float = 0.0762,
  mass: float = 0.136,
  color: tuple[float, float, float, float] = (0.9725, 0.42, 0.1137, 1.0),
  spawn_pos: tuple[float, float, float] = (10.0, 10.0, 0.0762),
) -> EntityCfg:
  """A free-floating sphere projectile.

  Matched to OUR (AMP-task) ball (``src/assets/objects/ball.py``) so the two tasks
  throw the identical object: a 6-inch-diameter foam dodgeball -> default radius
  0.0762 m, fixed mass 0.136 kg. The geom is named ``ball_collision`` so a per-episode
  size-randomization event (``dr.geom_size`` on this geom) can write the radius
  absolutely, exactly as OUR task does. Mass stays fixed (the launch writes velocity
  directly, mass-independent; hit detection is geometric), so only the radius matters
  for the dodge dynamics.
  """

  def spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=PROJECTILE_BODY_NAME)
    body.add_freejoint()
    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size[0] = radius
    geom.rgba[:] = color
    geom.mass = mass
    geom.condim = 3
    geom.name = "ball_collision"
    return spec

  return EntityCfg(
    spec_fn=spec_fn,
    init_state=EntityCfg.InitialStateCfg(pos=spawn_pos),
  )


def get_projectile_contact_sensor_cfg(
  name: str,
  proj_entity_name: str,
  robot_entity_name: str = "robot",
  robot_root_body: str = "pelvis",
  history_length: int = 0,
) -> ContactSensorCfg:
  """Contact sensor for projectile↔robot impacts.

  ``primary`` is the projectile body, ``secondary`` the whole robot subtree, so
  only projectile-vs-character contacts are reported — projectile-vs-ground
  contacts are filtered out by the sensor itself (MimicKit instead needed an
  explicit proximity gate to discard ground contacts). ``reduce="netforce"``
  yields the net contact force in the **global** frame, matching MimicKit's
  world-frame ``contact_force[..., 0:2]`` read. Set ``history_length`` to the
  env decimation to catch brief impacts that resolve mid-substep.
  """
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(
      mode="subtree", pattern=PROJECTILE_BODY_NAME, entity=proj_entity_name
    ),
    secondary=ContactMatch(
      mode="subtree", pattern=robot_root_body, entity=robot_entity_name
    ),
    fields=("force",),
    reduce="netforce",
    history_length=history_length,
  )
