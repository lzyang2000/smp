"""Dodgeball termination: the character was hit by a projectile.

Faithful to MimicKit's ``compute_dodgeball_fail_flags``: a hit is the logical
OR of two detectors, both gated by a proximity check (projectile within
``hit_dist`` of any robot body):

1. **contact force** — the projectile↔robot contact force exceeds
   ``hit_force_threshold`` (read from a dedicated contact sensor, see
   ``get_projectile_contact_sensor_cfg``);
2. **velocity change** — the projectile's measured velocity deviates from the
   gravity-only ballistic prediction by more than ``hit_delta_v_threshold``,
   i.e. it bounced off the character.

The two are redundant by design (a contact strong enough to register force also
deflects the projectile); the force branch fires with lower latency, the
delta-v branch needs no sensor. Pass ``contact_sensor_names=()`` to use the
delta-v branch alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor.contact_sensor import ContactSensor

  from smp.rl.tasks.dodgeball.mdp.commands import DodgeballCommand


def _contact_force_xy(
  env: "ManagerBasedRlEnv", sensor_names: tuple[str, ...]
) -> torch.Tensor:
  """Per-projectile xy-plane contact-force magnitude, shape ``(N, P)``.

  One sensor per projectile (primary = that projectile, secondary = robot), so
  ``data.force`` is the net global-frame projectile↔robot force. Uses the
  per-substep history when available (catches brief mid-step impacts), else the
  latest force."""
  cols: list[torch.Tensor] = []
  for sname in sensor_names:
    sensor: "ContactSensor" = env.scene[sname]
    data = sensor.data
    if data.force_history is not None:
      # force_history: (B, N, H, 3) -> max xy-magnitude over slots and substeps.
      fmag = torch.linalg.norm(data.force_history[..., :2], dim=-1)  # (B, N, H)
      cols.append(fmag.amax(dim=(-1, -2)))  # (B,)
    else:
      assert data.force is not None
      fmag = torch.linalg.norm(data.force[..., :2], dim=-1)  # (B, N)
      cols.append(fmag.amax(dim=-1))  # (B,)
  return torch.stack(cols, dim=1)  # (N, P)


def projectile_hit(
  env: "ManagerBasedRlEnv",
  command_name: str,
  hit_dist: float = 0.8,
  hit_delta_v_threshold: float = 1.5,
  contact_sensor_names: tuple[str, ...] = (),
  hit_force_threshold: float = 0.1,
) -> torch.Tensor:
  """True for envs where a projectile struck the character this step."""
  cmd: "DodgeballCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]
  body_pos = cmd.robot.data.body_link_pos_w  # (N, B, 3)
  proj_pos, proj_vel = cmd.proj_states()  # (N, P, 3)
  prev_vel = cmd.prev_proj_vel  # (N, P, 3)
  g = float(env.cfg.sim.mujoco.gravity[2])
  dt = env.step_dt

  # Proximity gate: nearest robot body within hit_dist of the projectile.
  delta = proj_pos.unsqueeze(2) - body_pos.unsqueeze(1)  # (N, P, B, 3)
  near = torch.linalg.norm(delta, dim=-1).min(dim=2).values < hit_dist  # (N, P)

  # Branch (b): velocity change beyond free-fall expectation.
  expected_vel = prev_vel.clone()
  expected_vel[..., 2] += g * dt
  delta_v = torch.linalg.norm(proj_vel - expected_vel, dim=-1)  # (N, P)
  hit = (delta_v > hit_delta_v_threshold) & near

  # Branch (a): projectile↔robot contact force.
  if contact_sensor_names:
    assert len(contact_sensor_names) == cmd.num_proj, (
      f"got {len(contact_sensor_names)} contact sensors for {cmd.num_proj} "
      "projectiles; expected one per projectile in projectile order."
    )
    contact_force_xy = _contact_force_xy(env, contact_sensor_names)  # (N, P)
    hit = hit | ((contact_force_xy > hit_force_threshold) & near)

  return hit.any(dim=1)
