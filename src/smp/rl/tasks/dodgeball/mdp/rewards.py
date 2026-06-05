"""Dodgeball reward component: stay away from projectiles, move little.

SMP-gated via the generic ``smp.rl.rewards.task_smp_product``. Mirrors
MimicKit's ``compute_dodge_reward``: a distance term (reward grows as the
nearest projectile gets farther) plus a small stillness term on the horizontal
root velocity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from smp.rl.tasks.dodgeball.mdp.commands import DodgeballCommand


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def dodge(
  env: "ManagerBasedRlEnv",
  command_name: str,
  pos_w: float = 0.9,
  vel_w: float = 0.1,
  pos_scale: float = 0.3,
  vel_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``pos_w·(1 - exp(-pos_scale·d_min)) + vel_w·exp(-vel_scale·‖v_xy‖²)``.

  ``d_min`` is the distance to the nearest projectile; the second term rewards
  not running away (low horizontal root speed)."""
  asset = env.scene[asset_cfg.name]
  cmd: "DodgeballCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  proj_pos, _ = cmd.proj_states()
  pos_diff = proj_pos - asset.data.root_link_pos_w.unsqueeze(1)
  pos_err = torch.linalg.norm(pos_diff, dim=-1).min(dim=1).values
  pos_reward = 1.0 - torch.exp(-pos_scale * pos_err)

  vel_err = torch.sum(asset.data.root_link_lin_vel_w[:, :2] ** 2, dim=-1)
  vel_reward = torch.exp(-vel_scale * vel_err)

  return pos_w * pos_reward + vel_w * vel_reward


def dodge_link_cbf_reward(
  env: "ManagerBasedRlEnv",
  robot_name: str = "robot",
  ball_name: str = "projectile",
  ball_geom: str = "projectile/ball_collision",
  alpha: float = 1.0,
  margin: float = 0.05,
  constraint_clip: float = 2.0,
  z_active: float = 0.25,
  min_ball_speed: float = 0.5,
  reduce: str = "min",
) -> torch.Tensor:
  """Per-link full-body control-barrier dodge reward (<= 0; scale with the term weight).

  Verbatim port of OUR (AMP-task) ``dodge_link_cbf_reward``
  (``src/tasks/amp_loco/mdp/rewards.py``, used at weight 0.27): for every robot link
  form a clearance ``h = ||p_ball - p_link|| - (r_ball + r_link)`` (per-env true ball
  radius + the link's collision cross-section radius + ``margin``), its closing rate
  ``h_dot``, and the discrete-time CBF constraint ``min(h_dot + alpha*h, 0)`` -- 0 when
  the link is safely clearing, negative by the violation. ``reduce="min"`` charges only
  the single most-binding link. Gated on a live threat (ball airborne + moving). Being
  ZERO when safe, it leaves the SMP stand/sidestep behavior untouched and only penalizes
  leaving a *limb* in the ball's path -- the residual arm/foot hits the root-only ``dodge``
  term cannot see. Privileged (ground-truth positions). Unlike the ``dodge`` task term this
  is added as its OWN additive reward (NOT inside ``task_smp_product``), matching how OUR
  task sums it as a separate RewardTermCfg rather than gating it by the motion prior.
  """
  robot: Entity = env.scene[robot_name]
  ball: Entity = env.scene[ball_name]

  # Cache static per-link safety radii + the ball geom id (radius is the only randomized
  # quantity). For each robot body take the max cross-section radius over its round
  # (sphere/capsule/cylinder) collision geoms, default 0.05 m, then add `margin`.
  if not hasattr(env, "_dodge_link_radii"):
    mjm = env.sim.mj_model
    round_types = {
      int(mujoco.mjtGeom.mjGEOM_SPHERE),
      int(mujoco.mjtGeom.mjGEOM_CAPSULE),
      int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    }
    body_r: dict[int, float] = {}
    for g in range(mjm.ngeom):
      if int(mjm.geom_type[g]) in round_types:
        b = int(mjm.geom_bodyid[g])
        body_r[b] = max(body_r.get(b, 0.0), float(mjm.geom_size[g, 0]))
    body_ids = robot.indexing.body_ids.tolist()
    radii = [body_r.get(int(b), 0.05) for b in body_ids]
    env._dodge_link_radii = torch.tensor(  # type: ignore[attr-defined]
      radii, device=env.device, dtype=torch.float32
    ) + margin
    env._dodge_ball_geom_id = int(mjm.geom(ball_geom).id)  # type: ignore[attr-defined]

  r_link = env._dodge_link_radii  # (L,)
  r_ball = env.sim.model.geom_size[:, env._dodge_ball_geom_id, 0]  # (N,) per-env radius

  ball_p = ball.data.root_link_pos_w  # (N, 3)
  ball_v = ball.data.root_link_lin_vel_w  # (N, 3)
  link_p = robot.data.body_link_pos_w  # (N, L, 3)
  link_v = robot.data.body_link_lin_vel_w  # (N, L, 3)

  rel = ball_p.unsqueeze(1) - link_p  # (N, L, 3) link -> ball
  d = rel.norm(dim=-1).clamp_min(1e-6)  # (N, L)
  h = d - (r_ball.unsqueeze(1) + r_link.unsqueeze(0))  # (N, L)
  h_dot = (rel * (ball_v.unsqueeze(1) - link_v)).sum(dim=-1) / d  # (N, L)
  cbf = (h_dot + alpha * h).clamp(min=-constraint_clip, max=0.0)  # (N, L) <= 0
  if reduce == "sum":
    agg = cbf.sum(dim=1)
  elif reduce == "mean":
    agg = cbf.mean(dim=1)
  else:  # "min"
    agg = cbf.min(dim=1).values

  threat = (ball_p[:, 2] > z_active) & (ball_v[:, :2].norm(dim=-1) > min_ball_speed)
  return torch.where(threat, agg, torch.zeros_like(agg))
