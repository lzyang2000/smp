"""Dodgeball reward component: stay away from projectiles, move little.

SMP-gated via the generic ``smp.rl.rewards.task_smp_product``. Mirrors
MimicKit's ``compute_dodge_reward``: a distance term (reward grows as the
nearest projectile gets farther) plus a small stillness term on the horizontal
root velocity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
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
