"""Dodgeball command — ballistic projectile launcher + state machine.

Each env owns one or more projectile entities that are parked far away until a
per-env random *trigger time* elapses, at which point the projectile is launched
on a ballistic arc aimed at (a noisy lead-prediction of) the character's target
body. The command exposed to the policy is the heading-frame projectile position
and velocity, so the observation is yaw-invariant (mirrors MimicKit's
``compute_dodgeball_observations``).

Unlike the periodic commands (location/steering) this term does not use the
built-in resampling timer — projectile triggering is its own per-projectile
clock managed in ``_update_command``. ``resampling_time_range`` is set huge so
the only ``_resample`` call comes from env reset, where it parks the projectiles
and rolls a fresh trigger time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class DodgeballCommand(CommandTerm):
  """Ballistic projectile launcher aimed at the character."""

  cfg: DodgeballCommandCfg

  def __init__(self, cfg: DodgeballCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.projectiles: list[Entity] = [env.scene[n] for n in cfg.projectile_names]
    self.num_proj = len(self.projectiles)
    assert self.num_proj > 0

    self.target_body_id = int(
      self.robot.find_bodies([cfg.target_body], preserve_order=True)[0][0]
    )

    # Per-env, per-projectile launch clock and last-step velocity (for the
    # delta-v hit test in terminations).
    self.trigger_times = torch.zeros(self.num_envs, self.num_proj, device=self.device)
    self.prev_proj_vel = torch.zeros(
      self.num_envs, self.num_proj, 3, device=self.device
    )

    # Heading-frame obs: [local_pos(3), local_vel(3)] per projectile, flattened.
    self.command_b = torch.zeros(self.num_envs, self.num_proj * 6, device=self.device)

    self.metrics["proj_min_dist"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  # --- State accessors -----------------------------------------------------

  def proj_states(self) -> tuple[torch.Tensor, torch.Tensor]:
    """World-frame projectile ``(pos, lin_vel)``, each ``(num_envs, P, 3)``."""
    pos = torch.stack([e.data.root_link_pos_w for e in self.projectiles], dim=1)
    vel = torch.stack([e.data.root_link_lin_vel_w for e in self.projectiles], dim=1)
    return pos, vel

  def _elapsed(self) -> torch.Tensor:
    """Per-env episode time in seconds (reset to 0 on env reset)."""
    return self._env.episode_length_buf.to(self.device) * self._env.step_dt

  def _sample_dt(self, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, device=self.device).uniform_(
      self.cfg.trigger_time_min, self.cfg.trigger_time_max
    )

  def _park_pos(self, n: int, env_ids: torch.Tensor, proj_idx: int) -> torch.Tensor:
    """Inactive projectile position: per-env-origin offset (avoids cross-env
    contacts) far from where the character spawns."""
    origins = self._env.scene.env_origins[env_ids]
    pos = origins.clone()
    pos[:, 0] += 10.0 + proj_idx
    pos[:, 1] += 10.0
    pos[:, 2] = self.cfg.proj_radius
    return pos

  # --- Lifecycle -----------------------------------------------------------

  def _update_metrics(self) -> None:
    proj_pos, _ = self.proj_states()
    root_pos = self.robot.data.root_link_pos_w
    dist = torch.linalg.norm(proj_pos - root_pos.unsqueeze(1), dim=-1)
    self.metrics["proj_min_dist"] = dist.min(dim=1).values

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Env-reset hook: park every projectile and roll a fresh trigger time."""
    n = int(env_ids.numel())
    if n == 0:
      return
    reset_state = torch.zeros(n, 13, device=self.device)
    reset_state[:, 3] = 1.0  # identity quat (w, x, y, z)
    for p, ent in enumerate(self.projectiles):
      reset_state[:, 0:3] = self._park_pos(n, env_ids, p)
      ent.write_root_state_to_sim(reset_state, env_ids)
      self.prev_proj_vel[env_ids, p] = 0.0
    self.trigger_times[env_ids] = self._elapsed()[env_ids].unsqueeze(
      -1
    ) + self._sample_dt((n, self.num_proj))

  def _update_command(self) -> None:
    elapsed = self._elapsed()
    proj_pos, proj_vel = self.proj_states()
    # Default the delta-v reference to the current (post-physics) velocity;
    # freshly-launched projectiles overwrite it with their launch velocity.
    self.prev_proj_vel[:] = proj_vel

    for p, ent in enumerate(self.projectiles):
      trigger = elapsed >= self.trigger_times[:, p]
      launch_ids = trigger.nonzero(as_tuple=False).flatten()
      if launch_ids.numel() == 0:
        continue
      launch_pos, launch_vel = self._ballistic_launch(launch_ids)
      m = int(launch_ids.numel())
      state = torch.zeros(m, 13, device=self.device)
      state[:, 0:3] = launch_pos
      state[:, 3] = 1.0
      state[:, 7:10] = launch_vel
      ent.write_root_state_to_sim(state, launch_ids)
      self.prev_proj_vel[launch_ids, p] = launch_vel
      self.trigger_times[launch_ids, p] = elapsed[launch_ids] + self._sample_dt((m,))

    self._write_obs(proj_pos, proj_vel)

  # --- Launch + obs --------------------------------------------------------

  def _ballistic_launch(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (launch_pos, launch_vel) so the projectile arcs into a noisy
    lead-prediction of the target body (mirrors MimicKit's ballistic solve)."""
    n = int(env_ids.numel())
    g = float(self._env.cfg.sim.mujoco.gravity[2])

    target_pos = self.robot.data.body_link_pos_w[env_ids, self.target_body_id].clone()
    target_pos += self.cfg.aim_noise_scale * torch.randn_like(target_pos)
    target_vel = self.robot.data.body_link_lin_vel_w[env_ids, self.target_body_id]

    c = self.cfg

    def _u(lo: float, hi: float) -> torch.Tensor:
      return torch.empty(n, device=self.device).uniform_(lo, hi)

    speed = _u(c.proj_speed_min, c.proj_speed_max)
    rand_dist = _u(c.proj_dist_min, c.proj_dist_max)
    travel = rand_dist / speed

    # BIMODAL threat (matches OUR throw_ball_on_dwell high_throw_fraction): each ball is a
    # DUCK ball (launched low, arcs UP to torso/head) or a DESCENDING ball (launched high,
    # arrives at the lower body). Both solve vz to their impact target so neither lands short.
    high = torch.rand(n, device=self.device) < c.high_throw_fraction
    launch_h = torch.where(
      high, _u(c.high_launch_h_min, c.high_launch_h_max),
      _u(c.descend_launch_h_min, c.descend_launch_h_max),
    )
    z_target = torch.where(
      high, _u(c.high_target_z_min, c.high_target_z_max),
      _u(c.low_target_z_min, c.low_target_z_max),
    )
    # Vertical launch component so the arc passes through z_target at impact (g is negative).
    vel_z = (z_target - launch_h) / travel - 0.5 * g * travel

    # Lead the target by its current velocity over the travel time.
    target_pred = target_pos + target_vel * travel.unsqueeze(-1)

    # FRONT cone (matches OUR throw): launch from within +/- angle_deg of the robot's HEADING and
    # AHEAD of it, so the ball comes at the front (camera-relevant). theta is the azimuth from the
    # target to the launch point; setting it along the heading puts the launch point in front of the
    # robot and the velocity points back toward it. (SMP's default was a full 360-deg azimuth.)
    q = self.robot.data.root_link_quat_w[env_ids]  # (n, 4) wxyz
    yaw = torch.atan2(
      2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
      1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
    )
    half = math.radians(c.angle_deg)
    theta = yaw + torch.empty(n, device=self.device).uniform_(-half, half)
    proj_pos = torch.zeros(n, 3, device=self.device)
    proj_pos[:, 0] = target_pred[:, 0] + rand_dist * torch.cos(theta)
    proj_pos[:, 1] = target_pred[:, 1] + rand_dist * torch.sin(theta)
    proj_pos[:, 2] = launch_h

    proj_delta = target_pred - proj_pos
    proj_delta[:, 2] = 0.0
    proj_dir = torch.nn.functional.normalize(proj_delta, dim=-1)
    proj_vel = speed.unsqueeze(-1) * proj_dir
    proj_vel[:, 2] = vel_z
    return proj_pos, proj_vel

  def _write_obs(self, proj_pos: torch.Tensor, proj_vel: torch.Tensor) -> None:
    root_pos = self.robot.data.root_link_pos_w
    heading_quat = yaw_quat(self.robot.data.root_link_quat_w)
    hq = heading_quat.unsqueeze(1).expand(-1, self.num_proj, -1).reshape(-1, 4)

    rel_pos = (proj_pos - root_pos.unsqueeze(1)).reshape(-1, 3)
    local_pos = quat_apply_inverse(hq, rel_pos).reshape(self.num_envs, self.num_proj, 3)
    local_vel = quat_apply_inverse(hq, proj_vel.reshape(-1, 3)).reshape(
      self.num_envs, self.num_proj, 3
    )
    self.command_b[:] = torch.cat([local_pos, local_vel], dim=-1).reshape(
      self.num_envs, -1
    )


@dataclass(kw_only=True)
class DodgeballCommandCfg(CommandTermCfg):
  # Triggering is managed manually (see __post_init__); the periodic resample
  # timer is pinned far in the future so it only fires on env reset.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  entity_name: str = "robot"
  projectile_names: tuple[str, ...] = ("projectile",)
  target_body: str = "torso_link"

  # --- Launch distribution: matched to OUR (AMP-task) SLOW, BIMODAL throws. --
  # OUR ``throw_ball_on_dwell`` (src/tasks/amp_loco/mdp/events.py) launches a ball
  # 2-3 m ahead that reaches the robot in ~0.58-0.63 s (a horizontal speed of only
  # ~dist/flight ~= 3.3-5.2 m/s) -- far slower / closer than the MimicKit default
  # (8-10 m at 12-15 m/s). Here travel = dist/speed, so to land in ~0.55-0.65 s from
  # 2-3 m we need speed ~= 3.5-5.0 m/s.
  #
  # BIMODAL (reproduces OUR 50/50 ``high_throw_fraction`` mix): each ball is either
  #   * a DUCK ball (frac = high_throw_fraction): launched LOW (~waist) and arcing UP to
  #     arrive at torso/head height -> the robot must duck under it; or
  #   * a DESCENDING ball (the rest): launched HIGH (~2 m) and arriving at the LOWER body
  #     on the way down -> the robot sidesteps / lifts a leg over it.
  # We keep SMP's "solve vz to a target impact height" mechanism (robust: never lands
  # short on the fixed speed-based travel) but make BOTH the launch height and the impact
  # target height bimodal -- so the duck-vs-sidestep outcome matches OUR throw. (OUR
  # descending mode uses pure vz0=0; solving to a low target gives a near-identical
  # descending arrival without the land-short risk of a fixed-speed travel.)
  #
  # Launch AZIMUTH is a +/- ``angle_deg`` FRONT cone of the robot's heading (matches OUR throw,
  # which spawns the ball in the frontal cone -- camera-relevant), NOT the SMP/MimicKit 360-deg
  # default. Radius matches OUR foam dodgeball (0.0762 m); per-episode size DR (0.075-0.125 m) is
  # added by the env cfg. ``aim_noise_scale`` 0.1 + lead-the-target both match OUR throw.
  proj_radius: float = 0.0762
  proj_dist_min: float = 2.0
  proj_dist_max: float = 3.0
  proj_speed_min: float = 3.5
  proj_speed_max: float = 5.0
  trigger_time_min: float = 1.0
  trigger_time_max: float = 4.0
  aim_noise_scale: float = 0.1
  angle_deg: float = 25.0  # +/- heading half-cone for the launch azimuth (matches OUR throw)
  # Bimodal vertical profile (matches OUR throw_ball_on_dwell).
  high_throw_fraction: float = 0.5
  high_launch_h_min: float = 0.4   # DUCK ball launch height (~waist)
  high_launch_h_max: float = 0.9
  high_target_z_min: float = 1.0   # DUCK ball impact height (torso/head)
  high_target_z_max: float = 1.5
  descend_launch_h_min: float = 1.5  # DESCENDING ball launch height (~2 m)
  descend_launch_h_max: float = 2.3
  low_target_z_min: float = 0.2    # DESCENDING ball impact height (lower body)
  low_target_z_max: float = 0.9

  def __post_init__(self) -> None:
    for lo, hi, name in (
      (self.proj_dist_min, self.proj_dist_max, "proj_dist"),
      (self.proj_speed_min, self.proj_speed_max, "proj_speed"),
      (self.trigger_time_min, self.trigger_time_max, "trigger_time"),
      (self.high_launch_h_min, self.high_launch_h_max, "high_launch_h"),
      (self.high_target_z_min, self.high_target_z_max, "high_target_z"),
      (self.descend_launch_h_min, self.descend_launch_h_max, "descend_launch_h"),
      (self.low_target_z_min, self.low_target_z_max, "low_target_z"),
    ):
      if hi < lo:
        msg = f"{name}_max ({hi}) must be >= {name}_min ({lo})."
        raise ValueError(msg)

  def build(self, env: "ManagerBasedRlEnv") -> DodgeballCommand:
    return DodgeballCommand(self, env)
