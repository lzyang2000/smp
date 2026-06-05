"""G1 dodgeball task with SMP guidance.

A projectile is periodically launched on a ballistic arc aimed at the
character; the policy must dodge it. Reward is a distance-from-projectile term
(plus a small stillness term) gated by the SMP guidance reward; an episode ends
in failure if a projectile strikes the character.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import bad_orientation, dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.dodgeball import mdp


def g1_dodgeball_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 dodgeball env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Scene: add the projectile entity + its contact sensor ---------------
  cfg.scene.entities["projectile"] = mdp.get_projectile_cfg()
  proj_contact = mdp.get_projectile_contact_sensor_cfg(
    name="proj_contact_projectile",
    proj_entity_name="projectile",
    robot_entity_name="robot",
    history_length=cfg.decimation,  # cover one policy step of substeps
  )
  cfg.scene.sensors = (*cfg.scene.sensors, proj_contact)
  # Projectile impacts + landings add a few contacts per env beyond the robot.
  cfg.sim.nconmax = 60

  # --- Commands ------------------------------------------------------------
  cfg.commands["dodgeball"] = mdp.DodgeballCommandCfg(
    entity_name="robot",
    projectile_names=("projectile",),
    target_body="torso_link",
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "dodgeball"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Rewards -------------------------------------------------------------
  # task = avoid projectiles (+ stay roughly in place), gated by SMP. The ``dodge`` task
  # term IS OUR ``mimickit_dodge`` reward (identical formula: 0.9*(1-exp(-0.3*d)) +
  # 0.1*exp(-||v_xy||^2)); it is multiplied by the SMP guidance reward r_smp -- THE SMP
  # PRIOR MECHANISM UNDER TEST -- which stays intact (OUR AMP analog is the AMP
  # discriminator reward, intentionally NOT ported).
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": ((mdp.dodge, 1.0, {"command_name": "dodgeball"}),),
      "ws": 4,
    },
  )
  # NOTE: no dodge_link_cbf and no hand-designed regularizers here -- that is intentional and
  # FAITHFUL TO SMP: the reward is purely the SMP-gated task (dodge x r_smp), with the diffusion
  # prior (not hand-tuned penalties) providing naturalness/stability. The trained AMP state oracle
  # (g1_amp_dodge_mimickit, the 27% checkpoint) predates dodge_link_cbf and also did not use it, so
  # dropping it here both matches that checkpoint and keeps the SMP method clean.

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    # Balanced, mirrored full-LAFAN locomotion prior (built by
    # scripts/build_full_lafan_prior.sh). Replaces the forward-run-only
    # pretrained_lafan_run.pt so the policy steps in place / sidesteps to
    # dodge instead of running away.
    "datasets/pretrain_ckpt/full_lafan_prior.pt"
  )
  # Per-episode projectile SIZE randomization, matched to OUR (AMP-task)
  # ``randomize_ball_size``: write the sphere radius absolutely to a uniform
  # 0.075-0.125 m (15-25 cm diameter) each reset, on the ``ball_collision`` geom
  # (dr.geom_size recomputes geom_rbound/geom_aabb so the broadphase stays
  # consistent). Mass stays fixed (the launch writes velocity directly; hit
  # detection is geometric), so only the radius varies the dodge dynamics. Note
  # the launcher's proj_h_min clamp uses the nominal radius; that is unaffected.
  cfg.events["randomize_projectile_size"] = EventTermCfg(
    func=dr.geom_size,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("projectile", geom_names=("ball_collision",)),
      "operation": "abs",
      "ranges": (0.075, 0.125),
      "axes": [0],
    },
  )

  # --- Terminations --------------------------------------------------------
  # Matched to OUR (AMP-task) dodge terminations: a raised base-height floor (0.45 m,
  # vs SMP's old 0.3), a bad-orientation cut (torso tilted > 80 deg), and the
  # sustained "collapsed crouch" (< 0.55 m held > 0.3 s) -- so the SMP robot must
  # stand/sidestep upright instead of scraping low, exactly like OUR task. The
  # projectile-hit detection uses OUR ball_hit params (hit_dist 1.0, delta_v 1.5,
  # contact OR, hit_z_min 0.3). The hit ends the episode with no extra penalty (the
  # cost is the lost SMP-gated future reward), matching OUR is_terminated(exclude
  # ball_hit) intent; SMP has no -200 spike to exclude, so this is the same outcome.
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.45,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.terminations["bad_orientation"] = TerminationTermCfg(
    func=bad_orientation,
    params={
      "limit_angle": math.radians(80.0),
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.terminations["collapsed_crouch"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum_sustained,
    params={
      "minimum_height": 0.55,
      "duration_s": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.terminations["projectile_hit"] = TerminationTermCfg(
    func=mdp.projectile_hit,
    params={
      "command_name": "dodgeball",
      "hit_dist": 1.0,
      "hit_delta_v_threshold": 1.5,
      "contact_sensor_names": ("proj_contact_projectile",),
      "hit_force_threshold": 0.1,
      "hit_z_min": 0.3,
    },
  )

  return cfg
