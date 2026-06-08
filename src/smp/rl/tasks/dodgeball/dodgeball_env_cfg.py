"""G1 dodgeball task with SMP guidance.

A projectile is periodically launched on a ballistic arc aimed at the
character; the policy must dodge it. Reward is a distance-from-projectile term
(plus a small stillness term) gated by the SMP guidance reward; an episode ends
in failure if a projectile strikes the character.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import action_rate_l2, bad_orientation, dr, joint_acc_l2, joint_pos_limits
from mjlab.tasks.tracking.mdp import self_collision_cost
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
  # --- Regularizers + terminal penalty (ported from OUR AMP dodge task) ----
  # ADDED as separate additive RewardTermCfgs (NOT inside task_smp_product): we keep the
  # prior-vs-discriminator difference as the *only* naturalness-mechanism difference, but
  # otherwise match OUR AMP task's hand-tuned shaping so the two reward stacks line up.
  # Weights are copied verbatim from g1_amp_dodge_mimickit (both envs use scale_rewards_by_dt
  # at the same 50 Hz step, so the per-second weights transfer directly).
  #   - is_terminated -200, EXCLUDING projectile_hit (a hit ends the episode with no penalty
  #     spike; only genuine falls are penalized -- see is_terminated_except).
  #   - joint_acc / joint_pos_limits / action_rate: standard smoothness/limit regularizers.
  #   - self_collisions: reuses the base env's `self_collision` sensor (found-based; the cost
  #     fn falls back to `found` when the sensor has no force history).
  # NOTE: OUR AMP task also has `foot_slip` (-0.25), but it is gated on the locomotion command
  # (penalize sliding only while *commanded to move*). SMP is stand-in-place (no twist command),
  # so the gate is never active; an UNGATED foot-slip penalty would instead punish the sidestep
  # that IS SMP's dodge. So foot_slip is intentionally omitted here. `dodge_link_cbf` is likewise
  # omitted -- it is dodge *shaping*, not a regularizer, and keeping it out preserves the clean
  # prior-only-vs-discriminator-only comparison.
  cfg.rewards["is_terminated"] = RewardTermCfg(
    func=mdp.is_terminated_except,
    weight=-200.0,
    params={"exclude_terms": ("projectile_hit",)},
  )
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(func=joint_acc_l2, weight=-2.5e-7)
  cfg.rewards["joint_pos_limits"] = RewardTermCfg(func=joint_pos_limits, weight=-10.0)
  cfg.rewards["action_rate_l2"] = RewardTermCfg(func=action_rate_l2, weight=-0.01)
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=self_collision_cost,
    weight=-0.1,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    # Mixed-dodge prior trained on datasets/npz/mixed_dodge with
    # norm_stats_ours.npz (scripts/pretrain.py --name mixed_dodge). Copy the
    # final logs/pretrain/mixed_dodge/<ts>/pretrained.pt here to update it.
    "datasets/pretrain_ckpt/mixed_dodge_prior.pt"
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
  # Drop the base env's self-collision termination (pelvis-subtree self-contact).
  # It was ending ~18% of dodgeball episodes -- the robot tearing itself apart
  # rather than failing to dodge -- which both confounds the hit-rate comparison
  # against OUR AMP state oracle (where self-collision is only a reward penalty,
  # never a termination) and wastes episodes on a failure mode orthogonal to the
  # dodge task. Same treatment getup already applies (getup_env_cfg.py).
  cfg.terminations.pop("self_collision", None)
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
