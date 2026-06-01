"""G1 dodgeball task with SMP guidance.

A projectile is periodically launched on a ballistic arc aimed at the
character; the policy must dodge it. Reward is a distance-from-projectile term
(plus a small stillness term) gated by the SMP guidance reward; an episode ends
in failure if a projectile strikes the character.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
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
  # task = avoid projectiles (+ stay roughly in place), gated by SMP.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": ((mdp.dodge, 1.0, {"command_name": "dodgeball"}),),
      "ws": 4,
    },
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_lafan_run.pt"
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.terminations["projectile_hit"] = TerminationTermCfg(
    func=mdp.projectile_hit,
    params={
      "command_name": "dodgeball",
      "hit_dist": 0.8,
      "hit_delta_v_threshold": 1.5,
      "contact_sensor_names": ("proj_contact_projectile",),
      "hit_force_threshold": 0.1,
    },
  )

  return cfg
