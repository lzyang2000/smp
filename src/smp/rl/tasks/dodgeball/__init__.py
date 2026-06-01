"""SMP dodgeball task — registers ``Smp-Dodgeball-G1`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.dodgeball.dodgeball_env_cfg import g1_dodgeball_smp_env_cfg

_dodgeball_rl = unitree_g1_smp_ppo_runner_cfg()
_dodgeball_rl.experiment_name = "smp_dodgeball_g1"
_dodgeball_rl.run_name = "smp_dodgeball_g1"

register_mjlab_task(
  task_id="Smp-Dodgeball-G1",
  env_cfg=g1_dodgeball_smp_env_cfg(play=False),
  play_env_cfg=g1_dodgeball_smp_env_cfg(play=True),
  rl_cfg=_dodgeball_rl,
)

__all__ = ["g1_dodgeball_smp_env_cfg"]
