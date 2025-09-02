# tools/sb3_extensions/sac_with_alpha_lr.py
from __future__ import annotations
from typing import Callable, Optional, Union
import torch as th

# Repodaki SB3 (submodule) kurulu olduğundan normal import yeterli
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.type_aliases import Schedule

class SACWithAlphaLR(SAC):
    """
    SB3 SAC + ayrı alpha (entropy coef) learning rate desteği.
    Kullanım:
        SACWithAlphaLR(..., ent_coef="auto" or "auto_0.1", ent_coef_lr=1e-4)
    Not:
        - ent_coef sabit bir float ise (öğrenilmiyorsa) ent_coef_optimizer yoktur.
        - ent_coef_lr float ya da Schedule (Callable[[float], float]) olabilir.
    """

    def __init__(
        self,
        *args,
        ent_coef_lr: Union[float, Schedule, None] = None,
        **kwargs,
    ):
        self._alpha_lr_schedule: Optional[Schedule] = None
        self._user_alpha_lr = ent_coef_lr
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        # Önce SB3'ün kurulumunu yap
        super()._setup_model()

        # Kullanıcı schedule vermediyse: actor/critic ile aynı schedule
        if self._user_alpha_lr is None:
            self._alpha_lr_schedule = self.lr_schedule
        elif isinstance(self._user_alpha_lr, (float, int)):
            const_lr = float(self._user_alpha_lr)
            self._alpha_lr_schedule = lambda _: const_lr
        else:
            # Callable schedule
            self._alpha_lr_schedule = self._user_alpha_lr

        # ent_coef='auto' ise optimizer mevcuttur -> başlangıç LR'ını bizim schedule'dan ayarla
        if getattr(self, "ent_coef_optimizer", None) is not None and self._alpha_lr_schedule is not None:
            for g in self.ent_coef_optimizer.param_groups:
                g["lr"] = self._alpha_lr_schedule(1.0)

    def _update_learning_rate(self, optimizers):
        """
        Orijinal SB3 tüm optimizer'lara aynı LR'ı basar.
        Burada actor/critic -> base_lr, alpha -> alpha_lr olacak.
        """
        base_lr = self.lr_schedule(self._current_progress_remaining)
        alpha_lr = (
            self._alpha_lr_schedule(self._current_progress_remaining)
            if self._alpha_lr_schedule is not None
            else base_lr
        )

        for opt in optimizers:
            # ent_coef_optimizer varsa, ona alpha_lr uygula; diğerlerine base_lr
            lr_to_set = alpha_lr if (getattr(self, "ent_coef_optimizer", None) is not None and opt is self.ent_coef_optimizer) else base_lr
            for g in opt.param_groups:
                g["lr"] = lr_to_set

        # (Opsiyonel) Alpha LR'ını logger'a yaz ki W&B/TB'de görebilesin
        try:
            self.logger.record("train/alpha_lr", float(alpha_lr))
        except Exception:
            pass
