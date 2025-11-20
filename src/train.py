import math
import os
import random
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, List, Tuple

import hydra
import numpy as np
import optuna
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_

from src.model import load_model_and_tokenizer
from src.preprocess import CarbonTracker, build_dataloaders

################################################################################
#                              REPRODUCIBILITY                                 #
################################################################################

def set_random_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

################################################################################
#                        REGION/FORECAST COMPONENTS                            #
################################################################################
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class _RegionVAE(nn.Module):
    def __init__(self, seq_len: int = 288, embed_dim: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(seq_len, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
        )
        self.mu = nn.Linear(128, embed_dim)
        self.logvar = nn.Linear(128, embed_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

class _LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 32, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(h[:, -1:])

################################################################################
#                         GAUSSIAN-PROCESS UTILS                                #
################################################################################
class _ZoomGP:
    def __init__(self, beta: float, dim: int, bandwidth: float):
        kernel = RBF(length_scale=bandwidth) + WhiteKernel(1e-4)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
        self.X, self.y = [], []
        self.fitted = False
        self.beta = beta
        self.dim = dim

    def add(self, x: np.ndarray, r: float):
        self.X.append(x)
        self.y.append(r)
        if len(self.X) >= 3:
            self.gp.fit(np.stack(self.X), np.array(self.y))
            self.fitted = True

    def mean_var(self, x: np.ndarray):
        if not self.fitted:
            return 0.0, 1.0
        mu, std = self.gp.predict(x.reshape(1, -1), return_std=True)
        return mu[0], std[0] ** 2

    def ucb(self, x: np.ndarray):
        mu, var = self.mean_var(x)
        return mu + self.beta * math.sqrt(max(var, 1e-8))

################################################################################
#                               SCHEDULERS                                     #
################################################################################
class CEPGPZBScheduler:
    """Reactive LR-only GP zooming scheduler."""

    def __init__(self, opt, base_lr: float, cfg):
        self.opt = opt
        self.base_lr = base_lr
        self.lr_min, self.lr_max = map(float, cfg.lr_bounds)
        self.lr_grid = np.exp(np.linspace(np.log(self.lr_min), np.log(self.lr_max), 21))
        gp_cfg = cfg.gp_bandit
        self.zoom = _ZoomGP(beta=float(gp_cfg.beta), dim=1, bandwidth=float(gp_cfg.bandwidth_prior))
        self.update_every = int(getattr(gp_cfg, "update_steps", gp_cfg.get("window_size", 512)))
        self.cur_mul = 1.0
        self._apply()
        self.step_ct = 0

    def _apply(self):
        for pg in self.opt.param_groups:
            pg["lr"] = self.base_lr * self.cur_mul

    @property
    def cur_ga(self):
        # For LR-only scheduler GA is static (=1, handled by caller)
        return None

    def step(self, reward: float):
        self.zoom.add(np.array([math.log(self.cur_mul)]), reward)
        if self.step_ct % self.update_every == 0:
            best_mul, best_ucb = self.cur_mul, -1e9
            for m in self.lr_grid:
                ucb = self.zoom.ucb(np.array([math.log(m)]))
                if ucb > best_ucb:
                    best_mul, best_ucb = m, ucb
            self.cur_mul = float(best_mul)
            self._apply()
        self.step_ct += 1
        return self.opt.param_groups[0]["lr"], 1  # GA is always 1

class FADPBScheduler:
    """Forecast-Aware Dual-control Pareto-Bandit scheduler (LR + GA)."""

    def __init__(self, opt, base_lr: float, cfg, carbon_csv: str | None, init_ga: int, device):
        self.opt = opt
        self.base_lr = base_lr
        self.lr_min, self.lr_max = map(float, cfg.lr_bounds)
        self.ga_options: List[int] = list(sorted(set(cfg.ga_choices)))
        self.cur_lr_mul, self.cur_ga = 1.0, int(init_ga)
        self._apply_lr()
        # GP bandit ------------------------------------------------------
        gp_cfg = cfg.gp_bandit
        self.zoom = _ZoomGP(
            beta=float(gp_cfg.beta), dim=18, bandwidth=float(gp_cfg.bandwidth_prior)
        )
        self.update_every = int(getattr(gp_cfg, "update_steps", gp_cfg.get("window_size", 512)))
        # Region embedding ----------------------------------------------
        self.embed_dim = 16
        self.region_embed = self._get_region_embed(carbon_csv, cfg.meta_prior.vae_ckpt, device)
        # Forecast model -------------------------------------------------
        self.forecaster = self._load_forecaster(cfg.forecast.model_ckpt, device)
        self.horizon = int(cfg.mpc.horizon_steps)
        self.replan_sec = int(cfg.mpc.replan_interval_min) * 60
        self.population = int(cfg.mpc.cem_population)
        self.cem_iters = 3
        self._action_buf: Deque[Tuple[float, int]] = deque()
        self.last_plan_time = 0.0
        self.device = device
        self.step_ct = 0

    # ----- helpers -----------------------------------------------------
    def _apply_lr(self):
        lr = np.clip(
            self.base_lr * self.cur_lr_mul, self.base_lr * self.lr_min, self.base_lr * self.lr_max
        )
        for pg in self.opt.param_groups:
            pg["lr"] = float(lr)

    def _feat(self, lr_mul, ga):
        return np.concatenate([[math.log(lr_mul), math.log(ga)], self.region_embed])

    def _get_region_embed(self, csv_path, vae_ckpt, device):
        if not csv_path or not Path(csv_path).exists():
            return np.zeros(self.embed_dim, dtype=np.float32)
        dat = np.loadtxt(csv_path, delimiter=",", skiprows=1)[:, -1]
        dat = (dat - dat.mean()) / (dat.std() + 1e-6)
        vae = _RegionVAE(seq_len=len(dat), embed_dim=self.embed_dim).to(device)
        if Path(vae_ckpt).exists():
            vae.load_state_dict(torch.load(vae_ckpt, map_location=device), strict=False)
        with torch.no_grad():
            emb = vae.encode(torch.tensor(dat).float().view(1, -1).to(device)).cpu().numpy()[0]
        return emb.astype(np.float32)

    def _load_forecaster(self, ckpt, device):
        m = _LSTMForecaster().to(device)
        if Path(ckpt).exists():
            m.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        m.eval()
        return m

    def _forecast(self, hist: List[float]):
        if not hist:
            return [0.4] * self.horizon
        seq = torch.tensor(hist[-24:]).float().view(1, -1, 1).to(self.device)
        outs = []
        with torch.no_grad():
            cur = seq
            for _ in range(self.horizon):
                nxt = self.forecaster(cur)
                val = max(float(nxt.item()), 0.05)
                outs.append(val)
                cur = torch.cat([cur, nxt], dim=1)[:, -24:]
        return outs

    def _instant_r(self, lr_mul, ga, co2):
        mu, var = self.zoom.mean_var(self._feat(lr_mul, ga))
        return math.exp(mu) * (1.0 / ga) * (1.0 / (co2 + 1e-4))

    def _plan(self, forecast):
        pop_lr = np.random.uniform(
            np.log(self.lr_min), np.log(self.lr_max), size=(self.population, self.horizon)
        )
        pop_ga = np.random.choice(self.ga_options, size=(self.population, self.horizon))
        elite_frac = 0.2
        scores = np.empty(self.population)
        for _ in range(self.cem_iters):
            for i in range(self.population):
                r = 0.0
                for h in range(self.horizon):
                    r += self._instant_r(math.exp(pop_lr[i, h]), int(pop_ga[i, h]), forecast[h])
                scores[i] = r
            elite_idx = scores.argsort()[-max(1, int(self.population * elite_frac)) :]
            pop_lr, pop_ga = pop_lr[elite_idx], pop_ga[elite_idx]
            mu, std = pop_lr.mean(0), pop_lr.std(0) + 1e-4
            pop_lr = np.random.normal(mu, std, size=(self.population, self.horizon))
            ga_new = []
            for h in range(self.horizon):
                counts = np.bincount(pop_ga[:, h], minlength=max(self.ga_options) + 1)[
                    self.ga_options
                ]
                probs = counts / counts.sum()
                ga_new.append(np.random.choice(self.ga_options, p=probs, size=self.population))
            pop_ga = np.stack(ga_new, axis=1)
        best = scores.argmax()
        seq = [
            (float(math.exp(pop_lr[best, h])), int(pop_ga[best, h]))
            for h in range(self.horizon)
        ]
        return deque(seq)

    # ---------- public interface ---------
    def step(self, reward: float, co2_hist: List[float]):
        self.zoom.add(self._feat(self.cur_lr_mul, self.cur_ga), reward)
        if self.step_ct % self.update_every == 0:
            lr_grid = np.exp(np.linspace(np.log(self.lr_min), np.log(self.lr_max), 25))
            best_mul, best_ucb = self.cur_lr_mul, -1e9
            for m in lr_grid:
                u = self.zoom.ucb(self._feat(m, self.cur_ga))
                if u > best_ucb:
                    best_mul, best_ucb = m, u
            self.cur_lr_mul = float(best_mul)
        now = time.time()
        if not self._action_buf or now - self.last_plan_time > self.replan_sec:
            try:
                forecast = self._forecast(co2_hist)
            except Exception:
                forecast = [co2_hist[-1] if co2_hist else 0.4] * self.horizon
            self._action_buf = self._plan(forecast)
            self.last_plan_time = now
        if self._action_buf:
            planned_lr, planned_ga = self._action_buf.popleft()
            gain = self.zoom.ucb(self._feat(self.cur_lr_mul, planned_ga))
            cur = self.zoom.ucb(self._feat(self.cur_lr_mul, self.cur_ga))
            if gain > 1.05 * cur:
                self.cur_ga = planned_ga
            self.cur_lr_mul = planned_lr
        self._apply_lr()
        self.step_ct += 1
        return self.opt.param_groups[0]["lr"], self.cur_ga

################################################################################
#                            OPTIMISER BUILDER                                 #
################################################################################

def _build_optimizer(model, cfg: DictConfig):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if any(nd in n for nd in ["bias", "LayerNorm.weight"]) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.training.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    lr = getattr(
        cfg.training,
        "base_learning_rate",
        getattr(cfg.training, "learning_rate", 1e-4),
    )
    return torch.optim.AdamW(groups, lr=float(lr), betas=(0.9, 0.999))

################################################################################
#                       QUICK ACCURACY EVALUATION                              #
################################################################################
import re

def evaluate_accuracy(model, loader, tokenizer, max_batches=100):
    pat = re.compile(r"[-+]?[0-9]*\.?[0-9]+")
    model.eval()
    device = model.device
    correct = total = 0
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            prompts = [q + "\nAnswer:" for q in batch["raw_question"]]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)
            gen = model.generate(**enc, max_new_tokens=64)
            for j in range(len(prompts)):
                text = tokenizer.decode(
                    gen[j][enc.input_ids.shape[1] :], skip_special_tokens=True
                )
                pred = pat.findall(text)[-1] if pat.findall(text) else ""
                gold = pat.findall(batch["raw_answer"][j])[-1] if pat.findall(
                    batch["raw_answer"][j]
                ) else ""
                if pred.strip() == gold.strip():
                    correct += 1
                total += 1
    model.train()
    acc = correct / max(total, 1)
    return acc, {"tp": correct, "fp": 0, "fn": total - correct, "tn": 0}

################################################################################
#                              TRAINING LOOP                                   #
################################################################################

def _assert_grad(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient vanished at {n}"
            if torch.all(p.grad == 0):
                raise AssertionError(f"Zero gradient detected at {n}")


def run_training(cfg: DictConfig, do_log: bool) -> float:
    set_random_seeds(cfg.get("seed", 42))
    os.environ.setdefault("WANDB_CACHE_DIR", str(Path(".cache/wandb").absolute()))

    # -------------------------- WandB -----------------------------------
    wandb_run = (
        SimpleNamespace(log=lambda *a, **k: None, summary={}, finish=lambda: None)
        if (not do_log) or cfg.wandb.mode == "disabled"
        else wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            name=cfg.run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
    )
    if do_log and cfg.wandb.mode != "disabled":
        print(f"[wandb] run URL: {wandb_run.url}")

    # ------------------------- Model & Data -----------------------------
    tokenizer, model = load_model_and_tokenizer(cfg.model)
    with torch.no_grad():
        dummy = torch.tensor([[tokenizer.eos_token_id]], device=model.device)
        assert model(dummy).logits.shape[-1] == model.config.vocab_size

    dataloaders = build_dataloaders(cfg, tokenizer)
    train_loader, val_loader = dataloaders["train"], dataloaders["val"]
    optimizer = _build_optimizer(model, cfg)

    # ------------------------- Scheduler -------------------------------
    if cfg.training.lr_scheduler == "cep_gpzb":
        sched = CEPGPZBScheduler(
            optimizer,
            cfg.training.learning_rate,
            cfg.training.scheduler_cfg,
        )
        dyn_ga = False
    elif cfg.training.lr_scheduler == "fad_pb":
        sched = FADPBScheduler(
            optimizer,
            cfg.training.learning_rate,
            cfg.training.scheduler_cfg,
            cfg.dataset.get("carbon_trace_path", None),
            cfg.training.initial_gradient_accumulation_steps,
            model.device,
        )
        dyn_ga = True
    else:
        raise ValueError("Unknown scheduler")

    accum_default = int(
        getattr(
            cfg.training,
            "initial_gradient_accumulation_steps",
            getattr(cfg.training, "gradient_accumulation_steps", 1),
        )
    )

    accum = sched.cur_ga if getattr(sched, "cur_ga", None) else accum_default

    # ------------------------- Trackers --------------------------------
    carbon = CarbonTracker(cfg.dataset.get("carbon_trace_path", None))
    wall_start = time.time()
    co2_hist: List[float] = []
    global_step = 0

    for epoch in range(int(cfg.training.epochs)):
        model.train()
        for b_idx, batch in enumerate(train_loader):
            if cfg.mode == "trial" and b_idx > 1:
                break
            if global_step == 0:
                # Critical lifecycle assertion: shapes must match.
                assert batch["input_ids"].shape == batch["labels"].shape
            inps = {
                k: v.to(model.device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "labels"]
            }
            out = model(**inps)
            loss = out.loss / accum
            loss.backward()

            if (b_idx + 1) % accum == 0:
                _assert_grad(model)
                clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                reward = math.exp(-loss.item() * accum)
                carbon.update(); co2_hist.append(carbon.cumulative_kg)
                lr_now, ga_now = (
                    sched.step(reward, co2_hist) if dyn_ga else sched.step(reward)
                )
                if dyn_ga and ga_now != accum:
                    accum = max(1, int(ga_now))
                if do_log:
                    wandb_run.log(
                        {
                            "train_loss": loss.item() * accum,
                            "lr": lr_now,
                            "ga": accum,
                            "kgCO2": carbon.cumulative_kg,
                            "wall_time_h": (time.time() - wall_start) / 3600,
                            "step": global_step,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                global_step += 1

        # ----------------------- Validation -----------------------------
        val_batches = 2 if cfg.mode == "trial" else 100
        acc, conf = evaluate_accuracy(model, val_loader, tokenizer, val_batches)
        if do_log:
            wandb_run.log(
                {"val_accuracy": acc, **{f"conf_{k}": v for k, v in conf.items()}, "epoch": epoch},
                step=global_step,
            )

    wandb_run.summary.update(
        {
            "final_val_accuracy": acc,
            "final_kgCO2": carbon.cumulative_kg,
            "total_wall_time_h": (time.time() - wall_start) / 3600,
            **{f"conf_{k}": v for k, v in conf.items()},
        }
    )
    wandb_run.finish()
    return acc

################################################################################
#                               OPTUNA WRAPPER                                 #
################################################################################

def _apply_trial_params(cfg: DictConfig, trial: optuna.Trial):
    for p_name, spec in cfg.optuna.search_space.items():
        if spec.type == "loguniform":
            val = trial.suggest_float(p_name, spec.low, spec.high, log=True)
        elif spec.type == "uniform":
            val = trial.suggest_float(p_name, spec.low, spec.high, log=False)
        elif spec.type == "categorical":
            val = trial.suggest_categorical(p_name, spec.choices)
        else:
            raise ValueError(f"Unknown search space type {spec.type}")

        def _set(cfg_node, key_path, v):
            if len(key_path) == 1:
                cfg_node[key_path[0]] = v
                return
            _set(cfg_node[key_path[0]], key_path[1:], v)

        _set(cfg, p_name.split("."), val)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    if not cfg.get("run"):
        print("ERROR: Provide run=<run_id>", file=sys.stderr)
        sys.exit(1)
    run_yaml = to_absolute_path(f"config/runs/{cfg.run}.yaml")
    run_cfg = OmegaConf.load(run_yaml)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, run_cfg)
    cfg.run_id = cfg.get("run_id", cfg.run)
    OmegaConf.set_struct(cfg, True)

    # ------------ Mode adjustments ----------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(cfg.mode)

    print("================ Resolved Config ================")
    print(OmegaConf.to_yaml(cfg))

    # ------------------- Optuna -------------------
    if cfg.optuna.n_trials > 0:
        def objective(trial):
            trial_cfg = deepcopy(cfg)
            _apply_trial_params(trial_cfg, trial)
            trial_cfg.wandb.mode = "disabled"
            trial_cfg.training.epochs = max(1, int(math.ceil(cfg.training.epochs / 3)))
            acc = run_training(trial_cfg, do_log=False)
            return acc

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=int(cfg.optuna.n_trials),
            timeout=cfg.optuna.get("timeout_minutes", 480) * 60,
        )
        for k, v in study.best_params.items():
            ptr = cfg
            parts = k.split(".")
            for p in parts[:-1]:
                ptr = ptr[p]
            ptr[parts[-1]] = v
        print("[Optuna] Injected best params into config.")

    # ---------------- Final training --------------
    _ = run_training(cfg, do_log=True)

if __name__ == "__main__":
    main()