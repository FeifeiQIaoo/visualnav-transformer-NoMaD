#!/usr/bin/env python3
"""
offline_nav_corruption_eval.py

Offline evaluation for NoMaD-style visual navigation under observation corruption.

What this script does:
1. Loads an image trajectory from --traj-dir.
2. Samples a topological map from the trajectory.
3. Optionally corrupts all frames or only a ratio of frames.
4. Runs the NoMaD policy in offline replay mode.
5. Saves per-step statistics and an episode-level summary:
   - final closest node
   - whether reached goal
   - first success step
   - final node distance to goal
   - total action path length
   - total topomap progress
   - localization error
   - subgoal deviation
   - action variance
   - diffusion score statistics
"""

import os
import sys
import json
import yaml
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

PROJECT_ROOT = "/home/fqiao/Research/visualnav-transformer-main"
TRAIN_ROOT = os.path.join(PROJECT_ROOT, "train")

if TRAIN_ROOT not in sys.path:
    sys.path.insert(0, TRAIN_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import transform_images, load_model, to_numpy
from train.vint_train.training.train_utils import get_action


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_image_paths(traj_dir: str):
    traj_dir = Path(traj_dir)
    img_paths = sorted(traj_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if len(img_paths) == 0:
        img_paths = sorted(traj_dir.glob("*.png"), key=lambda p: int(p.stem))
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No .jpg or .png images found in {traj_dir}")
    return img_paths


def load_pil(img_path):
    return Image.open(img_path).convert("RGB")


def corrupt_pil_image(
    pil_img,
    corruption_type: str = "none",
    noise_std: float = 0.0,
    occlusion_ratio: float = 0.0,
    occlusion_fill: float = 0.0,
):
    img = np.array(pil_img).astype(np.float32) / 255.0

    if corruption_type in ["noise", "both"]:
        img = img + np.random.randn(*img.shape).astype(np.float32) * noise_std
        img = np.clip(img, 0.0, 1.0)

    if corruption_type in ["occlusion", "both"]:
        H, W, _ = img.shape
        ch = max(1, int(H * occlusion_ratio))
        cw = max(1, int(W * occlusion_ratio))
        top = (H - ch) // 2
        left = (W - cw) // 2

        fill_value = float(occlusion_fill)
        if fill_value > 1.0:
            fill_value = fill_value / 255.0
        fill_value = float(np.clip(fill_value, 0.0, 1.0))

        img[top:top + ch, left:left + cw, :] = fill_value

    return Image.fromarray((img * 255.0).clip(0, 255).astype(np.uint8))


def choose_corrupted_indices(
    num_frames: int,
    corruption_ratio: float,
    mode: str,
    seed: int,
):
    """
    mode:
      - none: corrupt no frames
      - all: corrupt all frames
      - random: corrupt corruption_ratio of frames randomly
      - periodic: corrupt approximately corruption_ratio of frames periodically
    """
    if mode == "none" or corruption_ratio <= 0:
        return set()

    if mode == "all":
        return set(range(num_frames))

    k = int(round(num_frames * corruption_ratio))
    k = max(0, min(num_frames, k))

    if mode == "random":
        rng = np.random.default_rng(seed)
        return set(rng.choice(num_frames, size=k, replace=False).tolist())

    if mode == "periodic":
        if k == 0:
            return set()
        stride = max(1, int(round(num_frames / k)))
        return set(range(0, num_frames, stride))

    raise ValueError(f"Unknown corrupt-frame-mode: {mode}")


def torch_score_from_epsilon(noise_pred, alpha_bar_t):
    """
    epsilon-prediction DDPM:
        score(x_t) ≈ -eps / sqrt(1 - alpha_bar_t)
    """
    if not torch.is_tensor(alpha_bar_t):
        alpha_bar_t = torch.tensor(
            alpha_bar_t,
            device=noise_pred.device,
            dtype=noise_pred.dtype,
        )
    denom = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-12))
    return -noise_pred / denom


def aggregate_diffpath_stats(all_scores_np, n_ddim_steps: int):
    """
    all_scores_np: [num_samples, num_denoising_steps, horizon, action_dim]
    """
    eps = all_scores_np
    reduce_axes = tuple(range(1, eps.ndim))

    score_l1 = np.sum(np.abs(eps), axis=reduce_axes)
    score_l2 = np.sum(eps ** 2, axis=reduce_axes)
    score_l3 = np.sum(np.abs(eps) ** 3, axis=reduce_axes)

    if eps.shape[1] >= 2:
        deps = np.diff(eps, axis=1) * n_ddim_steps
        diff_axes = tuple(range(1, deps.ndim))
        dscore_dt_l1 = np.sum(np.abs(deps), axis=diff_axes)
        dscore_dt_l2 = np.sum(deps ** 2, axis=diff_axes)
        dscore_dt_l3 = np.sum(np.abs(deps) ** 3, axis=diff_axes)
    else:
        z = np.zeros(eps.shape[0], dtype=np.float64)
        dscore_dt_l1 = dscore_dt_l2 = dscore_dt_l3 = z

    return {
        "score_l1_mean": float(np.mean(score_l1)),
        "score_l2_mean": float(np.mean(score_l2)),
        "score_l3_mean": float(np.mean(score_l3)),
        "dscore_dt_l1_mean": float(np.mean(dscore_dt_l1)),
        "dscore_dt_l2_mean": float(np.mean(dscore_dt_l2)),
        "dscore_dt_l3_mean": float(np.mean(dscore_dt_l3)),
    }


class RealNavModel:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device

        model_config_path = Path("deployment/config/models.yaml")
        with open(model_config_path, "r") as f:
            model_paths = yaml.safe_load(f)

        if model_name not in model_paths:
            raise ValueError(f"Model '{model_name}' not found in {model_config_path}")

        self.model_name = model_name
        self.model_entry = model_paths[model_name]

        config_path = (Path("deployment/src") / self.model_entry["config_path"]).resolve()
        ckpt_path = (Path("deployment/src") / self.model_entry["ckpt_path"]).resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        with open(config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        print(f"Loading model config from: {config_path}")
        print(f"Loading checkpoint from: {ckpt_path}")

        self.model = load_model(str(ckpt_path), self.model_params, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.context_size = self.model_params["context_size"]
        self.image_size = self.model_params["image_size"]
        self.model_type = self.model_params["model_type"]

        if self.model_type != "nomad":
            raise ValueError(f"This script supports NoMaD only, got {self.model_type}")

        self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
        self.len_traj_pred = self.model_params["len_traj_pred"]

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def predict(
        self,
        context_queue,
        topomap_pils,
        args,
        closest_node: int,
        goal_node: int,
    ):
        mask = torch.zeros(1).long().to(self.device)

        obs_images = transform_images(
            context_queue,
            self.image_size,
            center_crop=False,
        )
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(self.device)

        start = max(closest_node - args.radius, 0)
        end = min(closest_node + args.radius + 1, goal_node)

        goal_image = [
            transform_images(g_img, self.image_size, center_crop=False).to(self.device)
            for g_img in topomap_pils[start:end + 1]
        ]
        goal_image = torch.concat(goal_image, dim=0)

        obsgoal_cond = self.model(
            "vision_encoder",
            obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
            goal_img=goal_image,
            input_goal_mask=mask.repeat(len(goal_image)),
        )

        dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dists = to_numpy(dists.flatten())

        min_idx = int(np.argmin(dists))
        closest_node_new = min_idx + start

        sg_idx_local = min(
            min_idx + int(dists[min_idx] < args.close_threshold),
            len(obsgoal_cond) - 1,
        )
        subgoal_idx = start + sg_idx_local
        obs_cond = obsgoal_cond[sg_idx_local].unsqueeze(0)

        all_scores = []

        with torch.no_grad():
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

            naction = torch.randn(
                (args.num_samples, self.len_traj_pred, 2),
                device=self.device,
            )

            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond,
                )

                k_int = int(k.item()) if torch.is_tensor(k) else int(k)
                alpha_bar_t = self.noise_scheduler.alphas_cumprod[k_int].to(self.device)
                score = torch_score_from_epsilon(noise_pred, alpha_bar_t)
                all_scores.append(score.detach().cpu().numpy())

                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction,
                ).prev_sample

        naction = to_numpy(get_action(naction))  # [num_samples, horizon, 2]
        chosen_waypoint = naction[0][args.waypoint]

        if len(all_scores) > 0:
            all_scores_np = np.stack(all_scores, axis=0).transpose(1, 0, 2, 3)
            score_stats = aggregate_diffpath_stats(
                all_scores_np,
                n_ddim_steps=len(all_scores),
            )
        else:
            score_stats = {
                "score_l1_mean": 0.0,
                "score_l2_mean": 0.0,
                "score_l3_mean": 0.0,
                "dscore_dt_l1_mean": 0.0,
                "dscore_dt_l2_mean": 0.0,
                "dscore_dt_l3_mean": 0.0,
            }

        return {
            "closest_node": int(closest_node_new),
            "subgoal_idx": int(subgoal_idx),
            "waypoint": np.asarray(chosen_waypoint).tolist(),
            "sampled_actions": naction,
            "dists": dists.tolist(),
            "candidate_range": [int(start), int(end)],
            "action_sample_mean": naction.mean(axis=0).tolist(),
            "action_sample_var": naction.var(axis=0).tolist(),
            **score_stats,
        }


def expected_topomap_node(frame_idx: int, topomap_every: int, num_topomap_nodes: int):
    return min(frame_idx // topomap_every, num_topomap_nodes - 1)


def main(args):
    set_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    img_paths = load_image_paths(args.traj_dir)
    raw_traj_pils = [load_pil(p) for p in img_paths]

    corrupted_indices = choose_corrupted_indices(
        num_frames=len(raw_traj_pils),
        corruption_ratio=args.corruption_ratio,
        mode=args.corrupt_frame_mode,
        seed=args.seed,
    )

    traj_pils = []
    for i, img in enumerate(raw_traj_pils):
        if i in corrupted_indices:
            traj_pils.append(
                corrupt_pil_image(
                    img,
                    corruption_type=args.obs_corruption,
                    noise_std=args.obs_noise_std,
                    occlusion_ratio=args.obs_occlusion_ratio,
                    occlusion_fill=args.occlusion_fill,
                )
            )
        else:
            traj_pils.append(img)

    topomap_indices = list(range(0, len(traj_pils), args.topomap_every))
    if len(topomap_indices) == 0:
        raise ValueError("Topomap sampling produced zero nodes.")

    # You can choose whether the topomap itself is corrupted.
    # For observation-corruption experiments, keep topomap clean by default.
    if args.corrupt_topomap:
        topomap_source = traj_pils
    else:
        topomap_source = raw_traj_pils

    topomap_pils = [topomap_source[i] for i in topomap_indices]

    print(f"Loaded {len(img_paths)} trajectory frames")
    print(f"Corrupted frames: {len(corrupted_indices)} / {len(img_paths)}")
    print(f"Topomap nodes: {len(topomap_pils)}")

    model = RealNavModel(args.model, device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    context = model.context_size + 1
    max_steps = min(args.max_steps, len(img_paths))

    closest_node = args.start_closest_node
    goal_node = len(topomap_pils) - 1 if args.goal_node == -1 else args.goal_node

    if not (0 <= goal_node < len(topomap_pils)):
        raise ValueError(f"Invalid goal_node={goal_node}; topomap has {len(topomap_pils)} nodes.")

    per_step = []
    reached_goal = False
    first_success_step = None

    previous_closest_node = closest_node
    total_topomap_progress = 0.0
    total_action_path_length = 0.0

    localization_errors = []
    subgoal_errors = []
    action_var_values = []

    score_l1_values = []
    score_l2_values = []
    score_l3_values = []
    dscore_dt_l1_values = []
    dscore_dt_l2_values = []
    dscore_dt_l3_values = []

    for t in range(context - 1, max_steps):
        context_queue = traj_pils[t - context + 1: t + 1]

        pred = model.predict(
            context_queue=context_queue,
            topomap_pils=topomap_pils,
            args=args,
            closest_node=closest_node,
            goal_node=goal_node,
        )

        closest_node = pred["closest_node"]

        expected_node = expected_topomap_node(
            frame_idx=t,
            topomap_every=args.topomap_every,
            num_topomap_nodes=len(topomap_pils),
        )
        localization_error = abs(closest_node - expected_node)
        subgoal_error = abs(pred["subgoal_idx"] - expected_node)

        waypoint_xy = np.asarray(pred["waypoint"][:2], dtype=np.float64)
        action_step_length = float(np.linalg.norm(waypoint_xy))

        sampled_actions = np.asarray(pred["sampled_actions"])
        action_var_mean = float(np.mean(np.var(sampled_actions, axis=0)))

        total_topomap_progress += abs(closest_node - previous_closest_node)
        total_action_path_length += action_step_length
        previous_closest_node = closest_node

        localization_errors.append(localization_error)
        subgoal_errors.append(subgoal_error)
        action_var_values.append(action_var_mean)

        score_l1_values.append(pred["score_l1_mean"])
        score_l2_values.append(pred["score_l2_mean"])
        score_l3_values.append(pred["score_l3_mean"])
        dscore_dt_l1_values.append(pred["dscore_dt_l1_mean"])
        dscore_dt_l2_values.append(pred["dscore_dt_l2_mean"])
        dscore_dt_l3_values.append(pred["dscore_dt_l3_mean"])

        step_record = {
            "step": int(t),
            "closest_node": int(closest_node),
            "expected_node": int(expected_node),
            "goal_node": int(goal_node),
            "reached_goal": bool(closest_node == goal_node),
            "subgoal_idx": int(pred["subgoal_idx"]),
            "candidate_range": pred["candidate_range"],
            "localization_error": int(localization_error),
            "subgoal_error": int(subgoal_error),
            "waypoint": pred["waypoint"],
            "action_step_length": action_step_length,
            "action_var_mean": action_var_mean,
            "dists": pred["dists"],
            "score_l1_mean": pred["score_l1_mean"],
            "score_l2_mean": pred["score_l2_mean"],
            "score_l3_mean": pred["score_l3_mean"],
            "dscore_dt_l1_mean": pred["dscore_dt_l1_mean"],
            "dscore_dt_l2_mean": pred["dscore_dt_l2_mean"],
            "dscore_dt_l3_mean": pred["dscore_dt_l3_mean"],
            "is_corrupted_frame": bool(t in corrupted_indices),
        }

        per_step.append(step_record)
        save_json(step_record, save_dir / f"stats_step_{t:04d}.json")
        np.save(save_dir / f"actions_step_{t:04d}.npy", sampled_actions)

        if closest_node == goal_node and not reached_goal:
            reached_goal = True
            first_success_step = int(t)
            if args.stop_on_success:
                break

        print(
            f"[step {t}] closest={closest_node} "
            f"expected={expected_node} subgoal={pred['subgoal_idx']} "
            f"goal={goal_node} reached={closest_node == goal_node}"
        )

    final_closest_node = int(closest_node)
    final_node_distance_to_goal = int(abs(goal_node - final_closest_node))
    num_eval_steps = len(per_step)

    summary = {
        "traj_dir": args.traj_dir,
        "save_dir": args.save_dir,
        "model": args.model,

        "num_frames": len(img_paths),
        "num_topomap_nodes": len(topomap_pils),
        "topomap_every": args.topomap_every,

        "start_closest_node": args.start_closest_node,
        "goal_node": int(goal_node),
        "final_closest_node": final_closest_node,
        "final_node_distance_to_goal": final_node_distance_to_goal,

        "reached_goal": bool(reached_goal),
        "first_success_step": first_success_step,
        "num_eval_steps": int(num_eval_steps),

        "total_action_path_length": float(total_action_path_length),
        "total_topomap_progress": float(total_topomap_progress),
        "progress_ratio": float(final_closest_node / max(goal_node, 1)),

        "mean_localization_error": float(np.mean(localization_errors)) if localization_errors else None,
        "mean_subgoal_error": float(np.mean(subgoal_errors)) if subgoal_errors else None,
        "mean_action_var": float(np.mean(action_var_values)) if action_var_values else None,

        "mean_score_l1": float(np.mean(score_l1_values)) if score_l1_values else None,
        "mean_score_l2": float(np.mean(score_l2_values)) if score_l2_values else None,
        "mean_score_l3": float(np.mean(score_l3_values)) if score_l3_values else None,
        "mean_dscore_dt_l1": float(np.mean(dscore_dt_l1_values)) if dscore_dt_l1_values else None,
        "mean_dscore_dt_l2": float(np.mean(dscore_dt_l2_values)) if dscore_dt_l2_values else None,
        "mean_dscore_dt_l3": float(np.mean(dscore_dt_l3_values)) if dscore_dt_l3_values else None,

        "obs_corruption": args.obs_corruption,
        "obs_noise_std": args.obs_noise_std,
        "obs_occlusion_ratio": args.obs_occlusion_ratio,
        "occlusion_fill": args.occlusion_fill,

        "corrupt_frame_mode": args.corrupt_frame_mode,
        "corruption_ratio_requested": args.corruption_ratio,
        "corruption_ratio_actual": float(len(corrupted_indices) / max(len(img_paths), 1)),
        "num_corrupted_frames": len(corrupted_indices),
        "corrupt_topomap": args.corrupt_topomap,
        "seed": args.seed,
    }

    save_json(per_step, save_dir / "per_step.json")
    save_json(summary, save_dir / "summary.json")

    print("\n===== Episode Summary =====")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--traj-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--topomap-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)

    parser.add_argument("--goal-node", type=int, default=-1)
    parser.add_argument("--start-closest-node", type=int, default=0)
    parser.add_argument("--close-threshold", type=int, default=3)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--waypoint", type=int, default=2)

    parser.add_argument(
        "--obs-corruption",
        type=str,
        default="none",
        choices=["none", "noise", "occlusion", "both"],
    )
    parser.add_argument("--obs-noise-std", type=float, default=0.0)
    parser.add_argument("--obs-occlusion-ratio", type=float, default=0.0)
    parser.add_argument("--occlusion-fill", type=float, default=0.0)

    parser.add_argument(
        "--corrupt-frame-mode",
        type=str,
        default="all",
        choices=["none", "all", "random", "periodic"],
        help="Which observation frames to corrupt.",
    )
    parser.add_argument(
        "--corruption-ratio",
        type=float,
        default=1.0,
        help="Ratio of frames to corrupt for random/periodic mode. Ignored by all/none.",
    )
    parser.add_argument(
        "--corrupt-topomap",
        action="store_true",
        help="If set, the topomap is also sampled from corrupted frames. Default: topomap stays clean.",
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop the episode once closest_node == goal_node.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()
    main(args)
