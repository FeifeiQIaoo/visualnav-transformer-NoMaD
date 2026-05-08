import os
import json
import yaml
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from utils import transform_images, load_model, to_numpy
import os
import sys

PROJECT_ROOT = "/home/fqiao/Research/visualnav-transformer-main"
TRAIN_ROOT = os.path.join(PROJECT_ROOT, "train")

if TRAIN_ROOT not in sys.path:
    sys.path.insert(0, TRAIN_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from train.vint_train.training.train_utils import get_action


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_samples(x):
    """
    x: [N, ...] -> [N, D]
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x.reshape(x.shape[0], -1)


def covariance_eigvals(x):
    """
    x: [N, D] numpy
    return eigvals of covariance matrix, sorted descending
    """
    x = x.astype(np.float64)

    if x.shape[0] < 2:
        return np.zeros(x.shape[1], dtype=np.float64)

    x_centered = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x_centered, rowvar=False)

    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    return eigvals



def init_diffpath_stat_storage():
    return {
        "score_l1": [],
        "score_l2": [],
        "score_l3": [],
        "dscore_dt_l1": [],
        "dscore_dt_l2": [],
        "dscore_dt_l3": [],
    }


def append_diffpath_statistics(storage, eps: np.ndarray, n_ddim_steps: int):
    """
    eps: [B, T, ...]
    对每个样本，把整条 diffusion path 上的 6 个统计量聚合出来：
        sum_t <eps_t>_1
        sum_t <eps_t>_2
        sum_t <eps_t>_3
        sum_t <d eps_t / dt>_1
        sum_t <d eps_t / dt>_2
        sum_t <d eps_t / dt>_3

    这里定义：
        <x>_1 = sum(abs(x))
        <x>_2 = sum(x^2)
        <x>_3 = sum(abs(x)^3)
    """
    if eps.ndim < 3:
        raise ValueError(f"eps must have shape [B, T, ...], got {eps.shape}")

    reduce_axes = tuple(range(1, eps.ndim))

    storage["score_l1"].extend(np.sum(np.abs(eps), axis=reduce_axes).tolist())
    storage["score_l2"].extend(np.sum(eps ** 2, axis=reduce_axes).tolist())
    storage["score_l3"].extend(np.sum(np.abs(eps) ** 3, axis=reduce_axes).tolist())

    if eps.shape[1] >= 2:
        eps_diff = np.diff(eps, axis=1) * n_ddim_steps
        diff_reduce_axes = tuple(range(1, eps_diff.ndim))

        storage["dscore_dt_l1"].extend(np.sum(np.abs(eps_diff), axis=diff_reduce_axes).tolist())
        storage["dscore_dt_l2"].extend(np.sum(eps_diff ** 2, axis=diff_reduce_axes).tolist())
        storage["dscore_dt_l3"].extend(np.sum(np.abs(eps_diff) ** 3, axis=diff_reduce_axes).tolist())
    else:
        batch_size = eps.shape[0]
        zeros = [0.0] * batch_size
        storage["dscore_dt_l1"].extend(zeros)
        storage["dscore_dt_l2"].extend(zeros)
        storage["dscore_dt_l3"].extend(zeros)


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


def save_json(obj, path):
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


def sample_topomap(img_paths, every: int):
    if every <= 0:
        raise ValueError("topomap_every must be > 0")
    sampled = img_paths[::every]
    if len(sampled) == 0:
        raise ValueError("Topomap sampling produced zero nodes.")
    return sampled


def load_pil(img_path):
    return Image.open(img_path).convert("RGB")


def corrupt_pil_image(
    pil_img,
    corruption_type: str = "none",
    noise_std: float = 0.0,
    occlusion_ratio: float = 0.0,
    occlusion_fill: float = 0.0,
):
    """
    对单张 PIL 图像做破坏，返回新的 PIL 图像
    """
    img = np.array(pil_img).astype(np.float32) / 255.0  # [H, W, 3]

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
        fill_value = np.clip(fill_value, 0.0, 1.0)
        img[top:top + ch, left:left + cw, :] = fill_value

    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


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

        # models.yaml 路径是相对 deployment/src/navigate.py 写的
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

        self.model = load_model(
            str(ckpt_path),
            self.model_params,
            self.device,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.context_size = self.model_params["context_size"]
        self.image_size = self.model_params["image_size"]
        self.model_type = self.model_params["model_type"]

        if self.model_type != "nomad":
            raise ValueError(
                f"This offline script currently supports NoMaD only, "
                f"but got model_type={self.model_type}"
            )

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

        # observation branch
        obs_images = transform_images(
            context_queue,
            self.image_size,
            center_crop=False,
        )
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        obs_images = obs_images.to(self.device)

        # 注意：这里不再重复对 obs_images 做 corruption
        # 因为 context_queue 已经来自被破坏后的 trajectory

        # candidate subgoals from corrupted topomap
        start = max(closest_node - args.radius, 0)
        end = min(closest_node + args.radius + 1, goal_node)

        goal_image = [
            transform_images(g_img, self.image_size, center_crop=False).to(self.device)
            for g_img in topomap_pils[start:end + 1]
        ]
        goal_image = torch.concat(goal_image, dim=0)

        # localization
        obsgoal_cond = self.model(
            "vision_encoder",
            obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
            goal_img=goal_image,
            input_goal_mask=mask.repeat(len(goal_image)),
        )

        dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dists = to_numpy(dists.flatten())

        min_idx = int(np.argmin(dists))
        closest_node = min_idx + start

        sg_idx_local = min(
            min_idx + int(dists[min_idx] < args.close_threshold),
            len(obsgoal_cond) - 1,
        )
        sg_idx_global = start + sg_idx_local
        obs_cond = obsgoal_cond[sg_idx_local].unsqueeze(0)

        # diffusion action sampling + score statistics
        score_per_timestep = []
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
                score = torch_score_from_epsilon(noise_pred, alpha_bar_t)  # [N, horizon, 2]

                score_np = score.detach().cpu().numpy()
                all_scores.append(score_np)

                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction,
                ).prev_sample

        naction = to_numpy(get_action(naction))   # [num_samples, horizon, 2]
        chosen_waypoint = naction[0][args.waypoint]

        action_sample_mean = naction.mean(axis=0)   # [horizon, 2]
        action_sample_var = naction.var(axis=0)     # [horizon, 2]

        if len(all_scores) > 0:
            # all_scores: list of [N, horizon, 2]
            # stack -> [Td, N, horizon, 2], transpose -> [N, Td, horizon, 2]
            all_scores_np = np.stack(all_scores, axis=0).transpose(1, 0, 2, 3)

            diffpath_storage = init_diffpath_stat_storage()
            append_diffpath_statistics(
                diffpath_storage,
                all_scores_np,
                n_ddim_steps=len(all_scores),
            )

            score_per_timestep = [
                {
                    "score_l1": float(diffpath_storage["score_l1"][i]),
                    "score_l2": float(diffpath_storage["score_l2"][i]),
                    "score_l3": float(diffpath_storage["score_l3"][i]),
                    "dscore_dt_l1": float(diffpath_storage["dscore_dt_l1"][i]),
                    "dscore_dt_l2": float(diffpath_storage["dscore_dt_l2"][i]),
                    "dscore_dt_l3": float(diffpath_storage["dscore_dt_l3"][i]),
                }
                for i in range(all_scores_np.shape[0])
            ]

            overall_score_stats = {
                "score_l1_mean": float(np.mean(diffpath_storage["score_l1"])),
                "score_l2_mean": float(np.mean(diffpath_storage["score_l2"])),
                "score_l3_mean": float(np.mean(diffpath_storage["score_l3"])),
                "dscore_dt_l1_mean": float(np.mean(diffpath_storage["dscore_dt_l1"])),
                "dscore_dt_l2_mean": float(np.mean(diffpath_storage["dscore_dt_l2"])),
                "dscore_dt_l3_mean": float(np.mean(diffpath_storage["dscore_dt_l3"])),
            }
        else:
            overall_score_stats = {
                "score_l1_mean": 0.0,
                "score_l2_mean": 0.0,
                "score_l3_mean": 0.0,
                "dscore_dt_l1_mean": 0.0,
                "dscore_dt_l2_mean": 0.0,
                "dscore_dt_l3_mean": 0.0,
            }

        return {
            "closest_node": int(closest_node),
            "subgoal_idx": int(sg_idx_global),
            "sampled_actions": naction,   # [num_samples, horizon, 2]
            "waypoint": chosen_waypoint,
            "dists": dists.tolist(),
            "start_node": int(start),
            "end_node": int(end),

            "action_sample_mean": action_sample_mean.tolist(),
            "action_sample_var": action_sample_var.tolist(),

            "score_l1_mean": overall_score_stats["score_l1_mean"],
            "score_l2_mean": overall_score_stats["score_l2_mean"],
            "score_l3_mean": overall_score_stats["score_l3_mean"],
            "dscore_dt_l1_mean": overall_score_stats["dscore_dt_l1_mean"],
            "dscore_dt_l2_mean": overall_score_stats["dscore_dt_l2_mean"],
            "dscore_dt_l3_mean": overall_score_stats["dscore_dt_l3_mean"],

            "score_per_sample": score_per_timestep,
        }


def main(args):
    set_seed(args.seed)

    # 先用 CPU 跑通，避免你之前的 CUDA kernel image 报错
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        if args.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print(f"Using device: {device}")

    img_paths = load_image_paths(args.traj_dir)

    # 先生成整条“被破坏后的轨迹”
    corrupted_traj_pils = [
        corrupt_pil_image(
            load_pil(p),
            corruption_type=args.obs_corruption,
            noise_std=args.obs_noise_std,
            occlusion_ratio=args.obs_occlusion_ratio,
            occlusion_fill=args.occlusion_fill,
        )
        for p in img_paths
    ]

    # 从被破坏后的轨迹里按间隔采样 topomap / goal 节点
    topomap_indices = list(range(0, len(corrupted_traj_pils), args.topomap_every))
    if len(topomap_indices) == 0:
        raise ValueError("Topomap sampling produced zero nodes.")

    topomap_pils = [corrupted_traj_pils[i] for i in topomap_indices]

    print(f"Loaded {len(img_paths)} trajectory frames")
    print(f"Topomap nodes: {len(topomap_pils)}")

    model = RealNavModel(args.model, device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []

    context = model.context_size + 1
    max_steps = min(args.max_steps, len(img_paths))

    closest_node = args.start_closest_node
    goal_node = len(topomap_pils) - 1 if args.goal_node == -1 else args.goal_node

    if not (-1 <= args.goal_node < len(topomap_pils)):
        raise ValueError(
            f"Invalid goal_node={args.goal_node}. "
            f"Topomap has {len(topomap_pils)} nodes."
        )

    for t in range(context - 1, max_steps):
        context_queue = corrupted_traj_pils[t - context + 1: t + 1]

        pred = model.predict(
            context_queue=context_queue,
            topomap_pils=topomap_pils,
            args=args,
            closest_node=closest_node,
            goal_node=goal_node,
        )

        closest_node = pred["closest_node"]

        sampled_actions = pred["sampled_actions"]
        mean_action = sampled_actions.mean(axis=0)
        std_action = sampled_actions.std(axis=0)

        result = {
            "step": t,
            "closest_node": int(pred["closest_node"]),
            "subgoal_idx": int(pred["subgoal_idx"]),
            "waypoint": np.asarray(pred["waypoint"]).tolist(),
            "sampled_actions_shape": list(sampled_actions.shape),
            "mean_action": mean_action.tolist(),
            "std_action": std_action.tolist(),
            "dists": pred["dists"],
            "candidate_range": [pred["start_node"], pred["end_node"]],
            "obs_corruption": args.obs_corruption,
            "obs_noise_std": args.obs_noise_std,
            "obs_occlusion_ratio": args.obs_occlusion_ratio,
        }
        results.append(result)

        detailed_step = {
            "step": t,
            "closest_node": int(pred["closest_node"]),
            "subgoal_idx": int(pred["subgoal_idx"]),
            "candidate_range": [pred["start_node"], pred["end_node"]],
            "obs_corruption": args.obs_corruption,
            "obs_noise_std": args.obs_noise_std,
            "obs_occlusion_ratio": args.obs_occlusion_ratio,

            "action_sample_mean": pred["action_sample_mean"],
            "action_sample_var": pred["action_sample_var"],

            "score_l1_mean": pred["score_l1_mean"],
            "score_l2_mean": pred["score_l2_mean"],
            "score_l3_mean": pred["score_l3_mean"],
            "dscore_dt_l1_mean": pred["dscore_dt_l1_mean"],
            "dscore_dt_l2_mean": pred["dscore_dt_l2_mean"],
            "dscore_dt_l3_mean": pred["dscore_dt_l3_mean"],

            "score_per_sample": pred["score_per_sample"],
        }

        save_json(detailed_step, save_dir / f"stats_step_{t:04d}.json")

        print(
            f"[step {t}] "
            f"closest={result['closest_node']} "
            f"subgoal={result['subgoal_idx']} "
            f"waypoint={result['waypoint']}"
        )

        np.save(save_dir / f"actions_step_{t:04d}.npy", sampled_actions)

    with open(save_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {save_dir}")


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

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()
    main(args)