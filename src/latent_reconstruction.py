"""
Train a decoder that reconstructs images from frozen LeWM encoder latents.

Run from examples/baselines:
    python -m lewm.latent_reconstruction --ckpt-path lewm/runs/lewm_bn_200ep/latest.pt
"""
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Optional

import dotenv
import torch
import torch.nn.functional as F
import tyro
from torch import nn
from torch.utils.data import DataLoader, random_split

from .config import Args, WandBConfig
from .dataset import ManiSkillTrajectoryDataset
from .utils import Logger, _dataclass_from_dict, get_model, preprocess_pixels


@dataclass
class ReconArgs:
    ckpt_path: str = "lewm/runs/lewm_bn_200ep/latest.pt"
    """LeWM checkpoint whose (frozen) encoder produces the latents"""
    demo_dataset_path: str = str(
        Path.home() / ".maniskill/demos/PushCube-v1/cam_front48/trajectory.rgb.pd_joint_pos.physx_cpu.h5"
    )
    load_count: Optional[int] = None
    """trajectories to load (None loads all, ~14GB for PushCube front48)"""
    num_windows: int = 64
    """frames sampled per episode per epoch"""
    episodes_per_batch: int = 4
    """batch size is episodes_per_batch * num_windows frames"""
    n_epochs: int = 50
    max_steps_per_epoch: Optional[int] = None
    lr: float = 1e-4
    n_layers: int = 6
    n_heads: int = 16
    patch_size: int = 14
    seed: int = 0
    device: Optional[str] = None
    out_dir: str = "lewm/runs/latent_reconstructor"
    """decoder checkpoint is written to <out_dir>/latest.pt every epoch"""
    wandb_config: WandBConfig = field(default_factory=WandBConfig)


def make_transformer(d_model=192, nhead=16, nlayers=6):
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(encoder_layer=layer, num_layers=nlayers)


class LatentReconstructor(nn.Module):
    """(B, D) latent -> (B, 3, S, S) image in [0, 1] space (unclamped)."""

    def __init__(self, d_model=192, n_heads=16, n_layers=6, image_size=224, patch_size=14):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.n_tokens = self.grid**2

        # learned per-patch positions so the tokens are distinguishable
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)
        self.transformer = make_transformer(d_model=d_model, nhead=n_heads, nlayers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, patch_size * patch_size * 3)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.shape[0]
        x = latent.view(B, 1, -1) + self.pos_emb  # (B, N, D)
        x = self.projector(self.norm(self.transformer(x)))  # (B, N, p*p*3)
        g, p = self.grid, self.patch_size
        x = x.view(B, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4)  # (B, 3, g, p, g, p)
        return x.reshape(B, 3, self.image_size, self.image_size)


def target_pixels(pixels: torch.Tensor, image_size: int) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, S, S) float in [0, 1]."""
    x = pixels.permute(0, 3, 1, 2).float().div_(255)
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", antialias=True, align_corners=False)
    return x


def run_batch(item, jepa, model, image_size, device):
    # item["pixels"]: (B, W, 1, H, W, 3) -> treat every window as an independent frame
    frames = item["pixels"].to(device).flatten(0, 2)  # (B*W, H, W, 3)
    with torch.no_grad():
        latent = jepa.encode(dict(pixels=preprocess_pixels(frames[:, None], image_size)))["emb"][:, 0]
    target = target_pixels(frames, image_size)
    output = model(latent)
    return F.mse_loss(output, target), output, target


def main(r: ReconArgs):
    torch.manual_seed(r.seed)
    ckpt = torch.load(r.ckpt_path, map_location="cpu", weights_only=False)
    args = _dataclass_from_dict(Args, ckpt["args"])
    image_size = args.env_args.image_size
    device = r.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ManiSkillTrajectoryDataset(
        r.demo_dataset_path,
        num_steps=1,
        num_windows=r.num_windows,
        load_count=r.load_count,
    )

    jepa = get_model(args, dataset.action_dim)
    jepa.load_state_dict(ckpt["model"])
    jepa.to(device).eval().requires_grad_(False)

    train_set, val_set = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(r.seed))
    train_loader = DataLoader(train_set, batch_size=r.episodes_per_batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=r.episodes_per_batch, shuffle=False)

    model = LatentReconstructor(
        d_model=args.model_args.embedding_dim,
        n_heads=r.n_heads,
        n_layers=r.n_layers,
        image_size=image_size,
        patch_size=r.patch_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=r.lr)

    logger = Logger(
        r.wandb_config.wandb_project_name,
        r.wandb_config.wandb_team,
        "latent_reconstructor",
        config=asdict(r),
        log_to_wandb=r.wandb_config.enable,
    )

    out_dir = Path(r.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_step, train_loss = 0, float("nan")
    for epoch in range(r.n_epochs):
        model.train()
        for i, item in enumerate(train_loader):
            if r.max_steps_per_epoch is not None and i >= r.max_steps_per_epoch:
                break
            loss, _, _ = run_batch(item, jepa, model, image_size, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            logger.log({"train/loss": train_loss}, step=global_step)
            global_step += 1

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i, item in enumerate(val_loader):
                loss, output, target = run_batch(item, jepa, model, image_size, device)
                total += loss.item() * len(output)
                n += len(output)
                if i == 0:  # first 8 frames: reconstruction above ground truth
                    k = min(8, len(output))
                    grid = torch.cat([output[:k].clamp(0, 1), target[:k]], dim=-2)  # stack vertically
                    grid = torch.cat(list(grid), dim=-1)  # side by side
                    sample = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        val_loss = total / max(n, 1)
        print(f"epoch {epoch}: train/loss {train_loss:.4f}  val/loss {val_loss:.4f}")
        metrics = {"val/loss": val_loss, "epoch": epoch}
        if logger.wandb is not None and n > 0:
            import wandb

            metrics["val/reconstruction"] = wandb.Image(sample, caption="top: reconstruction, bottom: target")
        logger.log(metrics, step=global_step)
        torch.save(
            {"model": model.state_dict(), "recon_args": asdict(r), "epoch": epoch, "val_loss": val_loss},
            out_dir / "latest.pt",
        )


if __name__ == "__main__":
    dotenv.load_dotenv()
    main(tyro.cli(ReconArgs))
