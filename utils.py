import numpy as np
import torch
from typing import Optional 

from functools import partial
from stable_pretraining import data as dt
from stable_pretraining import backbone
from lightning.pytorch.callbacks import Callback
from torch import nn
from .transforms import preprocess_actions, preprocess_pixels
from .jepa import JEPA
from .config import Args, NormType
from .module import ARPredictor, Embedder, MLP

def get_model(args: Args, action_dim: int, hooks: dict = {}):
    m = args.model_args
    encoder = backbone.utils.vit_hf(
        "tiny",
        patch_size=m.encoder_args.patch_size,
        image_size=args.env_args.image_size,
        pretrained=m.encoder_args.pretrained,
        use_mask_token=m.encoder_args.use_mask_token,
    )

    predictor = ARPredictor(
        num_frames=m.history_size,
        input_dim=m.embedding_dim,
        hidden_dim=m.embedding_dim,
        output_dim=m.embedding_dim,
        depth=m.predictor.depth,
        heads=m.predictor.heads,
        dim_head=m.predictor.dim_head,
        mlp_dim=m.predictor.mlp_dim,
        dropout=m.predictor.dropout,
        emb_dropout=m.predictor.emb_dropout,
    )

    # the action embedding conditions the predictor via AdaLN, so it must have embedding_dim
    action_encoder = Embedder(
        input_dim=args.env_args.frameskip * action_dim,
        emb_dim=m.embedding_dim,
    )

    norm_fn = nn.BatchNorm1d
    if m.projector_norm == NormType.LAYER_NORM:
        norm_fn = nn.LayerNorm
    elif m.projector_norm == NormType.GROUP_NORM:
        norm_fn = partial(nn.GroupNorm, m.projector_norm_groups)

    projector = MLP(
        input_dim=m.embedding_dim,
        output_dim=m.embedding_dim,
        hidden_dim=m.projector_hidden_dim,
        norm_fn=norm_fn,
        standardize_weights=m.renormalize_weights
    )

    prediction_projector = MLP(
        input_dim=m.embedding_dim,
        output_dim=m.embedding_dim,
        hidden_dim=m.projector_hidden_dim,
        norm_fn=norm_fn,
        standardize_weights=m.renormalize_weights
    )

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=prediction_projector,
        **hooks
    )

def to_device(batch, device, image_size):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    batch["pixels"] = preprocess_pixels(batch["pixels"], image_size)
    batch["action"] = preprocess_actions(batch["action"], 0.0)
    
    return batch

def autocast(device, enabled):
    return torch.autocast(
        device_type=torch.device(device).type, 
        dtype=torch.bfloat16, 
        enabled=enabled
    )

# def get_img_preprocessor(source: str, target: str, img_size: int = 224):
#     imagenet_stats = dt.dataset_stats.ImageNet
#     to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
#     resize = dt.transforms.Resize(img_size, source=source, target=target)
#     return dt.transforms.Compose(to_image, resize)

class Logger:
    def __init__(self, project_name, team, run_name: str, config: dict, log_to_wandb: bool = False):
        self.wandb = None
        if log_to_wandb:
            import wandb

            self.wandb = wandb.init(
                project=project_name,
                entity=team,
                name=run_name,
                config=config
            )

    def log(self, metrics: dict, step: int):
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def log_artifact(self, name: str, type: str, strpath: str):
        if self.wandb is not None:
            import wandb
            from pathlib import Path

            path = Path(strpath).resolve()
            self.wandb.log_artifact(artifact_or_path=str(path))
            
class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()

def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class EpisodeGroupedBatchSampler(torch.utils.data.Sampler):
    """Batches of `episodes_per_batch` random episodes x (batch_size / episodes_per_batch) windows each.

    With few episodes per batch, between-episode (static) variation cannot make the
    batch look Gaussian on its own, so SIGReg has to be satisfied by within-episode
    variance. Without this, PushCube embeddings collapse onto episode-static content.

    episodes: episode id of each item of the dataset being sampled.
    """

    def __init__(self, episodes, batch_size, episodes_per_batch, seed=0):
        assert batch_size % episodes_per_batch == 0
        self.per_ep = batch_size // episodes_per_batch
        self.episodes_per_batch = episodes_per_batch
        self.num_batches = len(episodes) // batch_size
        episodes = np.asarray(episodes)
        self.items = {e: np.nonzero(episodes == e)[0] for e in np.unique(episodes)}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        eps = list(self.items)
        for _ in range(self.num_batches):
            chosen = self.rng.choice(eps, self.episodes_per_batch, replace=False)
            yield np.concatenate([
                self.rng.choice(self.items[e], self.per_ep, replace=len(self.items[e]) < self.per_ep)
                for e in chosen
            ]).tolist()


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )

def _dataclass_from_dict(cls, d: dict):
    """Rebuild a (nested) config dataclass from its dataclasses.asdict() form."""
    from dataclasses import fields, is_dataclass

    kwargs = {}
    for f in fields(cls):
        if f.name in d:
            v = d[f.name]
            kwargs[f.name] = _dataclass_from_dict(f.type, v) if is_dataclass(f.type) and isinstance(v, dict) else v
    return cls(**kwargs)


def _concat_episodes(batch):
    # each item is one episode with (W, T, ...) windows; concat episodes along W
    return {k: torch.cat([b[k] for b in batch], dim=0) for k in batch[0]}


@torch.no_grad()
def recalibrate_bn_checkpoint(
    ckpt_path: str,
    dataset,
    episodes_per_batch: int | None = None,
    num_workers: int = 4,
    device: str = "cuda",
    out_path: str | None = None,
) -> str:
    """Recompute BatchNorm running statistics of a training checkpoint and save a copy.

    During training the running stats are an EMA (momentum 0.1) dominated by the last ~10 batches.
    This resets them and takes an equal-weight average (momentum=None) over one pass of `dataset`,
    batched like training (episodes_per_batch defaults to the checkpoint's). Only BatchNorm layers
    run in train mode, so dropout stays off and no weights change.

    dataset: a ManiSkillTrajectoryDataset (or a Subset of one), ideally the training episodes.
        Its action normalization is overwritten with the checkpoint's.
    Returns the path of the saved checkpoint (default: <ckpt>_recal.pt).
    """
    import os
    from .config import Args
    # from .train import get_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = _dataclass_from_dict(Args, ckpt["args"])

    base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    base.action_mean, base.action_std = ckpt["action_mean"], ckpt["action_std"]

    model = get_model(args, base.action_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise ValueError("checkpoint has no BatchNorm layers to recalibrate")
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative (equal-weight) average over all batches
        bn.train()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=episodes_per_batch or args.training_args.episodes_per_batch,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_concat_episodes,
    )
    ctx_len = args.model_args.history_size
    for batch in loader:
        batch = to_device(batch, device, args.env_args.image_size)
        with autocast(device, args.training_args.bf16):
            out = model.encode(batch)  # projector BN
            model.predict(out["emb"][:, :ctx_len], out["act_emb"][:, :ctx_len])  # pred_proj BN

    for bn, momentum in zip(bns, momenta):
        bn.momentum = momentum

    ckpt["model"] = model.state_dict()
    ckpt["bn_recalibration"] = dict(source=ckpt_path, num_batches=len(loader), episodes_per_batch=loader.batch_size)
    if out_path is None:
        root, ext = os.path.splitext(ckpt_path)
        out_path = f"{root}_recal{ext}"
    torch.save(ckpt, out_path)
    return out_path

