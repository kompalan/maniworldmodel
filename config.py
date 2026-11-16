from dataclasses import dataclass, field
from typing import Optional
from enum import StrEnum

@dataclass
class WandBConfig:
    wandb_project_name: str = "LeWorldModel-ManiSkill"
    """the wandb project name"""

    wandb_team: Optional[str] = "fall2026-ml"#"anuragkompalli"
    """the wandb team name"""

    enable: bool = False

class OptimizerType(StrEnum):
    ADAMW = "AdamW"
    SGD = "SGD"


class LRSchedulerType(StrEnum):
    LINEAR_WARMUP_COSINE_ANNEALING = "LinearWarmupCosineAnnealingLR"


class NormType(StrEnum):
    BATCH_NORM = "BatchNorm"
    LAYER_NORM = "LayerNorm"
    GROUP_NORM = "GroupNorm"


@dataclass
class OptimizerArgs:
    type: OptimizerType = OptimizerType.ADAMW
    """Optimizer type (AdamW or SGD)"""

    learning_rate: float = 5e-5
    """Learning Rate"""

    weight_decay: float = 1e-3
    """Weight Decay"""


@dataclass
class TrainingArgs:
    episodes_per_batch: int = 1
    """Distinct episodes to include in each batch.
    Too many episodes per batch lets between-episode variation satisfy SIGReg on its own,
    which makes embeddings collapse onto episode-static content (see EpisodeGroupedBatchSampler in utils.py)"""

    batch_size: int = 64
    """Number of windows per batch. Must be divisible by episodes_per_batch"""

    max_epochs: int = 100
    """Max epochs to run (one epoch = one batch per training episode)"""

    max_train_steps_per_epoch: Optional[int] = None
    """Cap on train steps per epoch (for smoke tests). None means a full epoch"""

    gradient_clip_val: float = 1.0
    """The argument to clip_grad_norm_"""

    warmup_steps: int = 200
    """LR warmup steps"""

    bf16: bool = True
    """Use bf16 autocast"""

    num_workers: int = 4

    pin_memory: bool = True

    log_every: int = 50
    """Log train losses every N steps"""

    optimizer_args: OptimizerArgs = field(default_factory=OptimizerArgs)

    lr_scheduler_type: LRSchedulerType = LRSchedulerType.LINEAR_WARMUP_COSINE_ANNEALING


@dataclass
class EnvArgs:
    env_id: str = "PushCube-v1"
    """the environment id. See https://maniskill.readthedocs.io/en/latest/tasks/index.html"""

    demo_dataset_path: str = "/home/anurag/.maniskill/demos/PushCube-v1/cam_front48/trajectory.rgb.pd_joint_pos.physx_cpu.h5"
    """the path to demonstrations. Must be in HDF5 format"""

    max_trajectories: Optional[int] = None
    """the maximum number of trajectories to load"""

    image_size: int = 224
    """images are resized to this before the encoder (must be divisible by the patch size)"""

    frameskip: int = 5
    """How many frames to merge into 1.
    Example: a frameskip of 5 merges 5 8-dim actions into one 40-dim action.
    Observations are then t, t+5, t+10 etc."""

    train_split: float = 0.8
    """portion of demo data to assign to training"""


@dataclass
class PushCubeEnv(EnvArgs):
    env_id: str = "PushCube-v1"
    demo_dataset_path: str = "/home/anurag/.maniskill/demos/PushCube-v1/cam_front48/trajectory.rgb.pd_joint_pos.physx_cpu.h5"


@dataclass
class EncoderArgs:
    patch_size: int = 14

    pretrained: bool = False

    use_mask_token: bool = False


@dataclass
class PredictorArgs:
    depth: int = 6
    """Number of transformer blocks"""

    heads: int = 16
    """Number of attention heads"""

    dim_head: int = 64
    """Dimension of each head"""

    mlp_dim: int = 2048
    """Hidden dim of MLP"""

    dropout: float = 0.1
    """Dropout"""

    emb_dropout: float = 0.0

    num_preds: int = 1
    """How many steps ahead the target is"""


@dataclass
class LossArgs:
    sigreg_weight: float = 0.09

    sigreg_knots: int = 17

    sigreg_num_proj: int = 1024


@dataclass
class ModelArgs:
    encoder_args: EncoderArgs = field(default_factory=EncoderArgs)

    predictor: PredictorArgs = field(default_factory=PredictorArgs)

    embedding_dim: int = 192
    """must match the ViT-tiny hidden size"""

    projector_hidden_dim: int = 2048

    projector_norm: NormType = NormType.BATCH_NORM
    """hidden-layer norm of the projector and prediction projector.
    BatchNorm with few episodes per batch trains on within-episode statistics but evaluates on
    running (cross-episode) ones; LayerNorm is per-sample, so train and eval match"""

    projector_norm_groups: int = 32
    """number of groups when projector_norm is GroupNorm (must divide projector_hidden_dim)"""

    history_size: int = 3

    renormalize_weights: bool = False

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""

    output_dir: str = "runs"
    """checkpoints go to <output_dir>/<exp_name>"""

    device: str = "cuda"
    """the name of the torch device"""

    seed: int = 3072
    """random seed"""

    wandb_config: WandBConfig = field(default_factory=WandBConfig)
    """the wandb config. None means don't track with wandb"""

    env_args: EnvArgs = field(default_factory=PushCubeEnv)
    """the environment arguments. Make a new one for your env"""

    training_args: TrainingArgs = field(default_factory=TrainingArgs)
    """the training args. Handles things like batch size, episodes per batch etc."""

    model_args: ModelArgs = field(default_factory=ModelArgs)

    loss_args: LossArgs = field(default_factory=LossArgs)

@dataclass
class EvalArgs:
    exp_name: str
    """The name for the experiment"""

    ckpt_path: str
    """Path to a checkpoint saved by lewm.train (e.g. runs/<exp_name>/latest.pt)"""
    num_episodes: int = 1
    ep_start: int = 0
    
    seed: int = 0
    device: str | None = None
    """Override the device stored in the checkpoint's args"""
    video_dir: str | None = "videos"
    max_episode_steps: int = 100
    """env-step budget per episode (includes the seeded ground-truth prefix)"""
    plan_horizon: int = 13
    """action chunks (of frameskip env steps) planned per CEM call"""
    plan_receding_horizon: int = 1
    """planned chunks executed before replanning"""
    cem_num_samples: int = 300
    cem_topk: int = 30
    cem_iters: int = 30
    cem_var_scale: float = 1.0
    """initial std of the sampling Gaussian, in normalized action units"""
    action_prior_ckpt: str | None = None
    """checkpoint from lewm.action_prior. If set, CEM is seeded and pruned with the prior
    (cem_plan_with_action_prior) instead of starting from a Gaussian"""
    gmm_prior_ckpt: str | None = None
    """checkpoint from lewm.gmm_prior. If set, CEM is seeded with GMM samples
    (cem_plan_with_gmm_prior); uses prior_num_samples and prior_cem_iters"""
    prior_num_samples: int = 1000
    """prior (DDIM or GMM) samples scored by the world model; the best cem_topk of them seed CEM"""
    prior_inference_steps: int = 10
    """DDIM denoising steps per prior window"""
    prior_eta: float = 1.0
    """DDIM eta (0 = deterministic given the initial noise, 1 = DDPM-like, more diverse)"""
    prior_cem_iters: int = 10
    """CEM rounds after seeding with the prior's elites"""
    heldout_dataset: str = "/home/anurag/.maniskill/demos/PushCube-v1/heldout_front48/trajectory.rgb.pd_joint_pos.physx_cpu.h5"
    # heldout_dataset: str = "/home/anurag/.maniskill/demos/PushCube-v1/cam_front48/trajectory.rgb.pd_joint_pos.physx_cpu.h5"

    play_prefix: int = 3

    seed_mode: str = "demo"
    """how the planner's history is filled before the first plan:
    "demo" replays the first history_size ground-truth chunks of the demo;
    "hold" executes history_size chunks of a no-motion arm action from reset (no GT used)"""
    hold_gripper: float | None = -1.0
    """gripper action during the hold prefix. -1 (closed) matches the demos, whose gripper action
    is always -1; None holds the reset opening (+1), which normalizes to ~230 std off-manifold"""

    wandb_config: WandBConfig = field(default_factory=WandBConfig)

@dataclass
class OnlineArgs:
    """lewm.online: alternate collecting planner rollouts into a RolloutBuffer and training on it."""

    init_ckpt: str = "lewm/runs/lewm_bn_ep8/latest.pt"
    """train.py checkpoint to start from (model, optimizer, action normalization, Args)"""
    gmm_prior_ckpt: str = "runs/gmm_prior/gmm_prior.pt"
    exp_name: Optional[str] = None
    output_dir: str = "lewm/runs"
    seed: int = 0
    device: str = "cuda"

    rounds: int = 100
    """collect + train rounds"""
    rollouts_per_round: int = 8
    grad_steps_per_round: int = 200
    pin_frac: float = 0.5
    """fraction of the expert demos that are never evicted"""
    perturb_alpha: float = 0.0
    """after each round's rollouts, move the weights this fraction toward a fresh random init
    (theta <- (1 - a) theta + a theta_rand, shrink-and-perturb). 0 disables"""
    freeze_encoder: bool = False
    """freeze the encoder and projector (the embedding, incl. its BatchNorm stats) and train only
    the predictor, action encoder and pred_proj. SIGReg then has no gradient; only pred_loss trains"""
    freeze_bn: bool = False
    """freeze every BatchNorm layer: affine params get no gradient and the layers stay in eval mode,
    so training normalizes with the checkpoint's running stats (as planning does) and never updates them"""
    learning_rate: Optional[float] = None
    """constant LR for online training. None uses the pretraining peak LR (optimizer_args.learning_rate)"""
    num_workers: int = 4
    demo_dataset_path: str = EnvArgs.demo_dataset_path
    """expert demos (the checkpoint's own path may point at the machine it was trained on)"""
    max_trajectories: Optional[int] = None
    """override the checkpoint's demo load count (smoke tests)"""
    val_every: int = 1
    """rounds between expert-val evaluations (and checkpoint saves)"""

    seed_mode: str = "hold"
    """planner history before the first plan: "hold" (no-motion prefix) or "demo" (the task's demo prefix)"""
    hold_gripper: Optional[float] = -1.0
    max_episode_steps: int = 100
    """env steps per rollout, including the prefix. Rollouts also stop at the first success"""
    plan_horizon: int = 13
    prior_num_samples: int = 1000
    prior_cem_iters: int = 10

    wandb_config: WandBConfig = field(default_factory=WandBConfig)
