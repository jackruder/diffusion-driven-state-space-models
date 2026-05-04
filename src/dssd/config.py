from typing import List, Union, Literal, Optional, Annotated, TypeAlias

import ast
import yaml
from pydantic import Field, BaseModel, field_validator, model_validator


class Mamba2Config(BaseModel):
    """Configuration for Mamba2 attention module.

    Complexity Notation:
        L: Sequence length (e.g., T or j)
        C: Model dimension (d_model)

    Impact:
        - Time: O(L * C * state_dim)
        - Space: O(B * L * C + B * C * state_dim) (KV-state equivalent)

    Args:
        state_dim: SSM state dimension (N). Higher = better memory, slower compute.
        headdim: Dimension per head (P).
    """

    state_dim: int = 64
    headdim: int = 16

    @model_validator(mode="after")
    def check_dims(self):
        if self.state_dim <= 0 or self.headdim <= 0:
            raise ValueError("state_dim and headdim must be positive integers")
        if self.state_dim % self.headdim != 0:
            raise ValueError(
                f"state_dim ({self.state_dim}) must be divisible by headdim ({self.headdim})"
            )
        return self


class TransformerTimeConfig(BaseModel):
    type: Literal["transformer_time"] = "transformer_time"
    nheads: int = 4
    layers: int = 1
    dropout: float = 0.0
    ff_mult: int = 4


class MambaTimeConfig(BaseModel):
    type: Literal["mamba"] = "mamba"
    state_dim: int = 64
    headdim: int = 16


class GRUTimeConfig(BaseModel):
    type: Literal["gru"] = "gru"
    hidden_dim: int = 64
    gru_layers: int = 1


class ConvTimeConfig(BaseModel):
    type: Literal["conv"] = "conv"
    kernel_size: int = 3


class IdentityTimeConfig(BaseModel):
    type: Literal["identity"] = "identity"


TimeConfig: TypeAlias = Annotated[
    Union[
        MambaTimeConfig,
        ConvTimeConfig,
        IdentityTimeConfig,
        GRUTimeConfig,
        TransformerTimeConfig,
    ],
    Field(discriminator="type"),
]


class TransformerFeatureConfig(BaseModel):
    type: Literal["transformer"] = "transformer"
    nheads: int = 8
    layers: int = 1


class ConvFeatureConfig(BaseModel):
    type: Literal["conv"] = "conv"
    kernel_size: int = 3


class IdentityFeatureConfig(BaseModel):
    type: Literal["identity"] = "identity"


FeatureConfig: TypeAlias = Annotated[
    Union[TransformerFeatureConfig, ConvFeatureConfig, IdentityFeatureConfig],
    Field(discriminator="type"),
]


class DiffusionScheduleConfig(BaseModel):
    """How noise levels β₁…β_T are spaced."""

    S_k: int = 1  # Monte carlo samples for diffusion
    k_chunk: int = 1  # size to chunk over S_k
    num_steps: int  # Total diffusion steps T_diff.
    schedule: str = "linear"  # 'linear' (β_t linear) or 'quad' (√β_t linear)

    sigma_min: float = 2e-3  # Karras schedule, EDM
    sigma_max: float = 80.0
    rho: float = 7.0

    k_sampling_mode: str = "uniform"  # or "importance"
    pk_gamma: float = 1.0  # p(k) ∝ w_k**gamma (1.0 → w_k; 0.0 → uniform)
    pk_floor: float = 1e-12  # numerical floor


class DiffusionEmbeddingConfig(BaseModel):
    """How to embed a step index t→vector."""

    embedding_dim: int = 128
    projection_dim: Optional[int] = None

    @field_validator("embedding_dim")
    def must_be_power_of_two(cls, v: int) -> int:
        if v <= 0 or (v & (v - 1)) != 0:
            raise ValueError("embedding_dim must be a positive power of two")
        return v

    @field_validator("projection_dim", mode="after")
    def default_projection_dim(cls, v: Optional[int], info) -> int:
        return v if v is not None else info.data["embedding_dim"]


class ResidualBlockConfig(BaseModel):
    """U-Net ResidualBlock settings.

    Used in DiffResidualBlock (UNet) and ResidualBlock (ContextProducer).

    Complexity Constants:
        C: channels
        L: layers
        T: Sequence length (time axis)
        d: Feature dimension (spatial/latent axis)
    Args:
        channels: Model dimension (C).
        layers: Number of blocks (L).
        nheads: Number of attention heads.
    """

    channels: int = 64  # base width
    layers: int = 4  # number of residual blocks
    nheads: int = 8  # attention heads
    time: TimeConfig = Field(default_factory=MambaTimeConfig)
    feature: FeatureConfig = Field(default_factory=TransformerFeatureConfig)


class UNetConfig(BaseModel):
    """Complete diffusion U-Net configuration.

    Complexity Constants:
        T: Sequence length (time_len)
        d: Latent dimension (latent_dim)
        C: block.channels
        L: block.layers
        N: mamba.state_dim

    """

    embedding: DiffusionEmbeddingConfig
    block: ResidualBlockConfig


class GaussianHeadConfig(BaseModel):
    """Configuration for Gaussian parameterization head."""

    init_logvar: float = 0.0
    var_min: float = 1e-6
    clamp_logvar_min: float = -9.0
    clamp_logvar_max: float = 6.0


class ContextProducerConfig(BaseModel):
    """Configuration for ContextProducer module.

    Used in Encoder/Decoder to process history windows.

    Complexity Notation:
        j: History length
        C: Channels (channels)

    Impact:
        - Time: O(num_layers * j * C^2)
    """

    channels: int = 4
    num_layers: int = 1
    nheads: int = 8
    time: TimeConfig = Field(default_factory=MambaTimeConfig)
    feature: FeatureConfig = Field(default_factory=TransformerFeatureConfig)


class FutureSummaryConfig(BaseModel):
    """Future summary module configuration.

    Complexity Notation:
        T: Sequence length
        C: Hidden dim (hidden_dim)

    Impact:
        - Time: O(T * C^2) (time mixer scan)
    """

    summary_dim: int = 64  #  dimension (and output dimension) of the future summary
    num_layers: int = 2
    # Pluggable time mixer (Mamba/Conv/Identity), default to Mamba for BC
    time: TimeConfig = Field(default_factory=MambaTimeConfig)

    @model_validator(mode="after")
    def check_dims_match(self):
        # Specific check for GRU/Mamba/Conv which expose hidden_dim or state_dim
        # For GRUTimeConfig: hidden_dim
        if isinstance(self.time, GRUTimeConfig):
            if self.time.hidden_dim != self.summary_dim:
                raise ValueError(
                    f"FutureSummary summary_dim ({self.summary_dim}) must match "
                    f"GRU hidden_dim ({self.time.hidden_dim})"
                )
        return self


class EncoderConfig(BaseModel):
    """Encoder configuration.

    Complexity Notation:
        T: Sequence length (time_len)
        D: Data dimension (data_dim)
        C: Hidden dimension (hidden_dim)

    Impact:
        - Time: O(T * (D*C + C^2)) (Future summary + Context processing)
        - Space: O(B * T * C)

    Args:
        hidden_dim: Internal model dimension (C).
    """

    fut_mask_emb_dim: int = 8
    pad_mask_emb_dim: int = 8
    hidden_dim: int = 64  # hidden dimension of the encoder
    context: ContextProducerConfig = ContextProducerConfig()
    summary_config: FutureSummaryConfig

    gaussian_head: GaussianHeadConfig = GaussianHeadConfig()
    latent_init_head: GaussianHeadConfig = GaussianHeadConfig()


class InitPriorConfig(BaseModel):
    """Configuration for Initialization Prior."""

    pad_mask_emb_dim: int = 8
    hidden_dim: int = 64
    context: ContextProducerConfig = ContextProducerConfig()
    aux_context: ContextProducerConfig = ContextProducerConfig()
    gaussian_head: GaussianHeadConfig = GaussianHeadConfig()
    latent_init_head: GaussianHeadConfig = GaussianHeadConfig()


class TransitionConfig(BaseModel):
    """Configuration for Transition model."""

    type: str
    hidden_dim: int = 64


class GaussianTransitionConfig(TransitionConfig):
    # should assert that type is gaussian
    type: Literal["gaussian"] = "gaussian"
    context: ContextProducerConfig = ContextProducerConfig()
    gaussian_head: GaussianHeadConfig = GaussianHeadConfig()


class DiffusionTransitionConfig(TransitionConfig):
    type: Literal["diffusion"] = "diffusion"
    unet: UNetConfig
    schedule: DiffusionScheduleConfig


TransitionModelConfig: TypeAlias = Annotated[
    Union[GaussianTransitionConfig, DiffusionTransitionConfig],
    Field(discriminator="type"),
]


class DecoderConfig(BaseModel):
    """Decoder configuration.

    Complexity Notation:
        j: History length
        C: Hidden dim (hidden_dim)

    Impact:
        - Time: O(j * C^2) per step
    """

    context: ContextProducerConfig = ContextProducerConfig()
    gaussian_head: GaussianHeadConfig = GaussianHeadConfig()
    mask_emb_dim: int = 8
    hidden_dim: int = 64


class REWOConfig(BaseModel):
    """Configuration for REWO"""

    D0: float = 0.1  # Target distortion
    nu: float = 1e-3  # Learning rate for lambda
    alpha: float = 0.99  # EMA decay for distortion
    tau1: float = 1.0  # grid search for these?
    tau2: float = 1.0


class DSSDHyperParams(BaseModel):
    """Training hyperparameters for the DSSD model."""

    S: int = 1  # Monte Carlo samples
    ema_decay: float = 0.999
    weight_decay: float = 1e-2  # weight decay for optimizer
    batch_size: int = 16  # batch size for training
    grad_accum_steps: int = 4  # mini batch size for gradient accumulation
    t_chunk: int = 16  # chunk size over time for transition model pass
    clip_grad_norm: float | None = None  # max norm for gradient clipping (1.0)

    lambda_schedule: Literal["none", "linear", "cosine", "rewo"] = "none"
    lambda_start: float = 0.001
    lambda_end: float = 1.0
    lambda_warmup_steps: int = 10

    enc_lr: float = 5e-4
    dec_lr: float = 5e-4
    zinit_lr: float = 5e-4
    trans_lr: float = 5e-4

    logvar_min: float = -7.0  # min log-variance for decoder/encoder output
    logvar_max: float = 7.0  # max log-variance for decoder/encoder output

    rewo: REWOConfig = Field(default_factory=REWOConfig)


class LambdaRamp(BaseModel):
    # lambda at the beginning of the stage (and during delay)
    start: float = 0.05
    end: float | None = 1.0
    delay: int = 0  # flat steps at 'start' before ramp begins
    steps: int | None = None  # ramp duration in steps; if None, use stage.steps


class StageTrainable(BaseModel):
    encoder: bool = True
    decoder: bool = True
    z_init: bool = True
    transition: bool = False


class StageLrs(BaseModel):
    enc_lr: float = 5e-4
    dec_lr: float = 5e-4
    zinit_lr: float = 5e-4
    trans_lr: float = 0.0


class StageScheduler(BaseModel):
    type: Literal["none", "cosine"] = "none"
    warmup_steps: int = 0
    final_lr_scale: float = 1.0


class StageSpec(BaseModel):
    mode: Literal["recon_only", "trans_only", "joint"]
    steps: int
    trainable: StageTrainable
    lrs: StageLrs
    scheduler: StageScheduler = StageScheduler()
    carry_diff_moments: bool = False
    lambda_ramp: LambdaRamp = LambdaRamp()
    log_every: int = 10  # how often to log metrics, setps
    val_every: int = 100  # how often to validate, steps
    checkpoint_every: int = 1000  # how often to save checkpoints, steps


class StagesConfig(BaseModel):
    stage_1: StageSpec | None = None
    stage_2: StageSpec | None = None
    stage_3: StageSpec | None = None

    run: List[str] = ["stage_1", "stage_2", "stage_3"]

    @field_validator("run")
    def names_are_allowed(cls, v: List[str]) -> List[str]:
        allowed = {"stage_1", "stage_2", "stage_3"}
        unknown = set(v) - allowed
        if unknown:
            raise ValueError(f"Unknown stage names in 'run': {sorted(unknown)}")
        return v

    @model_validator(mode="after")
    def run_stages_must_exist(self):
        available = {
            name
            for name in ("stage_1", "stage_2", "stage_3")
            if getattr(self, name) is not None
        }
        missing = set(self.run) - available
        if missing:
            raise ValueError(
                f"Stages listed in 'run' are missing from the config: {sorted(missing)}"
            )
        return self


class DSSDConfig(BaseModel):
    """Top-level DSSD model configuration.

    Global Complexity Constants:
        B: Batch size (hyperparams.batch_size)
        T: Time length (time_len)
        D: Data dimension (data_dim)
        d: Latent dimension (latent_dim)
        j: History window (j)

    """

    encoder: EncoderConfig
    decoder: DecoderConfig
    z_init: InitPriorConfig
    transition: TransitionModelConfig

    j: int = 1  # number of latent steps to condition on

    data_dim: int = 1
    latent_dim: int = 1
    covariate_dim: int = 0  # dimension of any additional covariates

    # static categorical encodings
    static_embed_dim: int = 0  # dimension per categorical feature embedding (E_s)
    num_classes_per_static: List[int] = Field(
        default_factory=list
    )  # Vocabulary size per static feature.

    emb_time_dim: int = 16
    mask_emb_dim: int = 8
    use_observation_mask: bool = True  # whether to make use of observation mask
    stages: StagesConfig | None = None
    hyperparams: DSSDHyperParams

    checkpoint_dir: str = "./checkpoints"  # where to save model checkpoints

    @classmethod
    def load_yaml(cls, path: str) -> "DSSDConfig":
        """Load a DSSDConfig from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Pydantic v2: use model_validate; fallback to parse_obj for v1
        if hasattr(cls, "model_validate"):
            return cls.model_validate(data)
        return cls.parse_obj(data)


# helpers
def deep_merge(base, update):
    """Recursively merge update dict into base dict."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def apply_dot_overrides(base_dict: dict, overrides: list[str]) -> dict:
    """Applies dot-notation overrides (e.g., 'hyperparams.batch_size=32') to a dictionary."""
    if not overrides:
        return base_dict

    for override in overrides:
        if "=" not in override:
            print(f"[Warning] Invalid override format: {override}. Expected key=value.")
            continue

        key_path, val_str = override.split("=", 1)
        keys = key_path.split(".")

        # Safely parse the value
        if val_str.lower() == "true":
            val = True
        elif val_str.lower() == "false":
            val = False
        elif val_str.lower() == "null":
            val = None
        else:
            try:
                val = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                val = val_str  # Fallback to string

        # Traverse and set the value
        current = base_dict
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = val
        print(f"[Config Override] {key_path} = {val}")

    return base_dict


def load_config_from_files(paths: list[str], overrides: list[str] = None) -> DSSDConfig:
    """Load multiple YAMLs, merge them, and apply CLI overrides."""

    # 1. Load base
    print(f"[Config] Loading base configuration from {paths[0]}")
    with open(paths[0], "r") as f:
        merged_data = yaml.safe_load(f)

    for path in paths[1:]:
        print(f"[Config] Merging override configuration from {path}")
        with open(path, "r") as f:
            update_data = yaml.safe_load(f)
        deep_merge(merged_data, update_data)

    if overrides:
        merged_data = apply_dot_overrides(merged_data, overrides)

    config = DSSDConfig.model_validate(merged_data)
    return config
