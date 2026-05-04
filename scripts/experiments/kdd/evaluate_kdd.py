import os
import json
import argparse
import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from ddssm.ddssm import DDSSM_base
from ddssm.config import DDSSMConfig
from ddssm.data.dataload import build_loaders_for_expt, parse_batch
import numpy as np

from ddssm.config import apply_dot_overrides, load_config_from_files

torch.set_float32_matmul_precision("high")


def mae_metrics(pred_mean: torch.Tensor, y_future: torch.Tensor):
    abs_err = (pred_mean - y_future).abs()  # (B,D,L2)
    mae_global = abs_err.mean()  # scalar
    mae_per_t = abs_err.mean(dim=(0, 1))  # (L2,)
    return mae_global, mae_per_t


def crps_sum_metrics(pred_samples: torch.Tensor, y_future: torch.Tensor):
    B, S, D, L2 = pred_samples.shape
    quantile_levels = torch.arange(0.05, 1.0, 0.05, device=pred_samples.device)

    pred_sum_samples = pred_samples.sum(dim=2)  # (B,S,L2)
    y_sum = y_future.sum(dim=1)  # (B,L2)

    quantiles = torch.quantile(
        pred_sum_samples, quantile_levels, dim=1, keepdim=False
    )  # (Q,B,L2)
    quantiles = quantiles.permute(1, 2, 0)  # (B,L2,Q)

    y = y_sum.unsqueeze(-1)  # (B,L2,1)
    indicator = (y < quantiles).float()
    quantile_loss = (quantile_levels - indicator) * (y_sum.unsqueeze(-1) - quantiles)
    crps = 2 * quantile_loss.mean(dim=-1)  # (B,L2)

    crps_sum_global = crps.mean()
    crps_sum_per_t = crps.mean(dim=0)  # (L2,)
    return crps_sum_global, crps_sum_per_t


def main():
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 64
    torch._dynamo.config.recompile_limit = 64
    inductor_config.pad_outputs = False
    inductor_config.pad_dynamic_shapes = False
    inductor_config.force_shape_pad = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--config", type=str, nargs="+", required=True)
    parser.add_argument("--set", type=str, nargs="+", default=None)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--work_dir", type=str, default="runs/eval_kdd")
    parser.add_argument("--split", type=int, default=96)
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--forecast_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    data_payload = torch.load(args.data_path, weights_only=False)
    series_list = data_payload["series_list"]
    covariates_list = data_payload["covariates_list"]
    static_covariates = data_payload["static_covariates"]

    # Build config exactly like training
    cfg = load_config_from_files(args.config, args.set)
    config_dict = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    config_dict = apply_dot_overrides(config_dict, args.set)
    config_dict["data_dim"] = len(series_list)
    config_dict["covariate_dim"] = covariates_list[0].shape[0]
    config_dict["num_classes_per_static"] = [
        int(static_covariates[:, 0].max() + 1),
        int(static_covariates[:, 1].max() + 1),
        int(static_covariates[:, 2].max() + 1),
    ]
    config = DDSSMConfig.model_validate(config_dict)

    L1, L2 = args.split, args.seq_len - args.split

    _, val_loader, _, _ = build_loaders_for_expt(
        series_list=series_list,
        L1=L1,
        L2=L2,
        val_windows=24,
        test_windows=27,
        eval_step_size=24,
        batch_size=args.batch_size,
        num_train_batches_per_epoch=None,
        covariates_list=covariates_list,
        static_covariates=static_covariates,
        backend="torch",
        normalize=True,
        device=device,
    )

    model = DDSSM_base(config, device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        _dummy = torch.randn(8, 8, device=device) @ torch.randn(8, 8, device=device)
        del _dummy
        torch.cuda.synchronize()

    model.to(device)
    model.load_state_dict(
        torch.load(args.resume, map_location=device)["model_state"], strict=True
    )
    model.eval()

    acc_mae = []
    acc_mae_per_t = []
    acc_crps_sum = []
    acc_crps_sum_per_t = []

    with torch.no_grad():
        for batch in val_loader:
            batch = parse_batch(batch, device)
            x_hist = batch["observed_data"][..., :L1]
            x_mask = batch["observation_mask"][..., :L1]  # fixed key
            past_time = batch["timepoints"][:, :L1]
            future_time = batch["timepoints"][:, L1:]
            y_future = batch["observed_data"][..., L1:]
            static_cov = batch.get("static_covariates", None)
            covariates = batch.get("covariates", None)

            out = model.forecast(
                x_hist=x_hist,
                x_mask=x_mask,
                past_time=past_time,
                future_time=future_time,
                past_covariates=covariates[..., :L1]
                if covariates is not None
                else None,
                future_covariates=covariates[..., L1:]
                if covariates is not None
                else None,
                static_covariates=static_cov,
                num_samples=args.forecast_samples,
            )
            pred_samples = out["pred_samples"]  # (B,S,D,L2)
            pred_mean = pred_samples.mean(dim=1)  # (B,D,L2)

            mae_g, mae_t = mae_metrics(pred_mean, y_future)
            crps_sum_g, crps_sum_t = crps_sum_metrics(pred_samples, y_future)

            acc_mae.append(float(mae_g.item()))
            acc_mae_per_t.append(mae_t.detach().cpu().numpy().tolist())
            acc_crps_sum.append(float(crps_sum_g.item()))
            acc_crps_sum_per_t.append(crps_sum_t.detach().cpu().numpy().tolist())

    metrics = {
        "mae": float(np.mean(acc_mae)),
        "mae_per_t": np.mean(np.stack(acc_mae_per_t, axis=0), axis=0).tolist(),
        "crps_sum": float(np.mean(acc_crps_sum)),
        "crps_sum_per_t": np.mean(
            np.stack(acc_crps_sum_per_t, axis=0), axis=0
        ).tolist(),
    }

    os.makedirs(args.work_dir, exist_ok=True)
    metrics_path = os.path.join(args.work_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Save] {metrics_path}")
    print(f"[Eval Done] crps_sum={metrics['crps_sum']:.6f} mae={metrics['mae']:.6f}")


if __name__ == "__main__":
    main()
