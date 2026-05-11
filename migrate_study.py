import os
import json

import optuna


def migrate(old_name, new_name):
    storage_url = "sqlite:///runs/optuna/phase1/study.db"

    # Load old study
    old_study = optuna.load_study(study_name=old_name, storage=storage_url)

    # Create new study
    new_study = optuna.create_study(
        study_name=new_name,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,
    )

    print(f"Migrating trials from '{old_name}' to '{new_name}'...")

    count = 0
    for trial in old_study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        metrics_path = trial.user_attrs.get("metrics_path")
        if not metrics_path or not os.path.exists(metrics_path):
            print(f"Trial {trial.number} missing metrics.json, skipping.")
            continue

        try:
            with open(metrics_path, "r") as f:
                m = json.load(f)

            # EXTRACT THE STEP-1 CRPS
            new_value = float(m["sum_crps_per_t"][0])
        except Exception as e:
            print(f"Error reading trial {trial.number}: {e}")
            continue

        new_trial = optuna.trial.create_trial(
            params=trial.params,
            distributions=trial.distributions,
            value=new_value,
            user_attrs=trial.user_attrs,
            state=optuna.trial.TrialState.COMPLETE,
        )

        new_study.add_trial(new_trial)
        print(
            f"Migrated Trial {trial.number}: Old Avg={trial.value:.4f} --> New Step-1={new_value:.4f}"
        )
        count += 1

    print(f"\nMigration complete! {count} trials added to '{new_name}'.")
    if count > 0:
        print(
            f"Best migrated trial is Trial {new_study.best_trial.number} with Step-1 CRPS: {new_study.best_value:.4f}\n"
        )


if __name__ == "__main__":
    # Replace these with your actual old study names
    migrate("p1_bimodal", "p1_bimodal_gauss_step1_V2")
    migrate("p1_bimodal_diff_v2", "p1_bimodal_diff_step1_V2")
