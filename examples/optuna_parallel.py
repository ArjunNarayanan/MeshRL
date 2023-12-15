import optuna
from joblib import parallel_backend

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2


if __name__=="__main__":
    study = optuna.create_study(study_name="distributed-example")
    with parallel_backend("multiprocessing"):
        study.optimize(objective, n_trials=1_000, n_jobs=128)
    
    print("Best parameters : ", study.best_params)