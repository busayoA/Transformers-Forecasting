from optuna.integration import PyTorchLightningPruningCallback
import optuna

def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 8, 64)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    model = TemporalFusionTransformer.from_dataset(
        dataset,
        hidden_size=hidden_size,
        dropout=dropout,
        learning_rate=learning_rate,
        attention_head_size=4,
        hidden_continuous_size=8,
        output_size=7,
        loss=QuantileLoss(),
    )

    trainer = Trainer(
        max_epochs=10,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    return trainer.callback_metrics["val_loss"].item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best Hyperparameters:", study.best_params)
