import pandas as pd
import numpy as np
from pytorch_forecasting import QuantileLoss, TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import torch 

# Load dataset
data = pd.read_csv("data/betting_data.csv")

# Ensure data is sorted by time for each product
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values(["product_id", "date"])

# Feature engineering
data["month"] = data["date"].dt.month
data["year"] = data["date"].dt.year
data["time_idx"] = (data["year"] - data["year"].min()) * 12 + data["month"]

holidays = pd.read_csv("data/holidays.csv")
data = data.merge(holidays, on="date", how="left")

# Fill missing values
data.fillna(0, inplace=True)

# Normalize numerical features
data["stake_norm"] = data["stake"] / data["stake"].max()

# Target
target = "monthly_revenue"

max_encoder_length = 12  # use past 12 months
max_prediction_length = 3  # forecast next 3 months

# Define TimeSeriesDataSet
dataset = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target=target,
    group_ids=["product_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["product_id"],
    time_varying_known_reals=["time_idx", "stake_norm"],
    time_varying_unknown_reals=[target],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Split into training and validation
train, val = dataset.split_by_random_split(0.8)
train_dataloader = dataset.to_dataloader(train=True, batch_size=64)
val_dataloader = dataset.to_dataloader(train=False, batch_size=64)

# Define the model
tft = TemporalFusionTransformer.from_dataset(
    dataset,
    learning_rate=0.01,
    hidden_size=16,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # number of quantiles for quantile regression
    loss=QuantileLoss(),
    log_interval=10,
    log_val_interval=1,
)

# Define trainer
early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = Trainer(
    max_epochs=50,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback, lr_monitor],
    gpus=1 if torch.cuda.is_available() else 0,
)

# Train model
trainer.fit(tft, train_dataloader, val_dataloader)


# Calculate baseline metrics
actuals = torch.cat([y for x, y in iter(val_dataloader)])
predictions = tft.predict(val_dataloader)
baseline = Baseline().fit(actuals)

# Evaluate accuracy
print("TFT RMSE:", tft.evaluate(val_dataloader, actuals=actuals))
print("Baseline RMSE:", baseline.evaluate(val_dataloader, actuals=actuals))


# Forecast future data
future_data = TimeSeriesDataSet.from_dataset(
    dataset, 
    data.iloc[-max_encoder_length:],  # last sequence
    predict=True,
    stop_randomization=True,
)

forecast = tft.predict(future_data)
print("Predicted Revenue:", forecast)


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

