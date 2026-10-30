"""Native metric names, tqdm progress and TensorBoard scalar output."""
import numbers


def named_metrics(model, phase, values):
    if not isinstance(values, (tuple, list)):
        values = (values,)
    if phase == "pretrain":
        names = ("bow_loss",)
    elif model == "case_original":
        names = ("bow_loss", "kl_loss", "mim_loss", "ctx_loss", "ppl", "str_loss", "str_acc", "emo_loss", "emo_acc")
    else:
        names = ("loss", "ppl", "bce", "accuracy")
    return {name: float(value) for name, value in zip(names, values) if isinstance(value, numbers.Real)}


def metric_text(metrics):
    labels = {"bow_loss": "BOW", "kl_loss": "KL", "mim_loss": "MIM", "ctx_loss": "CTX",
              "ppl": "PPL", "str_loss": "STR_loss", "str_acc": "STR_acc", "emo_loss": "EMO_loss",
              "emo_acc": "EMO_acc", "loss": "Loss", "bce": "EMO_loss", "accuracy": "EMO_acc"}
    return " | ".join((labels.get(name, name) + "=" +
                       (f"{value * 100:.2f}%" if name in ("emo_acc", "str_acc", "accuracy") else f"{value:.4f}"))
                      for name, value in metrics.items() if name not in ("str_loss", "str_acc"))


def format_row(model, row):
    if "native_returns" in row:
        metrics = row.get("native_metrics") or named_metrics(model, row["phase"], row["native_returns"])
        return (f"TRAIN {row['phase']} | step={row['main_step']} | updates={row['completed_updates']} | "
                + metric_text(metrics) + f" | lr={row['lr']:.6g}")
    metrics = row.get("native_metrics", {"ppl": row["ppl"]})
    return (f"VALID | step={row['main_step']} | " + metric_text(metrics)
            + f" | saved_best={row['saved_best']} | bad_validations={row['bad_validations']}")


class TrainingDisplay:
    def __init__(self, model, recipe, output, progress, train_n):
        from torch.utils.tensorboard import SummaryWriter
        self.model, self.recipe = model, recipe
        self.pretrain_steps = ((train_n + recipe["batch_size"] - 1) // recipe["batch_size"]) * recipe["pretrain_epochs"]
        self.writer = SummaryWriter(str(output / "tensorboard"), purge_step=progress["completed_updates"] + 1)
        self.bar = None
        self.phase = None

    def update(self, progress, metrics, lr):
        from tqdm.auto import tqdm
        phase = progress["phase"]
        current = progress["completed_updates"] if phase == "pretrain" else progress["main_step"]
        if self.phase != phase:
            if self.bar is not None:
                self.bar.close()
            self.phase = phase
            total = self.pretrain_steps if phase == "pretrain" else self.recipe["max_main_steps"]
            self.bar = tqdm(total=total, initial=current - 1, desc=f"{self.model} {phase}",
                            unit="step", mininterval=1.0, dynamic_ncols=True)
        self.bar.set_postfix_str(metric_text(metrics) + f" | lr={lr:.3g}", refresh=False)
        self.bar.update(1)
        for name, value in metrics.items():
            if name not in ("str_loss", "str_acc"):
                self.writer.add_scalar(f"{phase}/{name}", value, progress["completed_updates"])
        self.writer.add_scalar("lr", lr, progress["completed_updates"])

    def record(self, row):
        from tqdm.auto import tqdm
        tqdm.write(format_row(self.model, row))
        if "native_returns" not in row:
            for name, value in row.get("native_metrics", {"ppl": row["ppl"]}).items():
                if name not in ("str_loss", "str_acc"):
                    self.writer.add_scalar("valid/" + name, value, row["completed_updates"])
            self.writer.flush()

    def close(self):
        if self.bar is not None:
            self.bar.close()
        self.writer.close()
