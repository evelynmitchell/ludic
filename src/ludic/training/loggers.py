from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence

from rich.console import Console
from rich.columns import Columns
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table


class TrainingLogger(Protocol):
    """
    Interface for logging training stats to arbitrary backends.
    """

    def log(self, step: int, stats: Dict[str, float]) -> None:
        ...


class PrintLogger:
    """
    Lightweight console logger (plain stdout).
    """

    def __init__(
        self,
        *,
        prefix: str = "[trainer]",
        keys: Sequence[str] | None = None,
        precision: int = 4,
        max_items_per_line: int = 6,
    ) -> None:
        self.prefix = prefix
        self.keys = list(keys) if keys is not None else None
        self.precision = precision
        self.max_items_per_line = max_items_per_line

    def _fmt_val(self, v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.{self.precision}f}"
        if isinstance(v, int):
            return str(v)
        return str(v)

    def log(self, step: int, stats: Dict[str, float]) -> None:
        if self.keys is not None:
            keys = [k for k in self.keys if k in stats]
        else:
            keys = sorted(stats.keys())
        pairs = [f"{k}={self._fmt_val(stats[k])}" for k in keys]
        if not pairs:
            print(f"{self.prefix} step={step}")
            return

        header = f"{self.prefix} step={step}"
        lines = []
        for i in range(0, len(pairs), self.max_items_per_line):
            lines.append(" ".join(pairs[i : i + self.max_items_per_line]))

        print(f"{header} {lines[0]}")
        indent = " " * len(header)
        for line in lines[1:]:
            print(f"{indent} {line}")


class RichLiveLogger:
    """
    Live-updating console logger using rich.Live with a sparkline.
    """

    def __init__(
        self,
        *,
        keys: Sequence[str] | None = None,
        spark_key: str = "avg_total_reward",
        history: int = 100,
        precision: int = 4,
        console: Console | None = None,
        live: Live | None = None,
    ) -> None:
        self.keys = list(keys) if keys is not None else None
        self.spark_key = spark_key
        self.history = history
        self.precision = precision
        self.console = console or Console()
        self.live = live or Live(console=self.console, refresh_per_second=4, transient=False)
        self.history_vals: list[float] = []
        self._own_live = live is None
        self._started = False

    def __enter__(self) -> "RichLiveLogger":
        if self._own_live:
            self.live.__enter__()
            self._started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._own_live:
            self.live.__exit__(exc_type, exc, tb)

    def _fmt_val(self, v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.{self.precision}f}"
        if isinstance(v, int):
            return str(v)
        return str(v)

    def _sparkline(self) -> tuple[str, float, float]:
        if not self.history_vals:
            return "", 0.0, 0.0
        vals = self.history_vals[-self.history :]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return "▁" * len(vals), lo, hi
        blocks = "▁▂▃▄▅▆▇█"
        scaled = []
        for v in vals:
            idx = int((v - lo) / (hi - lo) * (len(blocks) - 1))
            scaled.append(blocks[idx])
        return "".join(scaled), lo, hi

    def _render(self, step: int, stats: Dict[str, float]):
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("metric", style="cyan", no_wrap=True)
        table.add_column("value", style="magenta", overflow="fold")

        if self.keys is not None:
            keys = [k for k in self.keys if k in stats]
        else:
            keys = sorted(stats.keys())

        for k in keys:
            table.add_row(k, self._fmt_val(stats[k]))

        spark, lo, hi = self._sparkline()
        if spark:
            spark_panel = Panel(
                Text(spark, style="magenta"),
                title=f"{self.spark_key}",
                subtitle=f"{lo:.{self.precision}f} – {hi:.{self.precision}f}",
                padding=(0, 1),
            )
            step_label = Table(show_header=False, box=None, padding=(0, 0))
            step_label.add_column("step", style="cyan", no_wrap=True)
            step_label.add_column("val", style="magenta", no_wrap=True)
            step_label.add_row("step", str(step))

            table_with_step = Table.grid(padding=(0, 0))
            table_with_step.add_row(table)
            table_with_step.add_row(step_label)

            return Columns([table_with_step, spark_panel], expand=True, equal=True)

        # No spark: still show step
        step_label = Table(show_header=False, box=None, padding=(0, 0))
        step_label.add_column("step", style="cyan", no_wrap=True)
        step_label.add_column("val", style="magenta", no_wrap=True)
        step_label.add_row("step", str(step))

        table_with_step = Table.grid(padding=(0, 0))
        table_with_step.add_row(table)
        table_with_step.add_row(step_label)
        return table_with_step

    def log(self, step: int, stats: Dict[str, float]) -> None:
        spark_val = stats.get(self.spark_key)
        if isinstance(spark_val, (int, float)):
            self.history_vals.append(float(spark_val))
            if len(self.history_vals) > self.history:
                self.history_vals = self.history_vals[-self.history :]

        if self._own_live and not self._started:
            self.live.start()
            self._started = True

        table = self._render(step, stats)
        self.live.update(table)
class WandbLogger:
    """
    Minimal Weights & Biases logger. Lazily imports wandb.
    """

    def __init__(self, *, run: Any | None = None, init_kwargs: Dict[str, Any] | None = None) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - soft dependency
            raise ImportError("WandbLogger requires the 'wandb' package installed.") from exc

        self._wandb = wandb
        self._run = run or wandb.init(**(init_kwargs or {}))

    def log(self, step: int, stats: Dict[str, float]) -> None:
        self._wandb.log(stats, step=step)
