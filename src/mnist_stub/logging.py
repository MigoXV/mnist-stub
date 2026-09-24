from __future__ import annotations

import logging
import sys
from contextlib import contextmanager

from lightning.pytorch.callbacks import TQDMProgressBar
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class StderrProgressBar(TQDMProgressBar):
    def init_sanity_tqdm(self):
        return self._bar(super().init_sanity_tqdm)

    def init_train_tqdm(self):
        return self._bar(super().init_train_tqdm)

    def init_validation_tqdm(self):
        return self._bar(super().init_validation_tqdm)

    def init_test_tqdm(self):
        return self._bar(super().init_test_tqdm)

    @staticmethod
    def _bar(factory):
        # Lightning's factory selects stdout; replace only the stream, before first update.
        import contextlib

        with contextlib.redirect_stdout(sys.stderr):
            return factory()


@contextmanager
def training_logging(path, progress: bool):
    from datasets.utils import logging as dataset_logging
    from transformers.utils import logging as transformers_logging

    root = logging.getLogger()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    previous_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    loggers = [logging.getLogger(name) for name in ("lightning", "lightning.pytorch")]
    previous = [(item, item.handlers[:], item.propagate) for item in loggers]
    for item in loggers:
        item.handlers = []
        item.propagate = True
    was_disabled = dataset_logging.is_progress_bar_enabled()
    transformers_progress = transformers_logging.is_progress_bar_enabled()
    if not progress:
        dataset_logging.disable_progress_bar()
        transformers_logging.disable_progress_bar()
    logging.captureWarnings(True)
    try:
        with logging_redirect_tqdm(loggers=[root], tqdm_class=tqdm):
            yield
    finally:
        logging.captureWarnings(False)
        if was_disabled:
            dataset_logging.enable_progress_bar()
        if transformers_progress:
            transformers_logging.enable_progress_bar()
        for item, handlers, propagate in previous:
            item.handlers = handlers
            item.propagate = propagate
        root.removeHandler(file_handler)
        file_handler.close()
        root.setLevel(previous_level)
