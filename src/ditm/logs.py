import logging
from . import config
from datetime import datetime
from pathlib import Path


def setup_logging(
    logdir: Path = config.LOGDIR,
    logformat: str = config.LOGFORMAT,
    loglevel: int = config.LOGLEVEL,
) -> Path:
    """ Sets up logging to the path defined in config.
    Call this at the top of any script.
    """

    logpath = logdir / f"{datetime.now():%Y-%m-%d %H%M}.log"
    logpath.parent.mkdir(parents = True, exist_ok = True)

    logging.basicConfig(
        format = logformat,
        level = loglevel,
        handlers = [
            logging.FileHandler(filename = logpath, delay = True),
            logging.StreamHandler(),
        ],
        force = True
    )
    logger = logging.getLogger(__name__)
    return logpath

