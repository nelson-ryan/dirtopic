from pathlib import Path
from collections.abc import Callable

def get_or_make(
    path: Path | None,
    make_fn: Callable,
    *,
    save_fn: Callable,
    load_fn: Callable,
):
    """ Flexible-but-cryptic function to either load a cached object or build """
    if path is None:
        return make_fn()
    if path.exists():
        with open(path, 'rb') as f:
            return load_fn(f)
    # else
    result_obj = make_fn()
    # first save to temp path to prevent saving an interrupted/broken file
    tmpfile = path.with_name(path.name + ".part")
    with open(tmpfile, "wb") as f:
        save_fn(result_obj, f)
    tmpfile.replace(path)

    return result_obj

