import hashlib, json
from dataclasses import is_dataclass, asdict

def make_runhash(*parts):
    serialized = json.dumps(
        [asdict(part) if is_dataclass(part) else part for part in parts],
        sort_keys = True,
        default = str,
    )
    assert("0x" not in serialized)
    hashed = hashlib.sha256(serialized.encode()).hexdigest()[:12]
    return hashed
