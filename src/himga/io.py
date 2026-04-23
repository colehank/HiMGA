# %%
import json

from himga.utils import get_dataset

locomo = get_dataset("locomo")
longmemeval = get_dataset("longmemeval")
# %%
locomo_fps = [
    json.loads(fp.read_text(encoding="utf-8"))
    for fp in locomo.iterdir()
    if fp.is_file() and fp.suffix == ".json"
]
# %%
