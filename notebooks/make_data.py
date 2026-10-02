"""
Step 1: Generate the (simulated) cold-start dataset.

Each row = one cold-start invocation of an AWS Lambda function.
We simulate how long the boot (init) takes, then convert it to dollars
with the Lambda pricing formula:  cost = GB x seconds x price_per_GB_second

NOTE (be honest about this in the report/viva):
 - All numbers below are ASSUMED, realistic-looking ranges, not measured on AWS.
 - Check the current price on the AWS Lambda pricing page before quoting it.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(42)  # fixed seed -> same data every run

N = 1500
PRICE_PER_GB_SEC = 0.0000166667  # x86 rate, USD per GB-second (verify on AWS pricing page)

# ---- inputs (all known BEFORE the function runs) ----
memory = rng.choice([128, 256, 512, 1024, 2048, 3008], size=N)
runtime = rng.choice(["python", "node", "java"], size=N)
package_mb = rng.uniform(1, 100, size=N)
idle_min = rng.uniform(0, 60, size=N)

# ---- simulated boot (init) time in milliseconds ----
base = {"python": 250, "node": 200, "java": 900}      # runtime start-up cost
base_ms = np.array([base[r] for r in runtime])

init_ms = (
    base_ms
    + 3.0 * package_mb                 # more code to load -> slower
    + 1.5 * idle_min                   # assumption: longer idle -> slightly slower
    + rng.normal(0, 40, size=N)        # random noise
)
init_ms = init_ms * (1024 / memory) ** 0.3   # more memory -> more CPU -> faster boot
init_ms = np.clip(init_ms, 50, None)

# ---- target: dollar cost of the cold-start part ----
cost_overhead_usd = (memory / 1024) * (init_ms / 1000) * PRICE_PER_GB_SEC

df = pd.DataFrame({
    "memory_mb": memory,
    "runtime": runtime,
    "package_mb": package_mb.round(1),
    "idle_min": idle_min.round(1),
    "cost_overhead_usd": cost_overhead_usd,
})

out = ROOT / "data" / "coldstart_data.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows -> {out}")
print(df.head())
print(df.describe())
