"""List all valid Ultralytics train kwargs in the installed version."""
from ultralytics.cfg import DEFAULT_CFG
import inspect

cfg_dict = vars(DEFAULT_CFG)
print(f"Ultralytics version: {__import__('ultralytics').__version__}")
print(f"Total config keys: {len(cfg_dict)}\n")

# Print all keys related to loss / imbalance / augmentation
keywords = ['gamma', 'focal', 'cls', 'pw', 'smooth', 'amp', 'half',
            'box', 'dfl', 'mixup', 'mosaic', 'copy', 'bgr', 'degrees',
            'patience', 'save_period', 'plots', 'warmup', 'label']
print("=== Loss / imbalance / augment relevant params ===")
for k, v in sorted(cfg_dict.items()):
    if any(kw in k.lower() for kw in keywords):
        print(f"  {k:30s} = {v}")

print("\n=== ALL valid train params (sorted) ===")
for k, v in sorted(cfg_dict.items()):
    print(f"  {k:30s} = {repr(v)}")
