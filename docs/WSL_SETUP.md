# WSL setup notes for SkyLogic MAS

## SAM ViT-H memory requirement

SAM ViT-H needs **≈6 GB of RAM** just to set the image embedding for one
512×512 input. On a default WSL2 installation (≈50 % of host RAM, often
≈8 GB total), the API container is killed by the kernel when SAM tries
to allocate.

You have two clean fixes — pick one.

### Option A — Give WSL more memory (recommended, requires ≥ 16 GB host)

Create or edit `C:\Users\<your-windows-username>\.wslconfig`:

```ini
[wsl2]
memory=12GB
swap=8GB
processors=6
```

Then in PowerShell:

```powershell
wsl --shutdown
```

Re-open WSL, then `cd /tmp/skylogic && docker compose -p skylogic up -d`.
Verify the new limit with `free -m` inside WSL — `total` should now be ~12000.

### Option B — Use the smaller SAM ViT-B checkpoint

ViT-B is ~358 MB on disk and runs comfortably in <2 GB RAM. Quality is
lower than ViT-H but still solid for click-based segmentation.

```bash
# Inside WSL (or via the host)
mkdir -p models/sam
curl -L \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o models/sam/sam_vit_b_01ec64.pth
```

Then add to `.env`:

```
SAM_CHECKPOINT=models/sam/sam_vit_b_01ec64.pth
```

(The `SAMAgent` constructor accepts `model_type="vit_b"`; we'd need a small
config flag to switch — see `skylogic/api/routes/sam.py::get_sam`.)
