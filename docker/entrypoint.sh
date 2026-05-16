#!/usr/bin/env bash
# SkyLogic MAS — container entrypoint
# - Waits briefly for Postgres if DATABASE_URL points at it
# - Auto-loads patches from CSV on first boot (idempotent)
# - Then execs the requested CMD (uvicorn by default)
set -euo pipefail

log() { echo "[entrypoint] $*"; }

# --- Wait for Postgres if configured ---
if [ "${DATABASE_URL:-}" != "" ] && [[ "${DATABASE_URL}" == postgresql* ]]; then
    log "DATABASE_URL points at Postgres — waiting for it to accept TCP…"
    # Extract host:port from URL: postgresql://user:pw@HOST:PORT/db
    HOSTPORT=$(echo "$DATABASE_URL" | sed -E 's|.*@([^/]+)/.*|\1|')
    HOST=${HOSTPORT%%:*}
    PORT=${HOSTPORT##*:}
    [ "$PORT" = "$HOST" ] && PORT=5432
    for i in $(seq 1 60); do
        if (echo > /dev/tcp/$HOST/$PORT) 2>/dev/null; then
            log "Postgres reachable at $HOST:$PORT"
            break
        fi
        sleep 1
    done
fi

# --- Auto-seed patches from CSV on first boot ---
# Safe to run every boot; the loader skips already-existing rows.
if [ "${SKYLOGIC_AUTO_SEED:-1}" = "1" ] && [ -f /app/data/metadata/patches_metadata.csv ]; then
    log "Auto-seeding patches from CSV (idempotent)…"
    python -c "
from skylogic.database import init_db, SessionLocal
from skylogic.models.patch import Patch
import pandas as pd, ast
init_db()
db = SessionLocal()
try:
    df = pd.read_csv('/app/data/metadata/patches_metadata.csv')
    have = {p.filename for p in db.query(Patch.filename).all()}
    new = 0
    for _, r in df.iterrows():
        if r['filename'] in have:
            continue
        affine = None
        a = r.get('affine_transform')
        if isinstance(a, str) and a not in ('', 'nan'):
            try: affine = ast.literal_eval(a)
            except Exception: pass
        db.add(Patch(
            filename=r['filename'], source_image=r['source_image'],
            split=r.get('split','train'),
            x_offset=int(r.get('x_offset',0)), y_offset=int(r.get('y_offset',0)),
            width=int(r.get('width',512)), height=int(r.get('height',512)),
            affine_transform=affine, crs=r.get('crs'),
        ))
        new += 1
    db.commit()
    print(f'  seeded {new} new patches; total {len(have)+new}')
finally:
    db.close()
" || log "  seeding skipped: $?"
fi

log "Launching: $*"
exec "$@"
