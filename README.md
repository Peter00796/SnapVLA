# SnapVLA

A learning-oriented pipeline for running Vision-Language(-Action) models across
a split edge/server setup: a **Raspberry Pi + Rasprover** captures camera frames,
a **Windows PC with an RTX GPU** runs inference, and every stage of the pipeline
is instrumented so the data flow is visible end-to-end.

## Hosts

| Role        | Path                                                      | Python | Env manager |
|-------------|-----------------------------------------------------------|--------|-------------|
| Dev (Mac)   | `~/Desktop/Spring2026/Rasberry_Pi_VlA/SnapVLA`            | 3.11   | uv          |
| Edge (Pi)   | `~/vla_experiments`                                       | 3.11   | uv          |
| Server (PC) | `C:\Users\29838\Documents\Researches\vla_experiments`     | 3.11   | uv          |

## Layout

```
snapvla/            # shared Python package
  pipeline/         # BaseContext, Stage, VLAPipeline + StageTrace
  common/           # wire protocol, frame codec
  edge/             # sensor adapters (runs on Pi)
  server/           # VLM/VLA stages (runs on PC)
edge/               # Pi-side runtime entrypoint + pyproject
server/             # PC-side runtime entrypoint + pyproject
scripts/            # sync, smoke tests
examples/           # end-to-end demos
```

## Workflow

```
Mac  -- commit + push -->  GitHub  <-- git pull --  Pi / Windows
```

After any change on the Mac:

```bash
git push
scripts/sync.sh            # ff-pulls both runtime hosts
```

## Quickstart

```bash
# root (shared lib, dev tools)
uv sync

# Pi
ssh yanxin@raspberrypi.local 'cd ~/vla_experiments/edge && uv sync'

# Windows
ssh 29838@192.168.88.12 'cd C:\Users\29838\Documents\Researches\vla_experiments\server && uv sync'
```
