# 07 — MPMD pipeline parallelism

Real multi-program pipeline parallelism. `spx.run` auto-splits
`model.blocks` across an MPMD mesh axis; same `Llama3` class runs
under SPMD or MPMD depending on the mesh.

| file                                                     | what it shows                                                                                            |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [`01_train_homogeneous.py`](01_train_homogeneous.py)     | One `spx.run(mode='train')` call auto-splits `Llama3.blocks` across the PP axis. ~50 LOC.                |
| [`02_inference_forward.py`](02_inference_forward.py)     | `mode='forward'` inference — same model, same mesh, no grads.                                            |
| [`03_spmd_vs_mpmd.py`](03_spmd_vs_mpmd.py)               | Same model run under two meshes (no PP vs PP=2). Losses match — numerical parity of the unified runtime. |
| [`04_decode_loop.py`](04_decode_loop.py)                 | Greedy autoregressive decode via repeated `mode='forward'` calls.                                        |
| [`05_pp_fsdp_tp.py`](05_pp_fsdp_tp.py)                   | 3-D mesh (pp=2 x tp=2) composing pipeline with tensor-parallel FSDP.                                     |
| [`06_all_schedulers.py`](06_all_schedulers.py)           | Run the same model under all 9 pipeline schedules and compare losses.                                    |
| [`07_fused_tasks.py`](07_fused_tasks.py)                 | `fuse_1f1b=True` steady-state fusion — same loss, fewer dispatches.                                      |
| [`08_pp_stage_assignment.py`](08_pp_stage_assignment.py) | Manual `pp_stage` annotation to override automatic stage placement.                                      |
| [`09_dualpipev_tasks.py`](09_dualpipev_tasks.py)         | Per-rank task list inspection for DualPipe-V (FusedTask / Action breakdown).                             |
| [`10_mpmd_array.py`](10_mpmd_array.py)                   | `MpMdArray` — build, inspect shards, check locality, gather across processes.                            |
| [`11_mpmd_jit_generation.py`](11_mpmd_jit_generation.py) | `@mpmd_jit` — jaxpr-split true MPMD generation. Trace → split at markers → per-rank XLA executables. Load real Llama 3.2 3B. |

All examples share `examples/models/llama.py` — write the model
**once** and let the mesh decide whether it runs as SPMD or MPMD.

Run any example:

```bash
python -m examples.07_mpmd.01_train_homogeneous
```
