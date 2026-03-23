# sweep.py
import argparse
import itertools

import submitit
from cs336_systems.benchmarking_script import benchmark


def run_config(size, context_length, forward_only=False):
    args = argparse.Namespace(
        size=size,
        context_length=context_length,
        d_model=None,
        d_ff=None,
        num_layers=None,
        num_heads=None,
        warmup=5,
        steps=10,
        forward_only=forward_only,
        markdown=True,
    )
    return benchmark(args)


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="logs/submitit")
    executor.update_parameters(
        gpus_per_node=1,
        timeout_min=10,
        slurm_partition="your-partition",
    )

    sizes = ["small", "medium", "large"]
    contexts = [128, 512, 1024]

    jobs = []
    for size, ctx in itertools.product(sizes, contexts):
        job = executor.submit(run_config, size, ctx)
        jobs.append((size, ctx, job))

    for size, ctx, job in jobs:
        result = job.result()
        print(f"{size} | ctx={ctx} | {result}")
