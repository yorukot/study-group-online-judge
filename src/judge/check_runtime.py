"""Check the actual evaluation interpreter before requesting a Slurm GPU."""

import importlib
from importlib.metadata import version


def check_runtime() -> None:
    packages = (
        "torch",
        "transformers",
        "huggingface-hub",
        "tokenizers",
        "datasets",
        "kernels",
        "numpy",
        "safetensors",
        "pydantic",
        "wandb",
    )
    for package in packages:
        print(f"[judge] runtime {package}=={version(package)}", flush=True)
    # Import the same task registry that run_task imports. This catches version
    # checks, binary import failures, and missing trusted evaluator dependencies.
    for module in (
        "judge.run_task",
        "kernels",
        "transformers.integrations.hub_kernels",
    ):
        importlib.import_module(module)
    from transformers.utils import is_kernels_available

    if not is_kernels_available():
        raise RuntimeError("Transformers cannot use the installed kernels package")
    print("[judge] evaluator runtime imports passed", flush=True)


if __name__ == "__main__":
    check_runtime()
