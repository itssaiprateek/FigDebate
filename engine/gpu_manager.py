import gc
try:
    import torch
except ImportError:
    torch = None


class GPUManager:

    @staticmethod
    def clear():

        gc.collect()

        if torch is not None and torch.cuda.is_available():

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            print(
                f"[GPU] Memory : "
                f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
            )
