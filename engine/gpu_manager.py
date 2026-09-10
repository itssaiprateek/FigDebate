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
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print(
                    f"[GPU] Memory : "
                    f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
                )
            except RuntimeError as error:
                # Cleanup is best effort. In particular, an asynchronous CUDA
                # failure must not replace the earlier model exception.
                print(f"[GPU][cleanup-warning] {str(error).splitlines()[0]}")
                return False
        return True
