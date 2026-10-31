# Source before launching Python. Keep the Blackwell NCCL override project-local.
task_runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_nccl_lib="$task_runtime_root/.runtime/nccl_2_31_2/nvidia/nccl/lib/libnccl.so.2"
if [ -f "$task_nccl_lib" ]; then
  case ":${LD_PRELOAD:-}:" in
    *":$task_nccl_lib:"*) ;;
    *) export LD_PRELOAD="$task_nccl_lib${LD_PRELOAD:+:$LD_PRELOAD}" ;;
  esac
fi
