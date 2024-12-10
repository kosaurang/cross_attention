# utils/device_utils.py
import torch
import importlib.util

def is_module_available(module_name):
    """Kiểm tra một module có được cài đặt không"""
    return importlib.util.find_spec(module_name) is not None

def select_device(force_tpu=False, force_cuda=False, verbose=False):
    """
    Chọn device phù hợp với môi trường huấn luyện
    
    Args:
        force_tpu (bool): Ép buộc sử dụng TPU nếu có
        force_cuda (bool): Ép buộc sử dụng CUDA nếu có
        verbose (bool): In ra thông tin về device được chọn
    
    Returns:
        device: Thiết bị được chọn
    """
    try:
        # Kiểm tra TPU
        if is_module_available('torch_xla.core.xla_model'):
            import torch_xla.core.xla_model as xm
            
            if force_tpu:
                device = xm.xla_device()
                if verbose:
                    print(f"[DEVICE] Forced TPU: {device}")
                return device
            
            if force_cuda and torch.cuda.is_available():
                device = torch.device('cuda')
                if verbose:
                    print(f"[DEVICE] Forced CUDA: {device}")
                return device
            
            device = xm.xla_device()
            if verbose:
                print(f"[DEVICE] TPU Detected: {device}")
            return device
    except ImportError:
        pass
    
    # Kiểm tra CUDA
    if torch.cuda.is_available() or force_cuda:
        device = torch.device('cuda')
        if verbose:
            print(f"[DEVICE] CUDA: {device}")
        return device
    
    # Fallback về CPU
    device = torch.device('cpu')
    if verbose:
        print(f"[DEVICE] Fallback to CPU: {device}")
    return device

# Tạo một global device để các module khác có thể import
GLOBAL_DEVICE = select_device(verbose=True)
