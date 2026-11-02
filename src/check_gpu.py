
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA runtime used by PyTorch:", torch.version.cuda)

    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("GPU computation successful:", y.shape)
else:
    print("GPU not available. Check NVIDIA driver and PyTorch install.")
