import torch
import sys

print("=" * 60)
print("PyTorch CUDA 诊断报告")
print("=" * 60)

print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    计算能力: {torch.cuda.get_device_capability(i)}")
        print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
else:
    print("\n❌ CUDA 不可用！可能的原因：")
    print("1. PyTorch 安装的是 CPU 版本")
    print("2. CUDA 驱动未安装或版本不兼容")
    print("3. NVIDIA 显卡驱动未安装")

    # 尝试诊断
    print("\n正在尝试诊断...")

    # 检查是否安装了 NVIDIA 驱动
    import subprocess

    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA 驱动已安装")
            print(result.stdout.split('\n')[0:3])
        else:
            print("❌ 未检测到 NVIDIA 驱动，或 nvidia-smi 命令不可用")
    except FileNotFoundError:
        print("❌ 未检测到 NVIDIA 驱动 (nvidia-smi 命令不存在)")

    # 检查 PyTorch 安装信息
    print(f"\nPyTorch CUDA 编译信息:")
    print(f"  CUDA 编译版本: {torch.version.cuda}")
    print(f"  cuDNN 版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")