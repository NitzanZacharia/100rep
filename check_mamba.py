import torch
try:
    from mamba_ssm import Mamba
    from causal_conv1d import causal_conv1d_fn
    print("✅ Libraries imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # בדיקת ה-Kernels: נסיון ליצור מודל קטנטן ולהעביר ל-GPU
    try:
        model = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).to("cuda")
        test_input = torch.randn(1, 10, 128).to("cuda")
        output = model(test_input)
        print("🚀 Mamba kernels are working perfectly on GPU!")
    except Exception as e:
        print(f"❌ Kernel Error: {e}")
else:
    print("❌ GPU not detected by PyTorch.")
