from pathlib import Path
from tempfile import TemporaryDirectory

from test_train_v2_adapters import build_tiny_oftv2_model, run_tiny_train
from train_v2 import oft_identity_regularization


def main():
    reg = oft_identity_regularization(build_tiny_oftv2_model())
    print("oft_identity_regularizer", reg)

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        oft_reg_dir = run_tiny_train(root / "oft_reg", "oftv2", mode="asft", use_oft_regularizer=True)
        oft_plain_dir = run_tiny_train(root / "oft_plain", "oftv2", mode="sft", use_oft_regularizer=False)
        lora_dir = run_tiny_train(root / "lora", "lora", mode="sft", use_oft_regularizer=False)
        print("oftv2_reg_adapter_config", (oft_reg_dir / "adapter_config.json").exists())
        print("oftv2_plain_adapter_config", (oft_plain_dir / "adapter_config.json").exists())
        print("lora_adapter_config", (lora_dir / "adapter_config.json").exists())


if __name__ == "__main__":
    main()
