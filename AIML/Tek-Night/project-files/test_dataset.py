import json
import matplotlib.pyplot as plt
from pathlib import Path

from data.dataset import SuperResolutionDataset


def load_config(path):
    with open(path, "r") as file:
        return json.load(file)


def main():

    PROJECT_ROOT = Path(__file__).resolve().parent
    config = load_config(PROJECT_ROOT/"configs"/"espcn_x4.json")

    dataset = SuperResolutionDataset(
        image_dir=PROJECT_ROOT/"dataset"/"train",
        config=config,
        train=True
    )

    print("=" * 50)
    print("Testing SuperResolutionDataset")
    print("=" * 50)

    print(f"Dataset Size : {len(dataset)}")

    lr_tensor, hr_tensor = dataset[0]

    print(f"\nLR Tensor Shape : {lr_tensor.shape}")
    print(f"HR Tensor Shape : {hr_tensor.shape}")

    print(f"\nLR Min : {lr_tensor.min():.3f}")
    print(f"LR Max : {lr_tensor.max():.3f}")

    print(f"\nHR Min : {hr_tensor.min():.3f}")
    print(f"HR Max : {hr_tensor.max():.3f}")

    # Convert tensors back to images for visualization
    lr_image = lr_tensor.permute(1, 2, 0).numpy()
    hr_image = hr_tensor.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(lr_image)
    plt.title("Low Resolution")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hr_image)
    plt.title("High Resolution")
    plt.axis("off")

    plt.tight_layout()

    plt.show()

    print("\nDataset test passed!")


if __name__ == "__main__":
    main()