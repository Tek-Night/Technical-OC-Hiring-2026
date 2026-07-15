import json
import os

import matplotlib.pyplot as plt
from PIL import Image

from data.degradation import degrade_image


def load_config(config_path):
    """Load a JSON configuration file."""
    with open(config_path, "r") as file:
        return json.load(file)


def main():
    # -------------------------------
    # Paths
    # -------------------------------
    config_path = "configs/espcn_x4.json"

    # Change this to any image you want to test
    image_path = "dataset/raw_hr/0010.png"

    output_dir = "outputs"
    output_path = os.path.join(output_dir, "degradation_test.png")

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------
    # Load configuration
    # -------------------------------
    config = load_config(config_path)

    # -------------------------------
    # Load HR image
    # -------------------------------
    hr_image = Image.open(image_path).convert("RGB")

    # -------------------------------
    # Apply degradation pipeline
    # -------------------------------
    lr_image = degrade_image(hr_image, config)

    # -------------------------------
    # Print useful information
    # -------------------------------
    print("=" * 40)
    print("Degradation Test")
    print("=" * 40)
    print(f"Original Size : {hr_image.size}")
    print(f"Degraded Size : {lr_image.size}")
    print(f"Scale Factor  : {config['model']['scale_factor']}")
    print("=" * 40)

    # -------------------------------
    # Display comparison
    # -------------------------------
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(hr_image)
    plt.title("High Resolution")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(lr_image)
    plt.title("Low Resolution")
    plt.axis("off")

    plt.tight_layout()

    # -------------------------------
    # Save comparison
    # -------------------------------
    plt.savefig(output_path, dpi=300)
    print(f"Comparison image saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()