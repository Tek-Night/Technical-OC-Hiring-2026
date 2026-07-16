import os

import matplotlib.pyplot as plt


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor (C,H,W) to a NumPy image (H,W,C).
    """

    image = tensor.detach().cpu().permute(1, 2, 0).numpy()

    image = image.clip(0, 1)

    return image


def save_prediction_images(
    lr,
    prediction,
    hr,
    epoch,
    output_dir="outputs"
):
    """
    Save LR, Prediction and HR side-by-side.
    """

    os.makedirs(output_dir, exist_ok=True)

    lr = tensor_to_image(lr)
    prediction = tensor_to_image(prediction)
    hr = tensor_to_image(hr)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(lr)
    plt.title("Low Resolution")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(prediction)
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(hr)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"epoch_{epoch:03d}.png"
        )
    )

    plt.close()