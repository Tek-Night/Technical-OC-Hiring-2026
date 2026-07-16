import time
import torch

from models.srcnn import SRCNN
from models.espcn import ESPCN


def benchmark(model, input_tensor, device, runs=100):

    model.eval()

    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    avg_time = (end - start) / runs

    return avg_time * 1000  # milliseconds


def main():

    device = torch.device("cuda")

    srcnn = SRCNN()
    espcn = ESPCN()

    # SRCNN receives bicubic-upscaled image
    srcnn_input = torch.randn(
        1, 3, 128, 128
    )

    # ESPCN receives LR image
    espcn_input = torch.randn(
        1, 3, 32, 32
    )

    srcnn_time = benchmark(
        srcnn,
        srcnn_input,
        device
    )

    espcn_time = benchmark(
        espcn,
        espcn_input,
        device
    )

    print("="*40)
    print(f"SRCNN inference time: {srcnn_time:.4f} ms")
    print(f"ESPCN inference time: {espcn_time:.4f} ms")
    print("="*40)


if __name__ == "__main__":
    main()