import torch

from models.srcnn import SRCNN
from models.espcn import ESPCN


def test_srcnn():
    print("=" * 50)
    print("Testing SRCNN")
    print("=" * 50)

    model = SRCNN()

    x = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        y = model(x)

    print(f"Input Shape : {x.shape}")
    print(f"Output Shape: {y.shape}")

    assert y.shape == x.shape

    print("✓ SRCNN passed!\n")


def test_espcn():
    print("=" * 50)
    print("Testing ESPCN")
    print("=" * 50)

    model = ESPCN()

    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        y = model(x)

    print(f"Input Shape : {x.shape}")
    print(f"Output Shape: {y.shape}")

    assert y.shape == (1, 3, 128, 128)

    print("✓ ESPCN passed!\n")


def main():
    test_srcnn()
    test_espcn()

    print("=" * 50)
    print("All model tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()