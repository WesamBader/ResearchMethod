
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Chapter 14: Convolutional Neural Networks

ROOT = Path(__file__).resolve().parent

# STEP 1: Create a 5 x 5 black-and-white image
# 1 = foreground, 0 = background

image = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

print("\nCHAPTER 14: CONVOLUTION")
print("\nOriginal Image:")
print(image)

# STEP 2: Create convolution filters

filters = {
    "Horizontal": np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ]),
    "Vertical": np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ]),
    "Diagonal": np.array([
        [-1, -1, 1],
        [-1, 2, -1],
        [1, -1, -1]
    ]),
    "Junction": np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
}

# STEP 3: Apply convolution and ReLU

def apply_filter(input_image, kernel):
    # Boundary padding keeps the output the same size.
    # convolve2d performs mathematical convolution,
    # so the kernel is rotated internally.
    raw_map = convolve2d(
        input_image,
        kernel,
        mode="same",
        boundary="fill",
        fillvalue=0
    )

    relu_map = np.maximum(0, raw_map)
    return raw_map, relu_map


# STEP 4: Create and save visualizations

def save_plot(data, title, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(data, cmap="viridis", interpolation="nearest")
    plt.title(title)
    plt.colorbar(label="Pixel Value / Activation")
    plt.xticks(range(data.shape[1]))
    plt.yticks(range(data.shape[0]))
    plt.tight_layout()
    plt.savefig(ROOT / filename, dpi=300)
    plt.show()
    plt.close()


save_plot(image, "Original Plus Sign", "original_image.png")

for name, kernel in filters.items():

    raw, activated = apply_filter(image, kernel)

    print(f"\n{name.upper()} FILTER")
    print("Kernel:")
    print(kernel)

    print("\nRaw Feature Map:")
    print(raw)

    print("\nAfter ReLU:")
    print(activated)

    print("Highest activation:", raw.max())
    print("Lowest activation:", raw.min())

    save_plot(
        activated,
        f"{name} Filter Feature Map",
        f"{name.lower()}_feature_map.png"
    )

# STEP 5: Compare all filters

fig, axes = plt.subplots(1, 5, figsize=(16, 4))

axes[0].imshow(image, cmap="gray_r")
axes[0].set_title("Original")

for ax, (name, kernel) in zip(
    axes[1:], filters.items()
):
    _, result = apply_filter(image, kernel)
    ax.imshow(result, cmap="viridis")
    ax.set_title(name)

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(ROOT / "filter_comparison.png", dpi=300)
plt.show()
plt.close()

# STEP 6: Build a two-layer feature detector

horizontal_map = apply_filter(
    image, filters["Horizontal"]
)[1]

second_layer = apply_filter(
    horizontal_map, filters["Vertical"]
)[1]

print("\nSECOND-LAYER FEATURE MAP")
print(second_layer)

save_plot(
    second_layer,
    "Two-Layer Feature Detection",
    "two_layer_feature_map.png"
)

# STEP 7: Save numerical results

np.savetxt(
    ROOT / "original_matrix.csv",
    image,
    delimiter=",",
    fmt="%d"
)

np.savetxt(
    ROOT / "second_layer_matrix.csv",
    second_layer,
    delimiter=",",
    fmt="%d"
)

print("\nPROJECT COMPLETE")
print("All graphs and matrices saved to:")
print(ROOT)
