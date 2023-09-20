import torch
import model

def interpolate_bottlenecks(model, image1, image2, n_steps):
    # Encode both input images to get the bottleneck tensors
    bottleneck1 = model.encode(image1)
    bottleneck2 = model.encode(image2)

    # Linearly interpolate between the two bottleneck tensors
    interpolated_bottlenecks = []
    for alpha in torch.linspace(0, 1, n_steps):
        interpolated_bottleneck = alpha * bottleneck1 + (1 - alpha) * bottleneck2
        interpolated_bottlenecks.append(interpolated_bottleneck)

    # Decode each interpolated bottleneck to get reconstructed images
    reconstructed_images = [model.decode(bottleneck) for bottleneck in interpolated_bottlenecks]

    return reconstructed_images

if __name__ == "__main__":
    n_steps = 0.19

    interpolate_bottlenecks(model,loader[])