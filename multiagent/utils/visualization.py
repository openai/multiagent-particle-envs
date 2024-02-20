import matplotlib.pyplot as plt

def visualize_image_observation(obs_grid):
    """
    Visualizes each channel of the observation grid using matplotlib.

    Parameters:
        obs_grid (numpy.ndarray): The observation grid with shape (grid_resolution, grid_resolution, num_channels).
    """
    
    num_channels = obs_grid.shape[2]
    channel_names = [
        "Agent's Position",
        "Agent's Field of Vision",
        "Other Agents' Positions",
        "Other Agents' Fields of Vision",
        "Obstacles",
        "Reward Values",
        "Reward Mask"
    ]

    fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 4, 4))
    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(obs_grid[:, :, i], cmap='gray', origin='lower')
        ax.set_title(channel_names[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()
