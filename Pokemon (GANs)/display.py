import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_images_in_grid(output_dir, num_rows, num_cols):
    # Get a list of image files in the directory
    image_files = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Create a subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Iterate over the image files and display them in the grid
    for i, ax in enumerate(axes.flat):
        if i < len(image_files):
            img = mpimg.imread(os.path.join(output_dir, image_files[i]))
            ax.imshow(img)
            ax.axis('off')
        else:
            # If there are fewer images than grid cells, remove the empty cells
            fig.delaxes(ax)

    # Adjust spacing and display the grid
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


