import torch
from aegan import Generator as G
import torchvision.utils as vutils
from display import display_images_in_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG = G()
netG.load_state_dict(torch.load('trained_generator_weights.pt', map_location=device))
vec = torch.randn((32, 16))
with torch.no_grad():
    gen = netG(vec)

for i in range(32):
    vutils.save_image(gen.data[i], f'output_dir/output.{i:02d}.png', normalize=True)

# Example usage:

output_dir = 'output_dir'
num_rows = 4
num_cols = 8
display_images_in_grid(output_dir, num_rows, num_cols)