import os

import torch.nn.functional

from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# Data
data_folder = 'ex_data'
test_data_names = ["ex_data"]
out_dir = f'{data_folder}_generated'
os.makedirs(out_dir, exist_ok=True)
# Model checkpoints
srgan_checkpoint = "checkpoints/checkpoint_srgan.pth.tar"

# Load model, either the SRResNet or the SRGAN
# srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
# srresnet.eval()
# model = srresnet
srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
srgan_generator.eval()
model = srgan_generator


def save_image(lr_imgs, sr_imgs_y, hr_imgs_y, save_path):
    w = int((hr_imgs_y.shape[0] - lr_imgs.shape[0]) / 2)
    h = int((hr_imgs_y.shape[1] - lr_imgs.shape[1]) / 2)
    lr_imgs = torch.nn.functional.pad(lr_imgs, (h, h, w, w), value=16)
    im_concat = torch.cat((hr_imgs_y, lr_imgs, sr_imgs_y), dim=1)

    im_concat = transforms.ToPILImage()(im_concat)
    draw = ImageDraw.Draw(im_concat, 'L')
    font = ImageFont.truetype('arial')
    draw.text((sr_imgs_y.shape[1] * 0.1, 5), "original", (0), font=font)
    draw.text((sr_imgs_y.shape[1] * 1.1, 5), "low resolution", (0), font=font)
    draw.text((sr_imgs_y.shape[1] * 2.1, 5), "generated image", (0), font=font)
    im_concat.save(save_path)


# Evaluate
for test_data_name in test_data_names:
    print("\nFor %s:\n" % test_data_name)

    # Custom dataloader
    test_dataset = SRDataset(data_folder,
                             mode='test',
                             crop_size=0,
                             scaling_factor=2,
                             lr_img_type='[-1, 1]',
                             hr_img_type='[-1, 1]',
                             test_data_name=test_data_name,
                             eval=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                              pin_memory=True)

    # Keep track of the PSNRs and the SSIMs across batches
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

    # Prohibit gradient computation explicitly because I had some problems with memory
    ssims = []
    with torch.no_grad():
        # Batches
        for i, (lr_imgs, hr_imgs, lr_imgs_orig) in enumerate(test_loader):
            # Move to default device
            # if i > 20:
            #     break
            # lr_img = lr_imgs.resize((int(hr_imgs.width), int(hr_imgs.height)), Image.BICUBIC)

            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            lr_imgs_orig = lr_imgs_orig.to(device)

            # Forward prop.
            sr_imgs = model(lr_imgs)
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
            lr_imgs_orig_y = convert_image(lr_imgs_orig, source='[-1, 1]', target='y-channel').squeeze(0)

            save_image(lr_imgs_orig_y, sr_imgs_y, hr_imgs_y, os.path.join(out_dir, f'{i:05}.png'))
