import os

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
out_dir = 'eval_results'
os.makedirs(out_dir, exist_ok=True)
# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
srgan_generator.eval()
model = srgan_generator

# Evaluate
for test_data_name in test_data_names:
    print("\nFor %s:\n" % test_data_name)

    data_type = '[-1, 1]'
    # Custom dataloader
    test_dataset = SRDataset(data_folder,
                             mode='test',
                             crop_size=0,
                             scaling_factor=2,
                             lr_img_type='[-1, 1]',
                             hr_img_type='[-1, 1]',
                             test_data_name=test_data_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                              pin_memory=True)

    # Keep track of the PSNRs and the SSIMs across batches
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

    # Prohibit gradient computation explicitly because I had some problems with memory
    ssims = []
    with torch.no_grad():
        # Batches
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
            # Move to default device
            # if i > 20:
            #     break
            lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

            # Forward prop.
            sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

            # Calculate PSNR and SSIM
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel

            diff = convert_image(torch.abs(hr_imgs - sr_imgs), source='[0, 1]', target='y-channel').squeeze(0)


            def save_image(sr_imgs_y, hr_imgs_y, diff, ssim, save_path):
                im_concat = torch.cat((sr_imgs_y, hr_imgs_y, diff), dim=1)
                im_concat = transforms.ToPILImage()(im_concat)
                draw = ImageDraw.Draw(im_concat, 'L')
                font = ImageFont.truetype('arial')
                draw.text((sr_imgs_y.shape[1] * 0.1, 5), "original", (0), font=font)
                draw.text((sr_imgs_y.shape[1] * 1.1, 5), "generated", (0), font=font)
                draw.text((sr_imgs_y.shape[1] * 2.1, 5), f"difference: ssmi={ssim:.4f}", (0), font=font)
                # im_concat.show()
                im_concat.save(save_path)


            psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
            ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
            ssims.append(ssim)
            save_image(sr_imgs_y, hr_imgs_y, diff, ssim, os.path.join(out_dir, f'{i:05}.png'))

            PSNRs.update(psnr, lr_imgs.size(0))
            SSIMs.update(ssim, lr_imgs.size(0))

    # Print average PSNR and SSIM
    print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
    print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

print("\n")
