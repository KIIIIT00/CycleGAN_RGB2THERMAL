"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import cv2
import numpy as np
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def generate_grad_cam(model, input_image, target_layer):
    """
    Generates a Grad-CAM heatmap for the given model and image.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        input_image (torch.Tensor): The input image.
        target_layer (str): The name of the layer for which Grad-CAM will be computed.

    Returns:
        grad_cam_heatmap (np.ndarray): The Grad-CAM heatmap.
    """
    
    activations = []
    gradients = []
    
    def save_activation_hook(module, input, output):
        activations.append(output)
    
    def save_gradient_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    target_layer = dict(model.netG.named_modules())[target_layer]
    # print("Target layer: {}".format(target_layer))
    
    if target_layer is None:
        print(f"Error: Target layer '{target_layer}' not found in the model.")
        return None
    
    # Register hooks
    hook = target_layer.register_backward_hook(save_gradient_hook)
    target_layer.register_forward_hook(save_activation_hook)
    target_layer.register_backward_hook(save_gradient_hook)
    
    # input_image = input_image.requires_grad_(True)  # 入力画像の勾配計算を許可

    # Forward pass
    output = model.netG(input_image)
    # print("Output:", output)
    # print("Output shape:", output.shape if output is not None else "None")

    # Debug: Check if output is valid
    if output is None:
        print("Error: Output is None, check the model's forward pass.")
        return None

    # Backward pass to get gradients
    model.netG.zero_grad()
    output.mean().backward()
    # print("INPUT",input_image.grad)
    
    # Debug: Check gradients and activations
    if not activations:
        print("Error: Activations list is empty. Check if the forward hook is being triggered.")
        return None

    if not gradients:
        print("Error: Gradients list is empty. Check if the backward hook is being triggered.")
        return None


    # Get the gradients and activations
    grad = gradients[0].cpu().data.numpy()  # (N, C, H, W)
    activation = activations[0].cpu().data.numpy()  # (N, C, H, W)

    # Compute the weights for each channel
    weights = np.mean(grad, axis=(2, 3), keepdims=True)  # (N, C, 1, 1)
    # grad_cam = torch.sum(weights * activations[0], dim=1, keepdim=True)  # 加重平均

    # Compute the Grad-CAM map
    grad_cam = np.sum(weights * activation, axis=1)  # (N, H, W)
    grad_cam = np.maximum(grad_cam, 0)  # ReLU activation
    grad_cam = cv2.resize(grad_cam[0], (input_image.shape[3], input_image.shape[2]))  # Resize to input size

    # Normalize the heatmap to [0, 1]
    grad_cam = grad_cam / np.max(grad_cam)
    return grad_cam

def overlay_grad_cam(heatmap, original_image):
    """
    Overlays the Grad-CAM heatmap on top of the original image.

    Parameters:
        heatmap (np.ndarray): The Grad-CAM heatmap.
        original_image (np.ndarray): The original input image.

    Returns:
        overlay (np.ndarray): The Grad-CAM overlayed on the image.
    """
    # Convert the heatmap to a color map
    if heatmap is not None:
        # print("Heatmap Shape:", heatmap.shape)
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0
    else:
        print("Heatmap is None, cannot apply colormap.")

    # Convert the original image to float32
    original_image = np.float32(original_image) / 255.0

    # Overlay the heatmap on the original image
    overlay = 0.7 * original_image + 0.3 * heatmap
    overlay = np.uint8(255 * overlay)
    
    return overlay

if __name__ == '__main__':
    # メモリ解放
    torch.cuda.empty_cache()
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.num_test = 1386 # テストの回数
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)
    # print("Model dict:", dict(model.netG.named_modules()).keys())
    # TensorBoard Log DIr
    log_dir = os.path.join('./logs/CycleGAN', opt.name, opt.phase)
    writer = SummaryWriter(log_dir=log_dir)

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        
    total_cycle_loss = 0.0
    num_images = 0

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        # model.set_input(data)  # unpack data from data loader
        # model.test()           # run inference


        model.set_input(data)  # unpack data from data loader
        model.eval() 
            
        # Grad-CAM
        # input_image = data['A'].to(model.device).requires_grad_(True)
        # grad_cam = generate_grad_cam(model, input_image, target_layer='module.model.26')
            
        # del input_image
        # del data
        # torch.cuda.empty_cache()
        
        # Calculate cycle consistency loss
        model.forward()        # perform forward pass to calculate losses
        losses = model.get_current_losses()
        print("Cycle Consistency Loss:", losses.get('cycle_consistency'))
        total_cycle_loss += losses.get('cycle_consistency', 0.0)
        num_images += 1

        visuals = model.get_current_visuals()  # get image results
        # real_image = visuals['real']
        # real_image = real_image[0]
        # original_image = real_image.cpu().detach().numpy().transpose(1, 2, 0)
        # original_image = (original_image + 1) / 2.0
        # overlay_image = overlay_grad_cam(grad_cam, original_image)
        # cv2.imwrite(f'{opt.results_dir}/grad_cam_{i}.png', overlay_image)
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        
        if i % 55 == 0: # save Grad-CAM images
            file_name = [Path(path).stem for path in img_path]
            print(file_name[0])
            target_layer_name = 'module.model.1'
            # Grad-CAM
            input_image = data['A'].to(model.device).requires_grad_(True)
            grad_cam = generate_grad_cam(model, input_image, target_layer=target_layer_name)
            del input_image
            del data
            torch.cuda.empty_cache()
            
            real_image = visuals['real']
            real_image = real_image[0]
            original_image = real_image.cpu().detach().numpy().transpose(1, 2, 0)
            original_image = (original_image + 1) / 2.0
            overlay_image = overlay_grad_cam(grad_cam, original_image)
            cv2.imwrite(f'{opt.results_dir}/grad_cam_{file_name[0]}_{target_layer_name}.jpg', overlay_image)
            
            # Log Grad-CAM and overlay to TensorBoard
            writer.add_image(f"Grad-CAM/{file_name[0]}_{target_layer_name}", overlay_image.transpose(2, 0, 1), i, dataformats="CHW")
            
        
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        
        writer.add_scalar('Loss/cycle_consistency', losses.get('cycle_consistency', 0.0), i)
        for key, img in visuals.items():
            if img.dim() == 4:  
                img = img[0]  # バッチ次元を削除
            # 次元を [C, H, W] → [H, W, C] に変更
            img_np = img.permute(1, 2, 0).detach().cpu().numpy()
            # ピクセル値を [0, 1] の範囲にスケール
            img_np = (img_np + 1) / 2.0  # 正規化
            writer.add_image(f'Image/{key}', img_np, i, dataformats="HWC")
            
         
    if num_images > 0:
        average_cycle_loss = total_cycle_loss / num_images
        print(f'Average Cycle Consistency Loss: {average_cycle_loss}')
        writer.add_scalar('Loss/average_cycle_consistency', average_cycle_loss, 0)
        
    webpage.save()  # save the HTML
    writer.close()
