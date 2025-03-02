import torch
import cv2
import numpy as np
from  torchvision.transforms import ToPILImage

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
        
    def hook_layers(self):
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, class_index = None):
        output = self.model(input_image)
        
        if class_index is None:
            class_index = torch.argmax(output)
        score = output[:, class_index]
        
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
        heatmap = cv2.resize(heatmap, (input_image.shape[2], input_image.shape[3]))
        heatmap = heatmap / heatmap.max()  # 正規化
        return heatmap
    
    def apply_heatmap(self, input_image, heatmap, alpha=0.4):
        """
        入力画像にヒートマップを重ね合わせる
        Args:
            input_image (torch.Tensor): 入力画像（1, C, H, Wのテンソル）
            heatmap (np.array): Grad-CAMで生成したヒートマップ
            alpha (float): ヒートマップの透明度
        Returns:
            np.array: 重ね合わせた画像
        """
        # 入力画像をnumpy配列に変換
        input_image = input_image.squeeze().cpu().numpy().transpose(1, 2, 0)
        input_image = np.uint8(255 * (input_image - input_image.min()) / (input_image.max() - input_image.min()))

        # ヒートマップをカラーにして重ね合わせ
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(input_image, 1 - alpha, heatmap_colored, alpha, 0)
        return superimposed_img