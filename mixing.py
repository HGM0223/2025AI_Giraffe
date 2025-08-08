import sys, os

sys.path.append(os.path.abspath('./code'))
sys.path.append(os.path.abspath('./stylegan3'))

print("[DEBUG] PYTHONPATH 설정 완료:", sys.path)

from encoder.inference_utils import infer_image_to_latent
import torch
import numpy as np
import PIL.Image
from typing import List, Optional
import dnnlib
import legacy


class StyleGAN3StyleMixer:
    def __init__(self, giraffe_model_path: str, human_model_path: str, device: str = 'cpu'):
        self.device = device
        self.giraffe_G = self.load_model(giraffe_model_path)
        self.human_G = self.load_model(human_model_path)
        self.z_dim = self.giraffe_G.z_dim
        self.w_dim = self.giraffe_G.w_dim
        print(f"모델 로드 완료: z_dim={self.z_dim}, w_dim={self.w_dim}")

    def load_model(self, model_path: str):
        print(f"모델 로드 중: {model_path}")
        with open(model_path, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
        return G

    def encode_image_to_w(self, image_path: str) -> torch.Tensor:
        print(f"사람 얼굴 인코딩 중: {image_path}")
        latent = infer_image_to_latent(image_path, device=self.device)
        return latent

    def generate_w_codes(self, num_samples: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        z = torch.randn(num_samples, self.z_dim, device=self.device)
        w = self.giraffe_G.mapping(z, None)
        return w

    def style_mixing(self, w_giraffe, w_human, mixing_layers: List[int], resolution: int = 512) -> PIL.Image.Image:
        num_layers = self.giraffe_G.synthesis.num_ws
        w_giraffe_full = w_giraffe.unsqueeze(1).expand(-1, num_layers, -1) if w_giraffe.ndim == 2 else w_giraffe
        w_human_full = w_human.unsqueeze(1).expand(-1, num_layers, -1) if w_human.ndim == 2 else w_human

        w_mixed = w_giraffe_full.clone()
        for layer_idx in mixing_layers:
            if layer_idx < num_layers:
                w_mixed[:, layer_idx] = w_human_full[:, layer_idx]

        with torch.no_grad():
            img = self.giraffe_G.synthesis(w_mixed, noise_mode='const')
        return self._postprocess_image(img, resolution)

    def _postprocess_image(self, img: torch.Tensor, resolution: int) -> PIL.Image.Image:
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()
        if img.shape[:2] != (resolution, resolution):
            img = np.array(PIL.Image.fromarray(img).resize((resolution, resolution), PIL.Image.LANCZOS))
        return PIL.Image.fromarray(img)

    def save_image(self, img: PIL.Image.Image, path: str):
        img.save(path)
        print(f"이미지 저장됨: {path}")

def main():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    giraffe_model_path = os.path.join(base_path, 'giraffe_model.pkl')
    human_model_path = os.path.join(base_path, 'human_model.pkl')
    person_image_path = os.path.join(base_path, 'jk_test.jpg')
    output_image_path = os.path.join(base_path, 'giraffe_with_human_eyes.png')

    mixer = StyleGAN3StyleMixer(giraffe_model_path, human_model_path, device='cuda' if torch.cuda.is_available() else 'cpu')

    w_giraffe = mixer.generate_w_codes(1, seed=2)
    w_human = mixer.encode_image_to_w(person_image_path)

    eye_layers = [6, 7, 8]
    mixed_img = mixer.style_mixing(w_giraffe, w_human, eye_layers)
    mixer.save_image(mixed_img, output_image_path)

    print("🎉 혼합 완료: 기린 + 사람 눈")

if __name__ == "__main__":
    main()
