from PIL import Image
import os

image_dir = "/Users/tov/Library/Mobile Documents/com~apple~CloudDocs/personal website/DoraemonJack.github.io/img/deep-learning-math/"

png_files = [
    "01_perceptron_vs_xor.png",
    "02_activation_functions.png",
    "03_gradient_vanishing.png",
    "04_loss_functions.png",
    "05_learning_rate_effect.png",
    "06_sgd_convergence.png",
    "07_mlp_architecture.png",
]

print("转换为 WebP 格式（更小，加载更快）...\n")

for filename in png_files:
    filepath = os.path.join(image_dir, filename)
    if os.path.exists(filepath):
        img = Image.open(filepath)
        webp_filepath = filepath.replace('.png', '.webp')
        
        # 转换为 WebP
        img.save(webp_filepath, 'WEBP', quality=85)
        
        png_size = os.path.getsize(filepath)
        webp_size = os.path.getsize(webp_filepath)
        reduction = (1 - webp_size / png_size) * 100
        
        print(f"✓ {filename}")
        print(f"  PNG:  {png_size/1024:.0f}KB")
        print(f"  WebP: {webp_size/1024:.0f}KB (节省 {reduction:.1f}%)")
        print()

