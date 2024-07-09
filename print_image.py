import os

from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import make_grid


def create_image_strip(directory, output_path):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
        ]
    )

    images = []

    # ディレクトリ内のすべての画像ファイルを読み込む
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        image = read_image(image_path)
        image = transform(image)
        images.append(image)

    # すべての画像を横に並べた画像を生成
    if images:
        image_grid = make_grid(images, nrow=len(images))
        save_image = transforms.ToPILImage()(image_grid)
        save_image.save(output_path)
        print(f"Image strip saved to {output_path}")
    else:
        print("No images found in the directory.")


if __name__ == "__main__":
    directory = "/home/eto/fewshot-finetune-domain-adaptation/image/BacterialSpot/test/"
    output_path = "/home/eto/fewshot-finetune-domain-adaptation/image/gen/gen.png"
    create_image_strip(directory, output_path)
