"""Download model assets before running experiments."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT/'.cache/torch'))


def main():
    from torchvision.models import resnet18, ResNet18_Weights
    from transformers import AutoImageProcessor, AutoModel
    # Cache only public pretrained assets; no project images are uploaded.
    
    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    revisions = {'facebook/deit-tiny-patch16-224': 'b3428f18dcc7b543470d07f14b4a4157815d1880',
                 'microsoft/swin-tiny-patch4-window7-224': 'd00d478bfaf1f417d34a1186673ad4b4be264c51'}
    for name, revision in revisions.items():
        AutoImageProcessor.from_pretrained(name, revision=revision)
        options = {'add_pooling_layer': False} if 'deit' in name else {}
        AutoModel.from_pretrained(name, revision=revision, **options)
        print(f'Cached {name}', flush=True)


if __name__ == '__main__':
    main()
