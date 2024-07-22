# ViViT_Small
Per utilizzare il modello:

````python
from vit_models.vivit_small_datasets import ViT as VITSmall
model = VITSmall(
    image_size = 224,
    patch_size = 16,
    num_classes = 24,
    dim = 64,
    depth = 4,
    temporal_depth = 4,
    heads = 4,
    mlp_dim = 32,
    channels = 3,
    dropout = 0.1,
    emb_dropout = 0.1,
)

model.cuda()
img = torch.rand(batch, 3, 224, 224)
out = model(img) # (batch, 24)

````
