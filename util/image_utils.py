import torch
from einops import rearrange

def compute_psnr(img1, img2):
    ''' Compute psnr, both [B, C, H, W] and [B, H, W, C] are supported.
    Args:
        img1/img2: [B, X, Y, Z] or [B, V, X, Y, Z]
    Returns:
        psnr: [B,] or [B, V]
    '''
    assert(img1.shape == img2.shape)
    if len(img1.shape) == 5:
        # x, y, z can be C, H, W or H, W, C
        img1 = rearrange(img1, 'b v x y z -> (b v) x y z')
        img2 = rearrange(img2, 'b v x y z -> (b v) x y z')
    else:
        assert(len(img1.shape) == 4)

    B = img1.shape[0]
    mse = (((img1 - img2)) ** 2).reshape(B, -1).mean(dim=-1)  # [B,] or [B*V,]
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    if len(img1.shape) == 5:
        psnr = rearrange(psnr, '(b v) -> b v', b=B)  # [B, V]

    return psnr



