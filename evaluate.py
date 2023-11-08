# References:
    # https://wandb.ai/wandb_fc/korean/reports/-Frechet-Inception-distance-FID-GANs---Vmlldzo0MzQ3Mzc

import torch
import argparse
import numpy as np
import scipy

from utils import get_config
from inceptionv3 import InceptionV3


def _get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--img_size", type=int, required=False, default=32)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)
    parser.add_argument("--run_id", type=str, required=False)
    parser.add_argument("--torch_compile", action="store_true", required=False)

    args = parser.parse_args()
    return args


# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, use_torch=False):
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, use_torch=True):
    """
    mu1: (2048)
    sigma1: (2048, 2048)
    """

    if use_torch:
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        mu1 = torch.randn((2048,))
        mu2 = torch.randn((2048,))
        out = ((mu1 - mu2) ** 2).sum()

        sigma1 = torch.randn((2048, 2048))
        sigma2 = torch.randn((2048, 2048))

        (sigma1 * sigma2)

        sigma1 = np.random.rand(2048, 2048)
        sigma2 = np.random.rand(2048, 2048)
        a = scipy.linalg.sqrtm(sigma1 @ sigma2)
        sigma1 @ sigma2
        a
        
        a = np.cov(embed1, rowvar=False)
        b = torch.cov(torch.from_numpy(embed1).T)
        a.shape, b.shape
        a
        b
        
        embed2 = np.random.rand(4, 2048)



        diff = mu1 - mu2
        # Run 50 itrs of newton-schulz to get the matrix sqrt of
        # sigma1 dot sigma2
        covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50)
        if torch.any(torch.isnan(covmean)):
            return float('nan')
        covmean = covmean.squeeze()
        out = (diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)).cpu().item()
    # else:
    #     mu1 = np.atleast_1d(mu1)
    #     mu2 = np.atleast_1d(mu2)

    #     sigma1 = np.atleast_2d(sigma1)
    #     sigma2 = np.atleast_2d(sigma2)

    #     assert mu1.shape == mu2.shape, \
    #         'Training and test mean vectors have different lengths'
    #     assert sigma1.shape == sigma2.shape, \
    #         'Training and test covariances have different dimensions'

    #     diff = mu1 - mu2

    #     # Product might be almost singular
    #     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    #     if not np.isfinite(covmean).all():
    #         msg = ('fid calculation produces singular product; '
    #                'adding %s to diagonal of cov estimates') % eps
    #         print(msg)
    #         offset = np.eye(sigma1.shape[0]) * eps
    #         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    #     # Numerical error might give slight imaginary component
    #     if np.iscomplexobj(covmean):
    #         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
    #             m = np.max(np.abs(covmean.imag))
    #             raise ValueError('Imaginary component {}'.format(m))
    #         covmean = covmean.real

    #     tr_covmean = np.trace(covmean)

    #     out = (diff.dot(diff) +
    #            np.trace(sigma1) +
    #            np.trace(sigma2) -
    #            2 * tr_covmean)
    return out


if __name__ == "__main__":
    args = _get_args()
    CONFIG = get_config(args)

    model = InceptionV3().to(CONFIG["DEVICE"])
    model.eval()