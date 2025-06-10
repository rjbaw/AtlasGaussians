#!/usr/bin/env python3
"""Calculates the Kernel Inception Distance (KID) to evalulate GANs
"""
import os
import json
import pathlib
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
torch.backends.cudnn.enabled = False
from sklearn.metrics.pairwise import polynomial_kernel
from scipy import linalg
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
import ipdb
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x
import cv2
from models.inception import InceptionV3
from models.lenet import LeNet5
import glob
import pathlib

def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False,reso=128):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    is_numpy = True if type(files[0]) == np.ndarray else False

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size
        if is_numpy:
            images = np.copy(files[start:end]) + 1
            images /= 2.
        else:
            images=[]
            #ipdb.set_trace()
            for f in files[start:end]:
                try:
                    img=cv2.imread(str(f))
                    #if img.mean(-1)>254.9:
                    #img[np.where(img.mean(-1)>254.9)]=0
                    img=cv2.resize(img,(reso,reso),interpolation=cv2.INTER_CUBIC)
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                except:
                    img=cv2.imread(str(files[0]))
                    #if img.mean(-1)>254.9:
                    #img[np.where(img.mean(-1)>254.9)]=0
                    img=cv2.resize(img,(reso,reso),interpolation=cv2.INTER_CUBIC)
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    print(str(f))
                #ipdb.set_trace()
                images.append(img)
            #ipdb.set_trace()


            #images = [np.array(Image.open(str(f)).convert('RGB')) for f in files[start:end]]
            images = np.stack(images).astype(np.float32) / 255.
            # Reshape to (n_images, 3, height, width)
            images = images.transpose((0, 3, 1, 2))
            #ipdb.set_trace()
            

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print('done', np.min(images))

    return pred_arr


def extract_lenet_features(imgs, net):
    net.eval()
    feats = []
    imgs = imgs.reshape([-1, 100] + list(imgs.shape[1:]))
    if imgs[0].min() < -0.001:
      imgs = (imgs + 1)/2.0
    print(imgs.shape, imgs.min(), imgs.max())
    imgs = torch.from_numpy(imgs)
    for i, images in enumerate(imgs):
        feats.append(net.extract_features(images).detach().cpu().numpy())
    feats = np.vstack(feats)
    return feats


def _compute_activations(path, model, batch_size, dims, cuda, model_type,reso,dataset):
    basepath=f'./objaverse-results/gt/kid{reso}'
    os.makedirs(os.path.join(basepath), exist_ok=True)
    if not os.path.exists(os.path.join(basepath,path.split('/')[-1]+str(reso)+'kid.npy')):
        import glob
        import pathlib
        path = pathlib.Path(path)
        if not type(path) == np.ndarray:
            
            if 'objaverse' in dataset:

                files=[]
                path = pathlib.Path(path)

                with open('../../datasets/splits/objaverse/eval250.txt', 'r') as f:
                    obj_ids = f.read().splitlines()

                files = []
                for fidx in obj_ids:
                    for v in range(24):
                        files.append(f"{path}/{fidx}/{v:05d}/{v:05d}.png")
                assert len(files) == 24 * len(obj_ids)

            elif 'shapenet' in dataset:
                from pathlib import Path
                root = Path(path)
                # gather only real PNGs under either rgb/ or images/
                rgb_pngs  = list(root.glob('**/rgb/*.png'))
                imgs_pngs = list(root.glob('**/images/*.png'))
                files = [str(p) for p in sorted(rgb_pngs + imgs_pngs)]
                if not files:
                    raise RuntimeError(f"No .png files found under {path}")
                os.makedirs(basepath, exist_ok=True)
                # directly compute the raw activations for KID
                act = get_activations(files, model, batch_size, dims, cuda, reso=reso)
                return act

            else:
                raise NotImplementedError
    
    #ipdb.set_trace()
    if model_type == 'inception':
        if os.path.exists(os.path.join(basepath,str(reso)+'kid.npy')):
            act=np.load(os.path.join(basepath,str(reso)+'kid.npy'))
            print('load_dataset',dataset)
        else:
            act = get_activations(files, model, batch_size, dims, cuda,reso=reso)
            np.save(os.path.join(basepath,str(reso)+'kid'),act)
    elif model_type == 'lenet':
        act = extract_lenet_features(files, model)
    return act


def _compute_activations_new(path, model, batch_size, dims, cuda, model_type,reso,dataset, basepath=None):
    if 'shapenet' in dataset:
        from pathlib import Path
        root = Path(path)
        rgb_pngs  = list(root.glob('**/rgb/*.png'))
        imgs_pngs = list(root.glob('**/*.png'))
        files = [str(p) for p in sorted(rgb_pngs + imgs_pngs)]
        if not files:
            raise RuntimeError(f"No .png files found under {path}")
        os.makedirs(basepath, exist_ok=True)
        # compute KID activations directly
        act = get_activations(files, model, batch_size, dims, cuda, reso=reso)
        return act
    sample_name=path.split('/')[-1]
    print(f"path={path}")
    print(f"basepath={basepath}")

    with open('../../datasets/splits/objaverse/eval250.txt', 'r') as f:
        obj_ids = f.read().splitlines()

    files = []
    if 'lgm' in basepath or 'shape' in basepath or 'gvgen' in basepath:
        seed = 0
    elif 'ours' in basepath:
        seed = 5
    elif 'ln3diff' in basepath:
        seed = 41
    else:
        raise NotImplementedError

    text_fpath = f"/scratch/cluster/yanght/Projects/AliGeoReg/Dataset/Objaverse/gobjaverse_cap3d/text_captions_cap3d.json"
    texts = json.load(open(text_fpath, 'r'))
    for test_idx, obj_idx in enumerate(obj_ids):
        text = texts[obj_idx]
        text_prefix = '_'.join(text.split()).replace('/', '=')                                                                                                                                                                                                         
        for v in range(24):
            files.append(f"{path}/seed{seed}_{test_idx:05d}-{text_prefix}-{v}.png")
    assert len(files) == 24 * len(obj_ids)

    os.makedirs(os.path.join(basepath), exist_ok=True)

    #ipdb.set_trace()
    if model_type == 'inception':
        if os.path.exists(os.path.join(basepath,sample_name+str(reso)+'kid.npy')):
            act=np.load(os.path.join(basepath,sample_name+str(reso)+'kid.npy'))
            print('load_sample')
        else:
            act = get_activations(files, model, batch_size, dims, cuda,reso=reso)
            np.save(os.path.join(basepath,sample_name+str(reso)+'kid'),act)
    elif model_type == 'lenet':
        act = extract_lenet_features(files, model)
    #ipdb.set_trace()
    return act

def calculate_kid_given_paths(paths, batch_size, cuda, dims, model_type='inception',reso=128,dataset='omni', basepath=None):
    """Calculates the KID of two paths"""
    pths = []
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
        if os.path.isdir(p):
            pths.append(p)
        # elif p.endswith('.npy'):
        #     np_imgs = np.load(p)
        #     if np_imgs.shape[0] > 50000: np_imgs = np_imgs[np.random.permutation(np.arange(np_imgs.shape[0]))][:50000]
        #     pths.append(np_imgs)

    if model_type == 'inception':
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])
    elif model_type == 'lenet':
        model = LeNet5()
        model.load_state_dict(torch.load('./models/lenet.pth'))
    if cuda:
       model.cuda()

    act_true = _compute_activations(pths[0], model, batch_size, dims, cuda, model_type,reso,dataset)
    pths = pths[1:]
    results = []
    #ipdb.set_trace()
    for j, pth in enumerate(pths):
        print(paths[j+1])
        actj = _compute_activations_new(pth, model, batch_size, dims, cuda, model_type,reso,dataset, basepath)
        #ipdb.set_trace()
        kid_values = polynomial_mmd_averages(act_true, actj, n_subsets=100)
        results.append((paths[j+1], kid_values[0].mean(), kid_values[0].std()))
    return results

def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                            ret_var=True, output=sys.stdout, **kernel_args):
    # m = min(codes_g.shape[0], codes_r.shape[0])
    m = min(codes_g.shape[0], codes_r.shape[0])
    if subset_size > m:
        subset_size = m

    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice
    #ipdb.set_trace()

    with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
        for i in bar:
            
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return (mmds, vars) if ret_var else mmds


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)

def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--true', type=str, required=True,
                        help=('Path to the true images'))
    parser.add_argument('--fake', type=str, nargs='+', required=True,
                        help=('Path to the generated images'))
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size to use')
    parser.add_argument('--reso', type=int, default=128,
                        help='Batch size to use')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('-c', '--gpu', default='', type=str,
                        help='GPU to use (leave blank for CPU only)')
    parser.add_argument('--model', default='inception', type=str,
                        help='inception or lenet')
    parser.add_argument('--dataset', default='omni', type=str,
                        help='inception or lenet')
    parser.add_argument('--basepath', type=str, help='path to prediction stats')
    args = parser.parse_args()
    print(args)
    #ipdb.set_trace()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    paths = [args.true] + args.fake
    

    results = calculate_kid_given_paths(paths, args.batch_size,True, args.dims, model_type=args.model,reso=args.reso,dataset=args.dataset, basepath=args.basepath)
    for p, m, s in results:
        print('KID (%s): %.6f (%.6f)' % (p, m, s))
