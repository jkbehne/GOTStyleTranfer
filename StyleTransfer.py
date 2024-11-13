import argparse as ap
from copy import deepcopy
import os
from typing import Callable, Tuple, List
from PIL import Image
import time

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchinfo import summary

from vgg_normalised_conv1_1 import vgg_normalised_conv1_1
from vgg_normalised_conv2_1 import vgg_normalised_conv2_1
from vgg_normalised_conv3_1 import vgg_normalised_conv3_1
from vgg_normalised_conv4_1 import vgg_normalised_conv4_1
from vgg_normalised_conv5_1 import vgg_normalised_conv5_1
from feature_invertor_conv1_1 import feature_invertor_conv1_1
from feature_invertor_conv2_1 import feature_invertor_conv2_1
from feature_invertor_conv3_1 import feature_invertor_conv3_1
from feature_invertor_conv4_1 import feature_invertor_conv4_1
from feature_invertor_conv5_1 import feature_invertor_conv5_1

class VGGEncDec(nn.Module):
    def __init__(self, args):
        super(VGGEncDec, self).__init__()
        # Create objects
        self.e1 = vgg_normalised_conv1_1
        self.e2 = vgg_normalised_conv2_1
        self.e3 = vgg_normalised_conv3_1
        self.e4 = vgg_normalised_conv4_1
        self.e5 = vgg_normalised_conv5_1
        self.d1 = feature_invertor_conv1_1
        self.d2 = feature_invertor_conv2_1
        self.d3 = feature_invertor_conv3_1
        self.d4 = feature_invertor_conv4_1
        self.d5 = feature_invertor_conv5_1

        # Set up pre-trained weights
        self.e1.load_state_dict(torch.load(args.vgg1))
        self.e1.eval()
        self.e2.load_state_dict(torch.load(args.vgg2))
        self.e2.eval()
        self.e3.load_state_dict(torch.load(args.vgg3))
        self.e3.eval()
        self.e4.load_state_dict(torch.load(args.vgg4))
        self.e4.eval()
        self.e5.load_state_dict(torch.load(args.vgg5))
        self.e5.eval()

        self.d1.load_state_dict(torch.load(args.decoder1))
        self.d1.eval()
        self.d2.load_state_dict(torch.load(args.decoder2))
        self.d2.eval()
        self.d3.load_state_dict(torch.load(args.decoder3))
        self.d3.eval()
        self.d4.load_state_dict(torch.load(args.decoder4))
        self.d4.eval()
        self.d5.load_state_dict(torch.load(args.decoder5))
        self.d5.eval()

class GOTTransform:
    def __init__(
        self,
        A: torch.Tensor,
        muc: torch.Tensor,
        mus: torch.Tensor,
        device: torch.device,
    ) -> None:
        self._A = A
        self._muc = muc.cpu().unsqueeze(1)
        self._mus = mus.cpu().unsqueeze(1)
        self._device = device

    def __call__(
        self,
        F: torch.Tensor,
    ) -> torch.Tensor:
        C, H, W = F.size()
        N = H * W
        Fr = torch.reshape(F, (C, N))
        Fc = Fr - self._muc.repeat(1, N).to(self._device)
        Ft = torch.matmul(self._A, Fc) + self._mus.repeat(1, N).to(self._device)
        return torch.reshape(Ft, (C, H, W))

def applyStlyeTransfer(
    model: VGGEncDec,
    I: torch.Tensor,
    tforms: List[GOTTransform],
    weights: List[float],
    device: torch.device,
) -> torch.Tensor:
    # Verify inputs
    assert len(tforms) == len(weights), "Must have as many weights as transforms"
    K, C, M, N = I.size()
    assert K == 1, "Expected a batch size of 1"

    # Setup lists and such
    encoders = [model.e5, model.e4, model.e3, model.e2, model.e1]
    decoders = [model.d5, model.d4, model.d3, model.d2, model.d1]
    Im = deepcopy(I)

    print(len(encoders))
    print(len(decoders))
    print(len(tforms))
    print(len(weights))

    # Perform transformation
    for encoder, decoder, tform, weight in zip(encoders, decoders, tforms, weights):
        print(f"Image shape = {Im.shape}")
        torch.cuda.empty_cache()
        F = encoder(Im)
        F = (F.data.cpu().squeeze(0)).to(device)
        Ft = tform(F)
        Fc = ((1.0 - weight) * F + weight * Ft).cpu().unsqueeze(0).to(device)
        Im = decoder(Fc)
        Im = Im[:, :, :M, :N]

    return Im

def getGOTTransform(muc: torch.Tensor,
                    Sigmac: torch.Tensor,
                    mus: torch.Tensor,
                    Sigmas: torch.Tensor,
                    device: torch.device) -> GOTTransform:
    '''
        @brief Write something here...
    '''
    Lc, Uc = torch.linalg.eigh(Sigmac)
    Ls = torch.linalg.eigvalsh(Sigmas)
    Lc = Lc + abs(torch.min(Lc)) + 1e-6
    Ls = Ls + abs(torch.min(Ls)) + 1e-6
    Sigmach = symmetrize(torch.matmul(torch.matmul(Uc, torch.diag(torch.sqrt(Lc))), Uc.t()), device)
    Sigmachi = symmetrize(torch.matmul(torch.matmul(Uc, torch.diag(torch.div(1.0, torch.sqrt(Lc)))), Uc.t()), device)
    mid = symmetrize(torch.matmul(torch.matmul(Sigmach, Sigmas), Sigmach), device)
    Lm, Um = torch.linalg.eigh(mid)
    Lm = Lm + abs(torch.min(Lm)) + 1e-6
    midh = symmetrize(torch.matmul(torch.matmul(Um, torch.diag(torch.sqrt(Lm))), Um.t()), device)
    A = torch.matmul(torch.matmul(Sigmachi, midh), Sigmachi)
    return GOTTransform(A, muc, mus, device)

def getGOTTransforms(mcsc: List[Tuple[torch.Tensor, torch.Tensor]],
                     mcss: List[Tuple[torch.Tensor, torch.Tensor]],
                     device: torch.device,
                     verbose: bool = False) -> List[GOTTransform]:
    tforms = list()
    for idx, ((muc, Sigmac), (mus, Sigmas)) in enumerate(zip(mcsc, mcss)):
        start = time.time()
        tforms.append(getGOTTransform(muc, Sigmac, mus, Sigmas, device))
        end = time.time()
        if verbose:
            print(f'Elapsed time to create GOT map for layer {idx + 1}: {end - start}')

    return tforms

def getMeanCov(
    F: torch.Tensor,
    scale: float,
    device: torch.device,
    numSamples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        @brief Get the mean and covariance of a set of features
        @param F The tensor containing the features of interest
        @param scale The scale factor (1 / N) where N is the total number of samples
        @param numSamples The number of samples to use in stochastic approximation
    '''
    C, H, W = F.size()
    if numSamples > H * W:
        numSamples = H * W
    Fr = torch.reshape(F, (C, H * W))
    idxs = torch.multinomial(torch.ones((H * W)), numSamples)
    Frs = Fr[:, idxs]
    mu = scale * torch.sum(Frs, dim = 1)
    Frsc = Frs - (mu.cpu().unsqueeze(1).repeat(1, numSamples)).to(device)
    Fo = torch.matmul((Frsc.t().cpu().unsqueeze(2)).to(device), (Frsc.t().cpu().unsqueeze(1)).to(device))
    Sigma = scale * torch.sum(Fo, dim = 0)
    return mu, Sigma

def getMeanCovImage(
    I: torch.Tensor,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    numSamples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, H, W = I.size()
    scale = 1.0 / float(H * W)
    F = encoder(I).data.cpu().squeeze(0).to(device)
    return getMeanCov(F=F, scale=scale, device=device, numSamples=numSamples)

def getMeanCovLayers(model: VGGEncDec,
                     Im: torch.Tensor,
                     device: torch.device,
                     numSamples: int,
                     verbose: bool = False) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    outputs = list()
    encoders = [model.e5, model.e4, model.e3, model.e2, model.e1]
    for idx, encoder in enumerate(encoders):
        start = time.time()
        outputs.append(getMeanCovImage(Im, encoder, device, numSamples=numSamples))
        end = time.time()
        if verbose:
            print(f'Elapsed time for layer {idx + 1}: {end - start}')

    return outputs

# Define the required functions
def symmetrize(A: torch.Tensor, device, tol: float = 1e-6) -> torch.Tensor:
    '''
        @brief Perform symmetrization and eigen-value shifting for a matrix that "should" be PSD
        @param A The tensor to symmetrize (should be a matrix)
        @param tol The amount to shift the eigenvalues by to make the matrix PSD
        @note PSD means positive semi-definite in this context
    '''
    m, _ = A.shape
    return 0.5 * (A + A.t()) + (tol * torch.eye(m)).to(device)

def gaussianOptimalTransport(Fc: torch.Tensor, Fs: torch.Tensor, nSamples: int = 1024) -> torch.Tensor:
    C, H, W = Fc.size()
    Fcrs = torch.reshape(Fc, (C, H * W))
    Fsrs = torch.reshape(Fs, (C, H * W))
    idxs = torch.multinomial(torch.ones((H * W)), nSamples)
    Fcds = Fcrs[:, idxs]
    Fsds = Fsrs[:, idxs]
    muc = torch.mean(Fcds, dim = 1)
    mus = torch.mean(Fsds, dim = 1)
    Fcdsc = Fcds - muc.cpu().unsqueeze(1).repeat(1, nSamples)
    Fsdsc = Fsds - mus.cpu().unsqueeze(1).repeat(1, nSamples)
    Fco = torch.matmul(Fcdsc.t().cpu().unsqueeze(2), Fcdsc.t().cpu().unsqueeze(1))
    Fso = torch.matmul(Fsdsc.t().cpu().unsqueeze(2), Fsdsc.t().cpu().unsqueeze(1))
    Sigmac = torch.mean(Fco, dim = 0)
    Sigmas = torch.mean(Fso, dim = 0)
    Lc, Uc = torch.symeig(symmetrize(Sigmac), eigenvectors = True)
    Ls, Us = torch.symeig(symmetrize(Sigmas), eigenvectors = True)
    Sigmach = torch.matmul(torch.matmul(Uc, torch.diag(torch.sqrt(Lc))), Uc.t())
    Sigmachi = torch.matmul(torch.matmul(Uc, torch.diag(torch.div(1.0, torch.sqrt(Lc)))), Uc.t())
    mid = torch.matmul(torch.matmul(Sigmach, Sigmas), Sigmach)
    Lm, Um = torch.symeig(symmetrize(mid), eigenvectors = True)
    midh = torch.matmul(torch.matmul(Um, torch.diag(torch.sqrt(Lm))), Um.t())
    A = torch.matmul(torch.matmul(Sigmachi, midh), Sigmachi)
    Fcc = Fcrs - muc.cpu().unsqueeze(1).repeat(1, H * W)
    return torch.reshape(torch.matmul(A, Fcc) + mus.cpu().unsqueeze(1).repeat(1, H * W), (C, H, W))

def getImages(contentPath: str, stylePath: str, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Load both images as PIL images
    Imc = Image.open(contentPath).convert('RGB')
    Ims = Image.open(stylePath).convert('RGB')
    if size != 0:
        scaleTform = transforms.Resize(size)
        Imc = scaleTform(Imc)
        m, n = Imc.size
        Ims = transforms.Resize((n, m))(Ims)

    toTensorTform = transforms.ToTensor()
    Imct = toTensorTform(Imc)
    Imst = toTensorTform(Ims)
    Imct = Imct.unsqueeze(0)
    Imst = Imst.unsqueeze(0)
    return Imct, Imst

def getImage(path: str) -> torch.Tensor:
    '''
        @brief Write something here...
    '''
    # Load image into PIL
    Im = Image.open(path).convert('RGB')
    tform = transforms.ToTensor()
    Imt = tform(Im)
    Imt = Imt.cpu().unsqueeze(0)
    return Imt

def styleTransferFineToCoarse(model: VGGEncDec,
                              Ic: torch.Tensor,
                              Is: torch.Tensor,
                              outName: str,
                              dirName: str,
                              weights: List[float]) -> None:
    # assert Ic.shape == Is.shape, "Input images must have the same shape"
    N, C, M, N = Ic.shape
    start = time.time()
    w5 = weights[0]
    c5 = model.e5(Ic)
    s5 = model.e5(Is)
    c5 = c5.data.cpu().squeeze(0)
    s5 = s5.data.cpu().squeeze(0)
    got5 = gaussianOptimalTransport(c5, s5)
    im5 = model.d5(((1.0 - w5) * c5 + w5 * got5).cpu().unsqueeze(0))
    im5 = im5[:, :, :M, :N]
    end = time.time()
    elapsed = end - start
    print(f'Elapsed time for 1st layer: {elapsed}')

    start = time.time()
    w4 = weights[1]
    c4 = model.e4(im5)
    s4 = model.e4(Is)
    c4 = c4.data.cpu().squeeze(0)
    s4 = s4.data.cpu().squeeze(0)
    got4 = gaussianOptimalTransport(c4, s4)
    im4 = model.d4(((1.0 - w4) * c4 + w4 * got4).cpu().unsqueeze(0))
    im4 = im4[:, :, :M, :N]
    end = time.time()
    elapsed = end - start
    print(f'Elapsed time for 2nd layer: {elapsed}')

    start = time.time()
    w3 = weights[2]
    c3 = model.e3(im4)
    s3 = model.e3(Is)
    c3 = c3.data.cpu().squeeze(0)
    s3 = s3.data.cpu().squeeze(0)
    got3 = gaussianOptimalTransport(c3, s3)
    im3 = model.d3(((1.0 - w3) * c3 + w3 * got3).cpu().unsqueeze(0))
    im3 = im3[:, :, :M, :N]
    end = time.time()
    elapsed = end - start
    print(f'Elapsed time for 3rd layer: {elapsed}')

    start = time.time()
    w2 = weights[3]
    c2 = model.e2(im3)
    s2 = model.e2(Is)
    c2 = c2.data.cpu().squeeze(0)
    s2 = s2.data.cpu().squeeze(0)
    got2 = gaussianOptimalTransport(c2, s2)
    im2 = model.d2(((1.0 - w2) * c2 + w2 * got2).cpu().unsqueeze(0))
    im2 = im2[:, :, :M, :N]
    end = time.time()
    elapsed = end - start
    print(f'Elapsed time for 4th layer: {elapsed}')

    start = time.time()
    w1 = weights[4]
    c1 = model.e1(im2)
    s1 = model.e1(Is)
    c1 = c1.data.cpu().squeeze(0)
    s1 = s1.data.cpu().squeeze(0)
    got1 = gaussianOptimalTransport(c1, s1)
    im = model.d1(((1.0 - w1) * c1 + w1 * got1).cpu().unsqueeze(0))
    im = im[:, :, :M, :N]
    end = time.time()
    elapsed = end - start
    print(f'Elapsed time for 5th layer: {elapsed}')
    torchvision.utils.save_image(im.data.cpu().float(), os.path.join(dirName, outName))
    return

def styleTransfer(model: VGGEncDec,
                  Ic: torch.Tensor,
                  Is: torch.Tensor,
                  outName: str,
                  dirName: str,
                  weights: List[float]) -> None:
    # assert Ic.shape == Is.shape, "Input images must have the same shape"
    N, C, M, N = Ic.shape
    start = time.time()
    w5 = weights[0]
    print(f"model.e1 is CUDA = {next(model.e1.parameters()).is_cuda}")
    print(f"Ic is CUDA = {Ic.is_cuda}")
    c5 = model.e1(Ic)
    s5 = model.e1(Is)
    c5 = c5.data.cpu().squeeze(0)
    s5 = s5.data.cpu().squeeze(0)
    got5 = gaussianOptimalTransport(c5, s5)
    im5 = model.d1(((1.0 - w5) * c5 + w5 * got5).cpu().unsqueeze(0))
    im5 = im5[:, :, :M, :N]
    end = time.time()
    elapsed = end - start
    print(f'Elapsed time for 1st layer: {elapsed}')

    start = time.time()
    w4 = weights[1]
    c4 = model.e2(im5)
    s4 = model.e2(Is)
    c4 = c4.data.cpu().squeeze(0)
    s4 = s4.data.cpu().squeeze(0)
    got4 = gaussianOptimalTransport(c4, s4)
    im4 = model.d2(((1.0 - w4) * c4 + w4 * got4).cpu().unsqueeze(0))
    im4 = im4[:, :, :M, :N]
    end = time.time()
    elapsed = end - start
    print(f'Elapsed time for 2nd layer: {elapsed}')

    start = time.time()
    w3 = weights[2]
    c3 = model.e3(im4)
    s3 = model.e3(Is)
    c3 = c3.data.cpu().squeeze(0)
    s3 = s3.data.cpu().squeeze(0)
    got3 = gaussianOptimalTransport(c3, s3)
    im3 = model.d3(((1.0 - w3) * c3 + w3 * got3).cpu().unsqueeze(0))
    im3 = im3[:, :, :M, :N]
    end = time.time()
    elapsed = end - start
    print(f'Elapsed time for 3rd layer: {elapsed}')

    start = time.time()
    w2 = weights[3]
    c2 = model.e4(im3)
    s2 = model.e4(Is)
    c2 = c2.data.cpu().squeeze(0)
    s2 = s2.data.cpu().squeeze(0)
    got2 = gaussianOptimalTransport(c2, s2)
    im2 = model.d4(((1.0 - w2) * c2 + w2 * got2).cpu().unsqueeze(0))
    im2 = im2[:, :, :M, :N]
    end = time.time()
    elapsed = end - start
    print(f'Elapsed time for 4th layer: {elapsed}')

    start = time.time()
    w1 = weights[4]
    c1 = model.e5(im2)
    s1 = model.e5(Is)
    c1 = c1.data.cpu().squeeze(0)
    s1 = s1.data.cpu().squeeze(0)
    got1 = gaussianOptimalTransport(c1, s1)
    im = model.d5(((1.0 - w1) * c1 + w1 * got1).cpu().unsqueeze(0))
    im = im[:, :, :M, :N]
    end = time.time()
    elapsed = end - start
    print(f'Elapsed time for 5th layer: {elapsed}')
    torchvision.utils.save_image(im.data.cpu().float(), os.path.join(dirName, outName))
    return

if __name__ == '__main__':
    # Do some imports
    from matplotlib import pyplot as plt
    import numpy as np

    # Set up the argument parser
    parser = ap.ArgumentParser(description='Gaussian Optimal Transport Style Transfer')
    parser.add_argument('--contentPath',default='images/content.png',help='Path to content image')
    parser.add_argument('--stylePath',default='images/style.png',help='Path to style image')
    parser.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.pth', help='Path to the VGG conv1_1')
    parser.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.pth', help='Path to the VGG conv2_1')
    parser.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.pth', help='Path to the VGG conv3_1')
    parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.pth', help='Path to the VGG conv4_1')
    parser.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.pth', help='Path to the VGG conv5_1')
    parser.add_argument('--decoder5', default='models/feature_invertor_conv5_1.pth', help='Path to the decoder5')
    parser.add_argument('--decoder4', default='models/feature_invertor_conv4_1.pth', help='Path to the decoder4')
    parser.add_argument('--decoder3', default='models/feature_invertor_conv3_1.pth', help='Path to the decoder3')
    parser.add_argument('--decoder2', default='models/feature_invertor_conv2_1.pth', help='Path to the decoder2')
    parser.add_argument('--decoder1', default='models/feature_invertor_conv1_1.pth', help='Path to the decoder1')
    parser.add_argument('--cuda', action='store_true', help='Enables cuda')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--fineToCoarse', action = 'store_true', help = 'Start with fine resolution as opposed to coarse')
    parser.add_argument('--size', type=int, default=0, help='Resize image to a height of size while maintaining aspect ratio. No resizing by default')
    parser.add_argument('--outFolder', default='samples/', help='Folder to store output images in')
    parser.add_argument('--outName', default = 'Example.png', help = 'The desired name of the output file')
    parser.add_argument('--gpu', type=int, default=0, help="Which GPU to run on. Defaults to zero.")
    parser.add_argument('--contentOutName', default = None, help = 'Save the content image after resizing')
    parser.add_argument('--styleOutName', default = None, help = 'Save the style image after resizing')
    parser.add_argument('--numSamples', type=int, default=1024)
    args = parser.parse_args()

    if not os.path.exists(args.outFolder):
        os.makedirs(args.outFolder)

    # Finish the setup
    # weights = [0.0, 0.0, 0.5, 0.85, 0.85]
    weights = [0.7, 0.7, 0.5, 0.1, 0.1]
    Ic = getImage(args.contentPath)
    Is = getImage(args.stylePath)
    # Ic, Is = getImages(args.contentPath, args.stylePath, args.size)
    vgg = VGGEncDec(args)
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Do the actual style transfer for a single image
    if args.contentOutName is not None:
        torchvision.utils.save_image(Ic.data.cpu().float(), os.path.join(args.outFolder, args.contentOutName))
    if args.styleOutName is not None:
        torchvision.utils.save_image(Is.data.cpu().float(), os.path.join(args.outFolder, args.styleOutName))
    with torch.no_grad():
        vgg.to(device)
        Ic = Ic.to(device)
        Is = Is.to(device)

        print("Starting the mean cov process for content image")
        mcsc = getMeanCovLayers(vgg, Ic, device, numSamples=args.numSamples, verbose = args.verbose)
        print("Starting the mean cov process with style image")
        mcss = getMeanCovLayers(vgg, Is, device, numSamples=args.numSamples, verbose = args.verbose)
        tforms = getGOTTransforms(mcsc, mcss, device, verbose = args.verbose)
        Imst = applyStlyeTransfer(vgg, Ic, tforms, weights, device)
        torchvision.utils.save_image(Imst.data.cpu().float(), os.path.join(args.outFolder, args.outName))
        plt.imshow((Imst - np.min(Imst)) / (np.max(Imst) - np.min(Imst)))
        plt.show()
