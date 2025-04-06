# py file contain optical elements forming optical system and
# essential optical propagation methods
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.fft import fftshift, fft2, ifft2, ifftshift
from torchvision import transforms
from utils import *

# BACKENDS = {
#     "riemann": RiemannBackend}

class Lens(nn.Module):
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda):
        super(Lens, self).__init__()
        # basic parameters
        temp = np.arange((-np.ceil((whole_dim - 1) / 2)),
                         np.floor((whole_dim - 1) / 2) + 0.5)
        x = temp * pixel_size
        xx, yy = np.meshgrid(x, x)
        lens_function = np.exp(-1j * math.pi / wave_lambda / focal_length * (xx ** 2 + yy ** 2))
        self.lens_function = torch.tensor(lens_function, dtype=torch.complex64).cuda()

    def forward(self, input_field):
        out = torch.mul(input_field, self.lens_function)
        return out
# TODO figure out the whole_dim parameter and its unit


class PhaseMask(nn.Module):
    def __init__(self, whole_dim, phase_dim, phase=None):
        super(PhaseMask, self).__init__()
        self.whole_dim = whole_dim
        phase = torch.ones(1, phase_dim, phase_dim, dtype=torch.float32) if phase is None else torch.tensor(
            phase, dtype=torch.float32)
        # trainable phase parameter
        self.w_p = nn.Parameter(phase)
        pad_size = (whole_dim - phase_dim)//2
        self.paddings = (pad_size, pad_size, pad_size, pad_size)

    def forward(self, input_field):
        mask_phase = torch.sigmoid(self.w_p) * 1.999 * math.pi
        mask_complex = F.pad(torch.complex(torch.cos(mask_phase), torch.sin(mask_phase)), self.paddings).cuda()
        output_field = torch.mul(input_field, mask_complex)
        return output_field


class Sensor(nn.Module):
    def __init__(self):
        super(Sensor, self).__init__()

    def forward(self, input_field):
        x = torch.square(torch.real(input_field)) + torch.square(torch.imag(input_field))
        return x



# propagation method
class AngSpecProp(nn.Module):  
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda):
        super(AngSpecProp, self).__init__()
        k = 2*math.pi/wave_lambda  # optical wavevector
        df1 = 1 / (whole_dim*pixel_size)
        f = np.arange((-np.ceil((whole_dim - 1) / 2)),
                      np.floor((whole_dim - 1) / 2)+0.5) * df1
        fxx, fyy = np.meshgrid(f, f)
        fsq = fxx ** 2 + fyy ** 2

        self.Q2 = torch.tensor(np.exp(-1j*(math.pi**2)*2*focal_length/k*fsq), dtype=torch.complex64).cuda()
        self.pixel_size = pixel_size
        self.df1 = df1

    def ft2(self, g, delta):
        return fftshift(fft2(ifftshift(g))) * (delta ** 2)

    def ift2(self, G, delta_f):
        N = G.shape[1]
        return ifftshift(ifft2(fftshift(G))) * ((N * delta_f)**2)

    def forward(self, input_field):
        # compute the propagated field
        Uout = self.ift2(self.Q2 * self.ft2(input_field, self.pixel_size), self.df1)
        return Uout



class LightPropagation(torch.nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        pixel_size,
        wavelength,
        layer_distance,
        samples_per_pixel=1,
        propagation_algorithm="riemann",
    ):

        super().__init__()

        self.zero_propagation = layer_distance == 0
        if self.zero_propagation and input_dimension != output_dimension:
            raise ValueError(f"Input and ouput dimensions must be same if layer distance is zero.")
        self.samples_per_pixel = samples_per_pixel
        self.backend = RiemannBackend(
            input_dimension,
            output_dimension,
            pixel_size,
            wavelength,
            layer_distance,
            samples_per_pixel,
        )

        self.register_buffer(
            "default_field",
            torch.ones(
                input_dimension * samples_per_pixel, input_dimension * samples_per_pixel, dtype=torch.cdouble
            ),
        )

    def forward(self, image: torch.Tensor = None, field: torch.Tensor = None):
        if field is None:
            field = self.default_field
        if field.dtype != torch.cdouble:
            raise ValueError(f"Field tensor must be cdouble type.")

        if self.zero_propagation:
            if image is not None:
                return modulate_field(field, image, self.samples_per_pixel)
            return field
        return self.backend(image, field)

    def get_H(self):
        return self.backend.H

    def get_H_fr(self):
        return self.backend.H_fr


class RiemannBackend(torch.nn.Module):
    def __init__(
        self, input_dimension, output_dimension, pixel_size, wavelength, layer_distance, samples_per_pixel
    ):
        super().__init__()

        self.samples_per_pixel = samples_per_pixel
        self.n_in_samples = input_dimension * samples_per_pixel
        # H and H_fr are dtype cdouble for precision
        self.H = self._calculate_transfer_function(
            input_dimension, output_dimension, pixel_size, wavelength, layer_distance, samples_per_pixel
        )
        self.register_buffer("H_fr", fft2(self.H), persistent=False)

    def forward(self, image, field):
        self._check_input_dimensions(field)
        if image is not None:
            field = modulate_field(field, image, self.samples_per_pixel)
        return conv2d_fft(self.H_fr, field)

    def _calculate_transfer_function(
        self, input_dimension, output_dimension, pixel_size, wavelength, layer_distance, samples_per_pixel
    ):
        n_in_samples = input_dimension * samples_per_pixel
        n_out_samples = output_dimension * samples_per_pixel
        distance_offset = (n_out_samples + n_in_samples) / 2
        dx = pixel_size / samples_per_pixel
        distance_grid = (torch.arange(-distance_offset + 1, distance_offset) * dx).double()
        differential_x, differential_y = torch.meshgrid(distance_grid, distance_grid, indexing="xy")

        r_squared = differential_x**2 + differential_y**2 + layer_distance**2
        r = torch.sqrt(r_squared)
        transfer_function = (
            (layer_distance / r_squared * (1 / (2 * torch.pi * r) - 1.0j / wavelength))
            * torch.exp(2j * torch.pi * r / wavelength)
            * dx**2
        )
        return transfer_function

    def _check_input_dimensions(self, field):
        if field.shape[-2:] != (self.n_in_samples, self.n_in_samples):
            raise ValueError(
                f"Field has incorrect size. Expected dimensions of {self.n_in_samples}x{self.n_in_samples} but got {field.size(-1)}x{field.size(-2)}."
            )


@torch.jit.script
def conv2d_fft(H_fr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Performs a 2D convolution using Fast Fourier Transforms (FFT).

    Args:
        H_fr (torch.Tensor): Fourier-transformed transfer function.
        x (torch.Tensor): Input complex field.

    Returns:
        torch.Tensor: Output field after convolution.
    """
    output_size = (H_fr.size(-2) - x.size(-2) + 1, H_fr.size(-1) - x.size(-1) + 1)
    x_fr = fft2(x.flip(-1, -2).conj(), s=(H_fr.size(-2), H_fr.size(-1)))  # cdouble necessary
    output_fr = H_fr * x_fr.conj()
    output = ifft2(output_fr)[..., : output_size[0], : output_size[1]].clone()
    return output  # return to cfloat


def scale_tensor(tensor, samples_per_pixel):
    return tensor.repeat_interleave(samples_per_pixel, dim=-1).repeat_interleave(samples_per_pixel, dim=-2)


def modulate_field(field, modulation_profile, samples_per_pixel):
    return field * scale_tensor(modulation_profile, samples_per_pixel)




