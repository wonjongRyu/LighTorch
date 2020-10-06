import numpy as np
import math
import torch
import matplotlib.pyplot as plt


class LightPropagation():
    def __init__(self, propDistance, sizeOfImage):
        self.n = sizeOfImage  # num of pixels
        self.d = 0.000008  # size of pixels # 8um
        self.wvl = 0.000000532  # wavelength # 532 nm
        self.l = self.n*self.d  # side length
        self.z = propDistance  # propagation distance # unit: [m]

    def amplitude_to_complex(self, amplitude):
        """ 
        Tensor stores complex value with real and imag values
        make 4D to 5D (shape of [:,:,:,:,2] = [4D,2])
        [4D, 0]: real values copied from amplitude
        [4D, 1]: imag values filled with 0
        """
        return torch.stack((amplitude, torch.zeros(amplitude.size())), dim=4)

    def phase_to_complex(self, phase):
        """ 
        Tensor stores complex value with real and imag values
        make 4D to 5D (shape of [:,:,:,:,2] = [4D,2])
        [4D, 0]: real values filled with cos(phase)
        [4D, 1]: imag values filled with sin(phase)
        """
        return torch.stack((torch.cos(phase), torch.sin(phase)), dim=4)

    def complex_to_amplitude(self, complex):
        """
        amplitude = sqrt(real**2 + imag**2)
        """
        return torch.sqrt(complex[:, :, :, :, 0].pow(2) + complex[:, :, :, :, 1].pow(2))

    def complex_to_phase(self, complex):
        """
        tan(phase) = imag/real
        phase = atan2(imag, real)
        """
        return torch.atan2(complex[:, :, :, :, 1], complex[:, :, :, :, 0])

    def fourier_transform(self, source, direction):
        """
        direction: forward == 1 / backward == 0
        """
        if direction:
            p = torch.fft(source, 2)  # forward
        else:
            p = torch.ifft(source, 2)  # backward
        return p

    def FFTshift(self, source):
        m, n = source.size()
        temp = torch.zeros((m, n))
        temp[0:m//2, 0:m//2] = source[m//2:m, m//2:m]
        temp[0:m//2, m//2:m] = source[m//2:m, 0:m//2]
        temp[m//2:m, 0:m//2] = source[0:m//2, m//2:m]
        temp[m//2:m, m//2:m] = source[0:m//2, 0:m//2]

        return temp

    def get_transfer_function(self, num, direction):
        if direction:
            z = self.z
        else:
            z = -self.z
        X = torch.linspace(-1/(2*self.d), 1/(2*self.d) - 1/self.l, self.n)
        Y = torch.linspace(-1/(2*self.d), 1/(2*self.d) - 1/self.l, self.n)
        X, Y = torch.meshgrid(X, Y)
        # h = (2*math.pi*self.z/self.wvl)*torch.ones((self.n, self.n)) - math.pi*self.wvl*self.z*(X**2+Y**2)
        h = - math.pi*self.wvl*z*(X**2+Y**2)
        h = self.FFTshift(h)
        h = h.repeat((num, 1, 1, 1))
        H = torch.stack((torch.cos(h), torch.sin(h)), dim=4)

        return H

    def propagation(self, source, direction):
        """
        input: 4D data, shape of [dataNum, 1, M, N]
        """
        dataNum = len(source)
        H = self.get_transfer_function(dataNum, direction=direction)
        U1 = self.fourier_transform(source, direction=direction)

        a = H[:, :, :, :, 0]  # real
        b = H[:, :, :, :, 1]  # imag
        c = U1[:, :, :, :, 0]  # real
        d = U1[:, :, :, :, 1]  # imag

        # H * U1 (elementwise multiplication of complex values)
        U2 = torch.stack((torch.sub(torch.mul(a, c), torch.mul(b, d)),
                          torch.add(torch.mul(a, d), torch.mul(b, c))), dim=4)

        u2 = self.fourier_transform(U2, direction=1-direction)

        return u2


def image_hologram_reconstruction(img):

    m, n = np.shape(img)

    # unit of propDistance: [m]
    LP = LightPropagation(propDistance=0.5, sizeOfImage=m)

    # numpy to torch, make it as 4D
    img = torch.reshape(torch.from_numpy(img), [1, 1, m, n])

    # amplitude to complex
    img = LP.amplitude_to_complex(img)

    # img to hologram, get phase from real and imag
    hologram = LP.propagation(img, 1)  # forward == 1 / backward == 0
    phase = LP.complex_to_phase(hologram)

    # complex hologram to reconimg, get amplitude from real and imag
    reconFromHologram = LP.propagation(hologram, 0)
    reconFromHologram = LP.complex_to_amplitude(reconFromHologram)

    # phase-only hologram to reconimg, get amplitude from real and imag
    poh = LP.phase_to_complex(phase)
    reconFromPOH = LP.propagation(poh, 0)  # forward == 1 / backward == 0
    reconFromPOH = LP.complex_to_amplitude(reconFromPOH)

    # make them as 2D
    phase = phase.numpy().squeeze()
    reconFromHologram = reconFromHologram.numpy().squeeze()
    reconFromPOH = reconFromPOH.numpy().squeeze()

    plt.imsave("phase.png", phase, cmap='gray')
    plt.imsave("reconFromHologram.png", reconFromHologram, cmap='gray')
    plt.imsave("reconFromPOH.png", reconFromPOH, cmap='gray')


def main():

    img = plt.imread('lenna256.png')  # read image as numpy array
    image_hologram_reconstruction(img)


if __name__ == "__main__":
    main()
