'''
MIT License

Copyright (c) 2022 beresandras

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import tensorflow as tf

from abc import ABC, abstractmethod


class DiffusionSchedule(ABC):
    def __init__(self, start_log_snr, end_log_snr):
        assert (
            start_log_snr > end_log_snr
        ), "The starting SNR has to be higher than the final SNR."

        self.start_snr = tf.exp(start_log_snr)
        self.end_snr = tf.exp(end_log_snr)

        self.start_noise_power = 1.0 / (1.0 + self.start_snr)
        self.end_noise_power = 1.0 / (1.0 + self.end_snr)

    def __call__(self, diffusion_times):
        noise_powers = self.get_noise_powers(diffusion_times)

        # the signal and noise power will always sum to one
        signal_powers = 1.0 - noise_powers

        # the rates are the square roots of the powers
        # variance**0.5 -> standard deviation
        signal_rates = signal_powers**0.5
        noise_rates = noise_powers**0.5

        return signal_rates, noise_rates

    @abstractmethod
    def get_noise_powers(self, diffusion_times):
        pass

class CosineSchedule(DiffusionSchedule):
    # noise rate increases sinusoidally
    # signal rate decreases as a cosine function
    # simplified from the "cosine schedule" of Improved DDPM https://arxiv.org/abs/2102.09672
    def get_noise_powers(self, diffusion_times):
        start_angle = tf.asin(self.start_noise_power**0.5)
        end_angle = tf.asin(self.end_noise_power**0.5)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        return tf.sin(diffusion_angles) ** 2