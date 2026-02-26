import numpy as np
import scipy.signal as sc
from Modulation import MODS
from dataclasses import dataclass

@dataclass
class SignalConfig:
    modulation: str
    symbol_rate: np.float32
    signal_length: int
    phase_shift: np.float32
    frequency_offset: np.float32
    sps_int: int
    SNR: np.float32
    seed: int
    sr_out: np.float32
    pulse_shaping_filter_num_taps: int

class SignalGenerator:
    def __init__():
        pass

    def gen_rrc(self, sps, num_taps):
        beta = 0.35
        Ts = sps # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
        t = np.arange(num_taps) - (num_taps-1)//2
        h = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)
        return h

    def generate_batch(self, signal_length: int, num_signals: int, modulation_types: dict):
        pass

    def generate_signal(self, cnfg : SignalConfig):

        rng = np.random.default_rng(cnfg.seed)

        # Get the lookup table and bit rate for the signal
        mod_dict = MODS[cnfg.modulation]
        bps = mod_dict["bps"] # bits per symbol
        LUT = mod_dict["LU"]

        # Get the requried number of bits to generate 
        # Add the pulse shaping filter
        symbol_length = cnfg.signal_length + cnfg.pulse_shaping_filter_num_taps 
        bit_str_length = int(symbol_length*bps*cnfg.symbol_rate/cnfg.sr_out)
        bit_str = rng.integers(0, 2, size=bit_str_length, dtype=np.uint8)

        # Build the symbol dict, with zero padding
        symbols = []
        zero_padding = [0]*cnfg.sps_int
        idx = 0
        while idx <= bit_str_length - bps:
            symbol_idx = 0
            for b in range(bps):
                symbol_idx += 2**b*bit_str[b+idx]
            symbols += [LUT[symbol_idx]]+zero_padding
            idx += bps

        # Upsampling using RRC fitler
        rrc = self.gen_rrc(cnfg.sps, cnfg.pulse_shaping_filter_num_taps)
        signal = sc.convolve(symbols, rrc, mode = 'valid')
