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
    sps_int: int                    # Resolution of internal representation of signal
    SNR: np.float32
    sr_out: np.float32
    pulse_shaping_filter_num_taps: int

class SignalGenerator:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def gen_rrc(self, sps, num_taps):
        beta = 0.35
        Ts = float(sps)
        t = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) // 2
        x = t / Ts
        den = 1 - (2 * beta * x) ** 2
        den[np.isclose(den, 0.0)] = np.finfo(np.float64).eps
        h = np.sinc(x) * np.cos(np.pi * beta * x) / den
        return h

    def generate_batch(self, signal_length: int, num_signals: int, modulation_types: list,
                        symbol_rate_bnds: list, phase_shift_bnds: list, frequency_offset_bnds: list,
                        SNR_bnds: list, sr_out: int):
        # signal_length: final length of signal
        # num_singals: number of signals to add to the batch
        # modulation_types: list of strings, each is a modulation type in the MODS dict defined in Modulation.py

        signal_cnfgs = []
        for i in range(num_signals):
            modulation = self.rng.choice(modulation_types)
            symbol_rate = self.rng.uniform(symbol_rate_bnds[0], symbol_rate_bnds[1])
            phase_shift = self.rng.uniform(phase_shift_bnds[0], phase_shift_bnds[1])
            frequency_offset = self.rng.uniform(frequency_offset_bnds[0], frequency_offset_bnds[1])
            SNR = self.rng.uniform(SNR_bnds[0], SNR_bnds[1])

            sps_int = max(1, int(symbol_rate_bnds[1] * 8))
            pulse_shaping_filter_num_taps = sps_int*signal_length + 1

            cnfg = SignalConfig(modulation, symbol_rate, signal_length, phase_shift, frequency_offset,
                                sps_int, SNR, sr_out, pulse_shaping_filter_num_taps)

            signal_cnfgs += [cnfg]
        signals = []
        for cnfg in signal_cnfgs:
            signals += [self.generate_signal(cnfg)]
        return signals

    def generate_signal(self, cnfg : SignalConfig):


        # Get the lookup table and bit rate for the signal
        mod_dict = MODS[cnfg.modulation]
        bps = mod_dict["bps"] # bits per symbol
        LUT = mod_dict["lut"] # lookup table for this modulation type

        # Get the requried number of bits to generate 
        # Add the pulse shaping filter
        symbol_length = cnfg.signal_length + cnfg.pulse_shaping_filter_num_taps 
        bit_str_length = max(
            bps,
            int(symbol_length * bps * cnfg.symbol_rate / cnfg.sr_out),
        )
        bit_str = self.rng.integers(0, 2, size=bit_str_length, dtype=np.uint8)

        # Build the symbol dict, with zero padding
        symbols = []
        zero_padding = [0] * (cnfg.sps_int - 1)
        idx = 0
        while idx <= bit_str_length - bps:
            symbol_idx = 0
            for b in range(bps):
                symbol_idx += 2**b*bit_str[b+idx]
            symbols += [LUT[symbol_idx]]+zero_padding
            idx += bps

        # Upsampling using RRC fitler
        rrc = self.gen_rrc(cnfg.sps_int, cnfg.pulse_shaping_filter_num_taps)
        signal = sc.convolve(symbols, rrc, mode = 'valid')

        # Adding white noise

        #Compute the standard deviation for the requested SNR
        signal_power = np.mean(np.abs(signal)**2) # Here, signal is complex, so to compute power we take the absolute value
        snr_linear = 10**(cnfg.SNR / 10)
        noise_power = signal_power / snr_linear
        sigma = np.sqrt(noise_power)
        #Generate the white noise
        noise = (self.rng.normal(0, sigma/np.sqrt(2), signal.shape) +
         1j*self.rng.normal(0, sigma/np.sqrt(2), signal.shape))
        signal = signal+noise


        # Apply transformations
        # Phase Transormation
        rotation = np.exp(1j * cnfg.phase_shift)
        signal = rotation*signal

        # Frequency Offset
        f_s = cnfg.sps_int*cnfg.symbol_rate  # frequency of the "continuous" signal (symbol rate)
        n = np.arange(len(signal))
        freq_offset = np.exp(1j*(cnfg.frequency_offset/f_s)*2*np.pi*n)
        
        # Offset the signal
        signal = signal*freq_offset
        signal = sc.resample(signal, cnfg.signal_length)
        return signal
