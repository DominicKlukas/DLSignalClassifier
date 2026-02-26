import numpy as np

def _normalize(lut: np.ndarray) -> np.ndarray:
    """Normalize constellation to unit average power."""
    lut = np.asarray(lut, dtype=np.complex64)
    return lut / np.sqrt(np.mean(np.abs(lut)**2))

def _gray(i: np.ndarray) -> np.ndarray:
    """Binary-to-Gray conversion, vectorized."""
    return i ^ (i >> 1)

def make_psk_lut(M: int) -> np.ndarray:
    """
    Gray-coded M-PSK LUT of length M (=2^bps).
    Indexing convention:
      - bits -> binary integer idx
      - symbol = exp(j*2π*gray(idx)/M)
    """
    idx = np.arange(M, dtype=np.uint32)
    g = _gray(idx)
    phases = 2 * np.pi * g / M
    lut = np.exp(1j * phases).astype(np.complex64)
    return _normalize(lut)

def make_pam_lut(M: int) -> np.ndarray:
    """
    Gray-coded M-PAM LUT (real axis), length M (=2^bps).
    Levels are equally spaced: -(M-1), ..., +(M-1) step 2.
    Gray mapping: binary idx -> gray idx -> level[gray idx]
    """
    idx = np.arange(M, dtype=np.uint32)
    g = _gray(idx)
    levels = np.arange(-(M - 1), M, 2, dtype=np.float32)  # length M
    lut = levels[g].astype(np.complex64)  # imag=0
    return _normalize(lut)

def make_square_qam_lut(M: int) -> np.ndarray:
    """
    Gray-coded square M-QAM LUT, length M (=2^bps), where M is a perfect square.
    Gray coding applied separately to I and Q (standard practical approach).

    Indexing convention:
      - bits -> binary integer idx in [0, M)
      - split idx bits into I and Q halves:
          i_bin = idx >> m
          q_bin = idx & (2^m - 1)
        where m = log2(sqrt(M))
      - i_gray, q_gray -> amplitude levels
    """
    sqrtM = int(np.sqrt(M))
    if sqrtM * sqrtM != M:
        raise ValueError(f"M must be a perfect square for square QAM. Got M={M}")
    m = int(np.log2(sqrtM))
    if 2**m != sqrtM:
        raise ValueError(f"M must be 2^(2m). Got M={M}")

    idx = np.arange(M, dtype=np.uint32)
    i_bin = idx >> m
    q_bin = idx & ((1 << m) - 1)

    i_g = _gray(i_bin)
    q_g = _gray(q_bin)

    # PAM levels for each axis
    levels = np.arange(-(sqrtM - 1), sqrtM, 2, dtype=np.float32)  # length sqrtM

    I = levels[i_g]
    Q = levels[q_g]
    lut = (I + 1j * Q).astype(np.complex64)
    return _normalize(lut)

# -------------------------
# Comprehensive MODS dict
# -------------------------

MODS = {}

# PSK family (Gray)
for M in (2, 4, 8, 16, 32):
    bps = int(np.log2(M))
    name = {2: "BPSK", 4: "QPSK"}.get(M, f"{M}PSK")
    MODS[name] = {"bps": bps, "lut": make_psk_lut(M)}

# PAM family (Gray)
for M in (2, 4, 8, 16):
    bps = int(np.log2(M))
    name = {2: "PAM2"}.get(M, f"PAM{M}")
    MODS[name] = {"bps": bps, "lut": make_pam_lut(M)}

# Square QAM family (Gray on I/Q)
# Note: "4QAM" is the square QAM grid (equivalent constellation to QPSK, different common naming).
for M in (4, 16, 64, 256, 1024):
    bps = int(np.log2(M))
    name = "4QAM" if M == 4 else f"{M}QAM"
    MODS[name] = {"bps": bps, "lut": make_square_qam_lut(M)}

# Optional aliases you might like
MODS["PAM2"] = MODS["PAM2"] if "PAM2" in MODS else {"bps": 1, "lut": make_pam_lut(2)}
MODS["QAM16"] = MODS["16QAM"]
MODS["QAM64"] = MODS["64QAM"]
MODS["QAM256"] = MODS["256QAM"]