'''
DWF functions: create_figure and output_bitstream for DC bitstream output using the Wavegen only.
'''

import time
from typing import Optional, Tuple
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pydwf import (
    DwfLibrary,
    DwfEnumConfigInfo,
    DwfAnalogOutNode,
    DwfAnalogOutFunction
)
from pydwf.utilities import openDwfDevice

# Type alias for figure size limits
RangeLimits = Optional[Tuple[float, float]]

class ScanType:
    TEST = 'TEST'
    DEMOD = 'DEMOD'
    LOCKIN = 'LOCKIN'


def create_figure(scan_type: str, figsize: RangeLimits = None) -> Tuple[Figure, NDArray]:
    if scan_type == ScanType.TEST:
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        axs[0].set(xlabel='Time (s)', ylabel='Voltage (V)', title='Signal')
        axs[1].set(xlabel='Frequency (Hz)', ylabel='Magnitude', title='FFT')
    else:
        fig, axs = plt.subplots(2, 2, figsize=figsize)
    for ax in fig.axes:
        ax.grid(True)
    fig.tight_layout()
    return fig, axs


def output_bitstream(
    bit_string: str,
    voltage_high: float = 3.0,
    voltage_low: float = 0.0,
    hold_time: float = 1.0
) -> Tuple[Figure, NDArray]:
    '''
    Outputs a digital bitstring as DC voltages using the Wavegen (no scope).
    Returns a plot of the voltage levels and their FFT.
    '''
    times = []
    values = []
    t0 = time.time()

    with openDwfDevice(
        DwfLibrary(),
        score_func=lambda conf_params: conf_params[DwfEnumConfigInfo.AnalogInBufferSize]
    ) as device:
        ao = device.analogOut
        CH = 0
        node = DwfAnalogOutNode.Carrier

        ao.reset(-1)
        ao.nodeEnableSet(CH, node, True)
        ao.nodeFunctionSet(CH, node, DwfAnalogOutFunction.Sine)
        ao.nodeFrequencySet(CH, node, 0.1)
        ao.nodeSymmetrySet(CH, node, 50.0)
        ao.nodeOffsetSet(CH, node, 0.0)
        ao.nodeAmplitudeSet(CH, node, voltage_low)
        ao.configure(CH, True)
        time.sleep(0.1)

        for bit in bit_string:
            v = voltage_high if bit == '1' else voltage_low
            ao.nodeOffsetSet(CH, node, v)
            ao.configure(CH, True)
            t = time.time() - t0
            times.append(t)
            values.append(v)
            time.sleep(hold_time)

        ao.nodeOffsetSet(CH, node, voltage_low)
        ao.configure(CH, True)

    # Plotting the output signal and its FFT
    fig, axs = create_figure(ScanType.TEST)
    axs[0].step(times, values, where='post')

    if len(times) > 1:
        dt = np.diff(times)
        fs = 1.0 / np.mean(dt)
    else:
        fs = 1.0
    sig = np.array(values)
    fft_vals = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1/fs)
    axs[1].plot(freqs, fft_vals)

    return fig, axs
