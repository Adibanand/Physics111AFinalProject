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
    DwfAnalogOutFunction,
    DwfAnalogInFilter, 
    DwfAcquisitionMode
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


def live_scope_and_decode_image(duration_sec=10, interval_sec=0.1, threshold=1.5, sampling_rate=1000):
    num_bits = int(duration_sec / interval_sec)
    bitstream = ""
    
    dwf = DwfLibrary()
    with openDwfDevice(dwf) as device:
        analog_in = device.analogIn
        analog_in.reset()
        analog_in.channelEnableSet(0, True)
        analog_in.channelFilterSet(0, DwfAnalogInFilter.Average)
        analog_in.channelRangeSet(0, 5.0)
        analog_in.acquisitionModeSet(DwfAcquisitionMode.Single)
        analog_in.frequencySet(sampling_rate)
        analog_in.bufferSizeSet(8192)

        # Setup live plot
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.set_ylim(-0.5, 5.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Live Scope Trace (Phototransistor Signal)")

        print("Starting acquisition and bitstream reconstruction...\n")
        start_time = time.time()

        for i in range(num_bits):
            analog_in.configure(reconfigure=True, start=True)
            while True:
                if analog_in.status(True).name == "Done":
                    break
                time.sleep(0.01)

            num_samples = analog_in.statusSamplesValid()
            data = analog_in.statusData(0, num_samples)
            t = np.linspace(0, len(data)/sampling_rate, len(data))

            # Live plot update
            line.set_data(t, data)
            ax.set_xlim(0, t[-1])
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Bitstream decoding
            avg_voltage = float(np.mean(data))
            bit = '1' if avg_voltage >= threshold else '0'
            bitstream += bit
            print(f"Bit {i+1}: {bit} (V = {avg_voltage:.2f})")

            elapsed = time.time() - start_time
            remaining = (i + 1) * interval_sec - elapsed
            if remaining > 0:
                time.sleep(remaining)

        plt.ioff()
        plt.close()

    print("\nAcquisition complete.")
    print("Reconstructed bitstream:", bitstream)

    # Convert bitstream to image
    if len(bitstream) == 100:
        image_array = np.array(list(map(int, bitstream))).reshape((10, 10))
        plt.imshow(image_array, cmap='gray', interpolation='nearest')
        plt.title("Reconstructed Image")
        plt.axis('off')
        plt.show()
    else:
        print("Error: Bitstream is not 100 bits (10×10 image).")

    return bitstream


