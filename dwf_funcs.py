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
    voltage_high: float = 4.0,
    voltage_low: float = 0.0,
    hold_time: float = 0.1
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


def live_scope_and_decode_image(duration_sec=10, interval_sec=0.1, threshold=1.5, sampling_rate=100000):
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

        # # Setup live plot
        # plt.ion()
        # fig, ax = plt.subplots()
        # line, = ax.plot([], [], lw=2)
        # ax.set_ylim(-0.5, 5.5)
        # ax.set_xlim(0, 1)
        # ax.set_xlabel("Time (s)")
        # ax.set_ylabel("Voltage (V)")
        # ax.set_title("Live Scope Trace (Phototransistor Signal)")

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

            # # Live plot update
            # line.set_data(t, data)
            # ax.set_xlim(0, t[-1])
            # fig.canvas.draw()
            # fig.canvas.flush_events()

            # Bitstream decoding
            avg_voltage = float(np.mean(data))
            bit = '1' if avg_voltage >= threshold else '0'
            bitstream += bit
            print(f"Bit {i+1}: {bit} (V = {avg_voltage:.2f})")

            elapsed = time.time() - start_time
            remaining = (i + 1) * interval_sec - elapsed
            if remaining > 0:
                time.sleep(remaining)

        # plt.ioff()
        # plt.close()

    print("\nAcquisition complete.")
    print("Reconstructed bitstream:", bitstream)

    # # Convert bitstream to image
    # if len(bitstream) == 100:
    #     image_array = np.array(list(map(int, bitstream))).reshape((10, 10))
    #     plt.imshow(image_array, cmap='gray', interpolation='nearest')
    #     plt.title("Reconstructed Image")
    #     plt.axis('off')
    #     plt.show()
    # else:
    #     print("Error: Bitstream is not 100 bits (10×10 image).")

    return bitstream



def output_bitstream2(
    bit_string: str,
    voltage_high: float = 3.0,
    voltage_low: float = 0.0,
    hold_time: float = 1.0
) -> Tuple[Figure, NDArray]:
    """
    Outputs a digital bitstring as DC voltages using the Wavegen (no scope).
    Prepends a 5-second high-voltage level before starting the bitstream.
    Returns a plot of the voltage levels and their FFT.
    """
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
        ao.nodeAmplitudeSet(CH, node, 0.0)
        ao.configure(CH, True)
        time.sleep(0.1)

        # --- Initial high voltage preamble (5 seconds) ---
        ao.nodeOffsetSet(CH, node, voltage_high)
        ao.configure(CH, True)
        t_start = time.time()
        while time.time() - t_start < 5.0:
            times.append(time.time() - t0)
            values.append(voltage_high)
            time.sleep(0.1)  # sample at ~10 Hz for this section

        # --- Bitstream voltage output ---
        for bit in bit_string:
            v = voltage_high if bit == '1' else voltage_low
            ao.nodeOffsetSet(CH, node, v)
            ao.configure(CH, True)
            t = time.time() - t0
            times.append(t)
            values.append(v)
            time.sleep(hold_time)

        # Return to low voltage at end
        ao.nodeOffsetSet(CH, node, voltage_low)
        ao.configure(CH, True)

    # Plot signal and FFT
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


def output_bitstream3(
    bit_string: str,
    voltage_high: float = 3.0,
    voltage_low: float = 0.0,
    hold_time: float = 1.0,
    sample_rate: int = 10000
) -> Tuple[Figure, NDArray]:
    """
    Outputs a digital bitstring as DC voltages using the Wavegen.
    Records and plots the commanded signal and measured scope data (Channel 1) in separate plots.
    """
    import numpy as np
    import time
    from pydwf import DwfLibrary, DwfAnalogOutNode, DwfAnalogOutFunction
    from pydwf import DwfAcquisitionMode, DwfAnalogInFilter, DwfEnumConfigInfo
    from pydwf.utilities import openDwfDevice
    import matplotlib.pyplot as plt

    times = []
    values = []
    t0 = time.time()

    with openDwfDevice(
        DwfLibrary(),
        score_func=lambda conf_params: conf_params[DwfEnumConfigInfo.AnalogInBufferSize]
    ) as device:
        ao = device.analogOut
        ai = device.analogIn
        CH = 0
        node = DwfAnalogOutNode.Carrier

        # Configure output (Wavegen)
        ao.reset(-1)
        ao.nodeEnableSet(CH, node, True)
        ao.nodeFunctionSet(CH, node, DwfAnalogOutFunction.Sine)
        ao.nodeFrequencySet(CH, node, 0.1)
        ao.nodeSymmetrySet(CH, node, 50.0)
        ao.nodeOffsetSet(CH, node, 0.0)
        ao.nodeAmplitudeSet(CH, node, 0.0)
        ao.configure(CH, True)
        time.sleep(0.1)

        # Configure input (Scope CH1)
        ai.reset()
        ai.channelEnableSet(0, True)
        ai.channelRangeSet(0, 5.0)
        ai.channelFilterSet(0, DwfAnalogInFilter.Decimate)
        ai.acquisitionModeSet(DwfAcquisitionMode.Record)
        ai.frequencySet(sample_rate)
        ai.bufferSizeSet(8192)
        total_duration = 5 + hold_time * len(bit_string)
        ai.recordLengthSet(total_duration)
        ai.configure(reconfigure=True, start=True)

        # 5-second preamble at high voltage
        ao.nodeOffsetSet(CH, node, voltage_high)
        ao.configure(CH, True)
        t_start = time.time()
        while time.time() - t_start < 5.0:
            times.append(time.time() - t0)
            values.append(voltage_high)
            time.sleep(0.1)

        # Output bitstream
        for bit in bit_string:
            v = voltage_high if bit == '1' else voltage_low
            ao.nodeOffsetSet(CH, node, v)
            ao.configure(CH, True)
            times.append(time.time() - t0)
            values.append(v)
            time.sleep(hold_time)

        # Final low voltage
        ao.nodeOffsetSet(CH, node, voltage_low)
        ao.configure(CH, True)

        # Wait for scope acquisition to finish
        while True:
            if ai.status(True).name == "Done":
                break
            time.sleep(0.1)

        num_samples = ai.statusSamplesValid()
        measured = ai.statusData(0, num_samples)
        measured_time = np.linspace(0, total_duration, num=len(measured))

    # Plot: separate subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].step(times, values, where='post', label="Commanded Voltage")
    axs[0].set_ylabel("Voltage (V)")
    axs[0].set_title("Commanded Signal")
    axs[0].grid(True)

    axs[1].plot(measured_time, measured, label="Measured (Scope CH1)", color='tab:orange')
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Voltage (V)")
    axs[1].set_title("Measured Signal (Scope CH1)")
    axs[1].grid(True)

    fig.tight_layout()
    return fig, axs



def output_bitstream4(
    bit_string: str,
    voltage_high: float = 3.0,
    voltage_low: float = 0.0,
    hold_time: float = 1.0,
    alt_interval: float = 0.5  # interval duration of each high/low alternation
) -> Tuple[Figure, NDArray]:
    """
    Outputs a digital bitstring as DC voltages using the Wavegen (no scope).
    Prepends a 5-second alternating high/low voltage signal before starting the bitstream.
    Returns a plot of the voltage levels and their FFT.
    """
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from pydwf import (
        DwfLibrary, DwfEnumConfigInfo,
        DwfAnalogOutNode, DwfAnalogOutFunction
    )
    from pydwf.utilities import openDwfDevice

    from dwf_funcs import create_figure, ScanType  # assuming this is in dwf_funcs.py

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
        ao.nodeAmplitudeSet(CH, node, 0.0)
        ao.configure(CH, True)
        time.sleep(0.1)

        # --- Alternating voltage preamble (5 seconds) ---
        duration = 5.0
        elapsed = 0.0
        state = True  # True = high, False = low

        while elapsed < duration:
            v = voltage_high if state else voltage_low
            ao.nodeOffsetSet(CH, node, v)
            ao.configure(CH, True)
            times.append(time.time() - t0)
            values.append(v)
            time.sleep(alt_interval)
            elapsed += alt_interval
            state = not state

        # --- Bitstream voltage output ---
        for bit in bit_string:
            v = voltage_high if bit == '1' else voltage_low
            ao.nodeOffsetSet(CH, node, v)
            ao.configure(CH, True)
            t = time.time() - t0
            times.append(t)
            values.append(v)
            time.sleep(hold_time)

        # Return to low voltage at end
        ao.nodeOffsetSet(CH, node, voltage_low)
        ao.configure(CH, True)

    # Plot signal and FFT
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


def output_bitstream5(
    bit_string: str,
    voltage_high: float = 3.0,
    voltage_low: float = 0.0,
    hold_time: float = 1.0
) -> Tuple[Figure, NDArray]:
    """
    Outputs a digital bitstring as DC voltages using the Wavegen (no scope).
    Prepends a preamble: 1s of low voltage, followed by 5s of high voltage.
    Returns a plot of the voltage levels and their FFT.
    """
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from pydwf import (
        DwfLibrary, DwfEnumConfigInfo,
        DwfAnalogOutNode, DwfAnalogOutFunction
    )
    from pydwf.utilities import openDwfDevice

    from dwf_funcs import create_figure, ScanType  # assuming from your existing module

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
        ao.nodeAmplitudeSet(CH, node, 0.0)
        ao.configure(CH, True)
        time.sleep(0.1)

        # --- Preamble: 1s low, then 5s high ---
        ao.nodeOffsetSet(CH, node, voltage_high)
        ao.configure(CH, True)
        t_start = time.time()
        while time.time() - t_start < 1.0:
            times.append(time.time() - t0)
            values.append(voltage_low)
            time.sleep(0.1)

        ao.nodeOffsetSet(CH, node, voltage_low)
        ao.configure(CH, True)
        t_start = time.time()
        while time.time() - t_start < 5.0:
            times.append(time.time() - t0)
            values.append(voltage_high)
            time.sleep(0.1)

        # --- Bitstream voltage output ---
        for bit in bit_string:
            v = voltage_high if bit == '1' else voltage_low
            ao.nodeOffsetSet(CH, node, v)
            ao.configure(CH, True)
            t = time.time() - t0
            times.append(t)
            values.append(v)
            time.sleep(hold_time)

        # Return to low voltage at end
        ao.nodeOffsetSet(CH, node, voltage_low)
        ao.configure(CH, True)

    # Plot signal and FFT
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
