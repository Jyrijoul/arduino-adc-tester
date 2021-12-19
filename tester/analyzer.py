from re import template
from jinja2.loaders import PackageLoader
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import norm
from scipy.stats import linregress
import jinja2
import os


plt.rc("font", size=14)


class Analyzer:
    def __init__(self) -> None:
        pass

    def print_results(
        static_data_filename,
        dynamic_data_filename,
        input_noise_data_filename,
        output_filename="results.html",
        rounding_digits=3,
    ):
        Analyzer.plot_input_referred_noise(input_noise_data_filename)
        offset_error, gain_error, dnl, inl, tue = Analyzer.perform_static_analysis(
            static_data_filename
        )
        (
            snr,
            snr_ideal,
            thd,
            sfdr,
            sinad,
            enob,
            noise_floor,
        ) = Analyzer.perform_dynamic_analysis(dynamic_data_filename)

        template_filename = "results_template.html"
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates"
        )

        variables = {
            "tue": tue,
            "offset_error": offset_error,
            "gain_error": gain_error,
            "dnl": dnl,
            "inl": inl,
            "snr": snr,
            "snr_ideal": snr_ideal,
            "thd": thd,
            "sfdr": sfdr,
            "sinad": sinad,
            "enob": enob,
            "noise_floor": noise_floor,
        }

        # Round all the values to the desired length.
        for k, v in variables.items():
            variables[k] = round(v, rounding_digits)

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_path),
            autoescape=jinja2.select_autoescape(["html"]),
        )
        output = env.get_template(template_filename).render(variables)
        # print(output)

        with open(output_filename, "w") as f:
            f.write(output)

    def read_data(filename: str) -> pd.DataFrame:
        data = pd.read_csv(filename)
        # Subtract start time.
        start = data.t.values[0]
        data.t -= start
        # To ms.
        data.t /= 1000000

        return data

    def to_decibels(signal: np.ndarray, power=False) -> np.ndarray:
        ref = np.max(signal)
        if power:
            output = 10 * np.log10(signal / ref)
        else:
            output = 20 * np.log10(signal / ref)
        return output

    def sine(
        n_samples: int, periods: float, min_value: float, max_value: float
    ) -> np.ndarray:
        x = np.linspace(0, 2 * np.pi * periods, n_samples, endpoint=False)
        # Create a sine wave and also map it to the specified range.
        return np.interp(np.sin(x), [-1, 1], [min_value, max_value])

    def code_to_v(code: np.ndarray, vref: float, res=1024) -> np.ndarray:
        return code * vref / res

    def fft_real(signal, power=False):
        fft = np.abs(np.fft.rfft(signal)) / len(signal)
        fft[1:] *= 2

        if power:
            fft **= 2

        return fft

    def v_to_code(voltage, res, vref):
        return voltage * res / vref

    def perfect_adc_code(voltage, res, vref):
        return np.round(voltage * res / vref) / (
            res / vref
        )  # "/ (res / vref)" just because it will be scaled later on.

    def plot(x=[], y=[], title="", xlabel="", ylabel="", legend="", figsize=(15, 5), filename=""):
        fig, ax = plt.subplots(figsize=figsize)
        if len(x) > 0:
            ax.plot(x, y, label=legend)
        else:
            ax.plot(y, label=legend)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        if title != "":
            ax.set_title(title)

        if filename != "":
            plt.savefig(filename)

    def perform_static_analysis(
        filename: str, resolution=10, output_filename="static_analysis", extension="png"
    ):
        data = Analyzer.read_data(filename)

        res = 2 ** resolution
        data["vin"] = data.code * data.vref / res
        data[5000:6000]
        vref = np.mean(data.vref)

        last_transition_v = data.vref / res * (res - 1)
        last_transition = data[data.vout >= last_transition_v].iloc[0].name
        # Keep only the data up until the last transition.
        # It will be used later for histogram testing.
        data_original = data.copy()
        data = data[:last_transition]

        code = True
        if code:
            coef = res / vref
        else:
            coef = 1
        # print(f"Coefficient = {coef}.")

        # Perform linear regression using least squares.
        lr = linregress(x=data.vout, y=data.vin)
        # print(lr)

        fig, ax = plt.subplots(figsize=(15, 10))
        # print(perfect_adc_code(data.vout))
        # print(v_to_code(data.vout))
        ax.axis("equal")
        # Plot the voltages.
        limits = [0 * coef, 0.025 * coef]
        # limits = [1000, 1023]
        # limits = [0, 1024]
        ax.plot(data.vout * coef, data.vin * coef, label="Measured transfer function")
        ax.plot(
            data.vout * coef,
            (lr.slope * data.vout + lr.intercept) * coef,
            label="Fitted transfer function",
        )
        ax.plot(data.vout * coef, data.vout * coef, label="Ideal transfer function")
        ax.plot(
            data.vout * coef,
            Analyzer.perfect_adc_code(data.vout, res, vref) * coef,
            label="Perfect transfer function",
        )
        ax.plot(
            data.vout * coef, data.vout_meas * coef, label="Measured voltage"
        )
        ax.plot(
            data.vout * coef,
            Analyzer.perfect_adc_code(lr.slope * data.vout + lr.intercept, res, vref)
            * coef,
            label="Perfect fitted transfer function",
        )

        # Set the limits to display only a part of the plot.
        ax.set_xlim(limits)
        ax.set_ylim(limits)

        # Set the legend.
        ax.legend()

        # Set the title.
        ax.set_title("Ideal, real, and best-fit ADC transfer functions; measured input voltage")

        # Calculate offset and gain errors.
        lsb_v = vref / res
        # print(lsb_v)

        def calculate_fit_error(lsbs, lsb_v, slope, intercept, verbose=False):
            offset_y = lsbs * lsb_v
            ideal_x = offset_y
            offset_x = (offset_y - intercept) / slope
            if verbose:
                print(f"Offset_x: {offset_x * coef}, offset_y (==ideal_x): {offset_y}.")
            return offset_x - ideal_x

        # Offset error calculated at 0.5 LSB.
        offset_error_v = calculate_fit_error(0.5, lsb_v, lr.slope, lr.intercept)
        # Full-scale error calculated at MAX - 1.5 LSB.
        fs_error_v = calculate_fit_error(res - 1.5, lsb_v, lr.slope, lr.intercept)
        # Gain error
        gain_error_v = -(fs_error_v - offset_error_v)

        offset_error = offset_error_v * coef
        fs_error = fs_error_v * coef
        gain_error = gain_error_v * coef
        print(
            f"Offset error = {offset_error} LSB, full-scale error: {fs_error} LSB, gain error = {gain_error} LSB."
        )

        # plt.show()
        plt.savefig("errors." + extension)

        # Histogram testing
        # TODO: Figure out why using the measured voltage causes the INL to accumulate to an unreasonably high value.
        # (right now, changed ".vout_meas" to just ".vout")
        perfect_codes = (
            np.maximum(
                np.round(
                    Analyzer.perfect_adc_code(
                        lr.slope * data_original.vout + lr.intercept, res, vref
                    )
                    * (res / vref)
                ),
                0,
            )
        ).astype(int)
        plt.figure(figsize=(15, 5))
        perfect_hist = plt.hist(perfect_codes, bins=res, range=(0, res))
        plt.xlabel("ADC code")
        plt.ylabel("Count")
        plt.title("Histogram of the codes of an ideal ADC corresponding to a linear ramp input")
        plt.savefig("perfect_histogram." + extension)
        # print(perfect_hist)

        plt.figure(figsize=(15, 5))
        real_hist = plt.hist(data_original.code.astype(int), bins=res, range=(0, res))
        plt.xlabel("ADC code")
        plt.ylabel("Count")
        plt.title("Histogram of the codes of the real ADC corresponding to a linear ramp input")
        plt.savefig("actual_histogram." + extension)

        # DNL
        dnls = real_hist[0] / perfect_hist[0] - 1
        # dnl = np.nanmax(np.abs(dnls[np.isfinite(dnls)]))
        dnl = dnls[np.abs(dnls[np.isfinite(dnls)]).argmax()]
        # print(f"Differential non-linearity: {round(dnl, 2)} LSB.")

        # INL
        inls = np.zeros_like(dnls)
        for i in range(len(dnls)):
            inls[i] = np.sum(dnls[: i + 1])

        # inl = np.nanmax(np.abs(inls[np.isfinite(inls)]))
        inl = inls[np.abs(inls[np.isfinite(inls)]).argmax()]
        # print(f"Integral non-linearity: {round(inl, 2)} LSB.")


        # Plot DNL and INL.
        Analyzer.plot(
            y=dnls, title="Differential non-linearity", xlabel="ADC code", ylabel="LSB", legend="DNL", filename="dnl.png"
        )
        plt.savefig("dnl." + extension)
        Analyzer.plot(
            y=inls, title="Integral non-linearity", xlabel="ADC code", ylabel="LSB", legend="INL", filename="inl.png"
        )
        plt.savefig("inl." + extension)

        # Also calculate the TUE.
        tue = np.sqrt(0.5 ** 2 + offset_error ** 2 + gain_error ** 2 + inl ** 2)

        # Print out all the errors found so far.
        print(
            f"""Offset error = {round(offset_error, 2)} LSB, 
        Full-scale error: {round(fs_error, 2)} LSB, 
        Gain error = {round(gain_error, 2)} LSB,
        Differential non-linearity = {round(dnl, 2)} LSB,
        Integral non-linearity = {round(inl, 2)} LSB.
        Total unadjusted error = {round(tue, 2)}"""
        )

        return offset_error, gain_error, dnl, inl, tue

    def plot_input_referred_noise(
        filename: str, output_filename="input_noise", extension="png"
    ):
        input_noise_data = Analyzer.read_data(filename)

        # In case there are any spurious values, remove them.
        input_noise_data = input_noise_data[input_noise_data["code"] >= 0]

        plt.figure(figsize=(15, 5))
        center_code = input_noise_data.code.mode().values[0]
        input_noise_data.code -= center_code

        mean, std_dev = norm.fit(input_noise_data.code.astype(int))

        values = input_noise_data.code.astype(int)

        bins = list(np.sort(values.unique()) - 0.5)
        bins.append(np.max(bins) + 1)
        hist = plt.hist(
            input_noise_data.code.astype(int), bins=bins, density=True, label="PMF"
        )

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std_dev)
        plt.plot(x, p, "k", linewidth=2, label="PDF")

        plt.xlabel("ADC code")
        plt.ylabel("Probability mass / density")
        plt.legend()
        plt.title("Input-referred noise")
        plt.savefig(f"{output_filename}.{extension}")

    def perform_dynamic_analysis(
        filename: str, output_filename="dynamic_analysis", extension="png"
    ):
        data = Analyzer.read_data(filename)

        n_samples = len(data)
        fs = 2800
        frequency = 280
        periods = n_samples / fs * frequency
        output_data = Analyzer.sine(n_samples, periods, 0, 4.6)

        fft_ideal = Analyzer.fft_real(output_data, power=True)
        fft_actual = Analyzer.fft_real(
            Analyzer.code_to_v(data.code, data.vref), power=True
        )

        # Signal power
        idx_fundamental = int(frequency * n_samples / fs)
        idx_fundamental
        fundamental_power = fft_actual[idx_fundamental]
        signal_power = fundamental_power

        # Harmonics' power
        f_harmonics = np.array([frequency * i for i in range(2, 6)])
        f_harmonics
        idx_harmonics = f_harmonics * n_samples / fs
        idx_harmonics = idx_harmonics.astype(np.int64)
        harmonics_powers = fft_actual[idx_harmonics]
        harmonics_power = np.sum(harmonics_powers)

        # Noise power
        idx_noise = list(
            range(1, len(fft_actual))
        )  # Start from 1 to exclude the DC component.
        idx_noise.remove(idx_fundamental)
        for harmonics_i in idx_harmonics:
            idx_noise.remove(harmonics_i)
        noise_powers = fft_actual[idx_noise]
        noise_power = np.sum(noise_powers)

        # THD
        thd = fundamental_power / harmonics_power
        thd = 10 * np.log10(thd)

        # SNR
        snr = 10 * np.log10(signal_power / noise_power)

        # Ideal SNR (just for comparison)
        snr_ideal = 6.02 * 10 + 1.76

        # SFDR
        powers_idx = list(np.flip(np.argsort(fft_actual)))
        powers_idx.remove(0)  # Remove the DC component.
        powers_idx.remove(idx_fundamental)  # Remove the signal.
        spurious_idx = powers_idx[0]
        sfdr = fundamental_power / fft_actual[spurious_idx]
        sfdr = 10 * np.log10(sfdr)

        # SINAD
        sinad = signal_power / (noise_power + harmonics_power)
        sinad = 10 * np.log10(sinad)

        # ENOB
        enob = (sinad - 1.76) / 6.02

        # Noise floor
        noise_floor = np.mean(noise_powers)
        noise_floor = 10 * np.log10(signal_power / noise_floor)

        # Combine (almost) everything and save the resulting figure.
        resolution = fs / n_samples
        x = [x * resolution for x in range(len(fft_actual))]

        plt.figure(figsize=(15, 10))

        fft_ideal_dbfs = Analyzer.to_decibels(fft_ideal, power=True)
        fft_actual_dbfs = Analyzer.to_decibels(fft_actual, power=True)
        plt.plot(x, fft_actual_dbfs, label="Power spectrum")

        plt.axhline(-snr, label="SNR", color="tab:orange", linestyle="--", linewidth=3)
        plt.axhline(-thd, label="THD", color="tab:red", linestyle="--")
        plt.axhline(
            -sinad,
            label="SINAD",
            color="tab:green",
            linestyle=":",
            alpha=1,
            linewidth=2,
        )
        plt.axhline(-sfdr, label="SFDR", color="tab:purple", linestyle="--")
        plt.axhline(
            -noise_floor, label="Noise floor", color="tab:olive", linestyle="--"
        )

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dBFS)")
        plt.legend()
        plt.title("ADC power spectrum with various dynamic parameters")
        plt.savefig(f"{output_filename}.{extension}")

        return snr, snr_ideal, thd, sfdr, sinad, enob, noise_floor


if __name__ == "__main__":
    Analyzer.print_results("linear_ramp_sw_1.csv", "sine_1.csv", "input_noise_1.csv")
