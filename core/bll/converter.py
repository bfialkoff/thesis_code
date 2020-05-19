import warnings
import pickle
from pathlib import Path

import numpy as np


class Converter:
    FORCE = np.array([0.05, 0.1, 0.2625, 0.5, 1, 2, 4, 7, 10])
    RESISTANCE = np.array([10, 6.5, 3.33, 2.182, 1.227, 0.77, 0.471, 0.335, 0.25])  # Ohms
    VCC = 5  # Volts
    VREG = 2.5  # Volts
    K_OHM = 10 ** 3
    RESISTOR = 10  # K_OHM

    def _get_fsr_line_coeffs(self):
        path_to_line_coefficients = Path(__file__).joinpath('..', '..', '..', 'files', 'line_coeffs.pkl').resolve()
        if not path_to_line_coefficients.exists():
            a, b = np.polyfit(np.log10(self.RESISTANCE), np.log10(self.FORCE), 1)
            with open(path_to_line_coefficients, 'wb') as f:
                pickle.dump((a, b), f)
        else:
            with open(path_to_line_coefficients, 'rb') as f:
                a, b = pickle.load(f)
        return a, b

    def fsr_voltage_to_resistance(self, fsr_voltage):
        fsr_resistance = self.RESISTOR * (fsr_voltage - self.VREG) / (self.VREG - self.VCC)
        return fsr_resistance

    def fsr_voltage_to_force(self, fsr_voltage):
        a, b = self._get_fsr_line_coeffs()
        fsr_resistance = self.fsr_voltage_to_resistance(fsr_voltage)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_force = a * np.log10(fsr_resistance) + b
        force = np.power(10, log_force)
        force = np.where(fsr_resistance < np.min(self.RESISTANCE), np.max(self.FORCE), force)
        force = np.where(fsr_resistance > np.max(self.RESISTANCE), np.min(self.FORCE), force)
        return force
