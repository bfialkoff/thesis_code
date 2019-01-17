import pickle
from pathlib import Path

import numpy as np

class Converter:
    FORCE = np.array([50, 100, 262.5, 500, 1000, 2000, 4000, 7000, 10000])
    RESISTANCE = np.array([10, 6.5, 3.33, 2.182, 1.227, 0.77, 0.471, 0.335, 0.25])


    def _get_line_coeffs(self):
        path_to_line_coefficients = Path(__file__).joinpath('..', '..', '..', 'files', 'line_coeffs.pkl').resolve()
        if not path_to_line_coefficients.exists():
            a, b = np.polyfit(np.log10(self.RESISTANCE), np.log10(self.FORCE, 1))
            with open(path_to_line_coefficients, 'wb') as f:
                pickle.dump((a, b), f)
        else:
            with open(path_to_line_coefficients, 'rb') as f:
                a, b = pickle.load(f)
        return a, b

    def convert_fsr_voltage_to_resistance(self, fsr_voltage):
        Vcc = 5
        Vreg = 2.5
        units_of_resitance = 10 ** 3
        fsr_resistance = (fsr_voltage - Vreg) / (Vreg - Vcc) * units_of_resitance
        return fsr_resistance

    def convert_fsr_voltage_to_force(self, fsr_voltage):
        a, b = self._get_line_coeffs()
        fsr_resistance = self.convert_fsr_voltage_to_resistance(fsr_voltage)
        log_force = a * np.log10(fsr_resistance) + b
        return np.power(10, log_force)