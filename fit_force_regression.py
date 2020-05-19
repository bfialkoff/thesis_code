import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from tqdm import tqdm
import matplotlib
#matplotlib.use('Agg')

from core.bll.features import Features
from core.bll.converter import Converter
from core.bll.preprocessing import Preprocessor
from core.bll.data_reader import DataReader
from core.utils.plot import *
from core.utils.array import permute_axes_subtract


path_to_signals = Path(__file__).joinpath('..', 'files', 'signals').resolve()
my_signal = path_to_signals.joinpath('signal1').resolve()

# TODO write a script that will generate a subplot for each channel containing plot_regression_line and
# doesnt work
def poly_n(x, *coeffs):
    powers = np.arange(len(coeffs)- 1, -1, -1)
    f = 0
    for c, p in zip(coeffs, powers):
        f += c * x ** p
    f = np.where(f < Features.MAX_FORCE, f, Features.MAX_FORCE)
    #f[0] = 0
    return f

def poly_7(x, a7, a6, a5, a4, a3, a2, a1, a0):
    return poly_n(x, a7, a6, a5, a4, a3, a2, a1, a0)

def poly_6(x, a6, a5, a4, a3, a2, a1, a0):
    return poly_n(x, a6, a5, a4, a3, a2, a1, a0)

def poly_5(x, a5, a4, a3, a2, a1, a0):
    return poly_n(x, a5, a4, a3, a2, a1, a0)

def sum_power(x, a1, b1, a2, b2):
    p1 = a1 * x ** b1
    p2 = a2 * x ** b2
    f = p1 + p2
    f = np.where(f < Features.MAX_FORCE, f, Features.MAX_FORCE)
    return f
    

def sum_exp(x, a1, b1, a2, b2, a3, b3):
    e1 = a1 * np.exp(b1 * x)
    e2 = a2 * np.exp(b2 * x)
    e3 = a3 * np.exp(b3 * x)
    f = e1 + e2 + e3
    f = np.where(f < Features.MAX_FORCE, f, Features.MAX_FORCE)
    return f
    
def sum_sigmoid(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4):
    sigmoid_1 = a1 / (1 + c1 * np.exp(-b1 * x))
    sigmoid_2 = a2 / (1 + c2 * np.exp(-b2 * x))
    sigmoid_3 = a3 / (1 + c3 * np.exp(-b3 * x))
    sigmoid_4 = a4 / (1 + c4 * np.exp(-b4 * x))
    f = sigmoid_1 + sigmoid_2 + sigmoid_3 + sigmoid_4
    f = np.where(f < Features.MAX_FORCE, f, Features.MAX_FORCE)
    return f 

if __name__ == '__main__':
    features = Features()
    preprocessor = Preprocessor()
    converter = Converter()
    sampling_frequency = 1980
    
    degree = 6
    fit_func = poly_6
    path = Path(__file__).joinpath('..', 'files', f'regressed_force_signals_deg_{degree}').resolve()
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)
    csv_path = path.joinpath('..', 'poly_6_coeffs.csv').resolve()
    scaler_path = path.joinpath('..', 'poly_6_coeffs_scaler.pkl').resolve()
    if not csv_path.exists():
        csv_file = open(csv_path, 'w')
        header = 'signal_number,a6,a5,a4,a3,a2,a1,a0\n'
        csv_file.write(header)
        for signal in tqdm(range(132)):
            save_path = path.joinpath(f'signal_{signal}.png').resolve()
            data_reader = DataReader(signal)
            fsr_voltage = data_reader.get_fsr_voaltage_signal()

            raw_force = converter.fsr_voltage_to_force(fsr_voltage)
            force = preprocessor.process_force_signal(raw_force, sampling_frequency)
            t = np.arange(len(force)) / sampling_frequency
            #bounds = (degree * [-np.inf] + [0], (degree + 1) * np.inf)
            try:
                popt, _ = curve_fit(fit_func, t, force)#, bounds=(4*[-5], 4*[5]))
                str_coef = ','.join(np.around(popt, 2).astype(str).tolist())
            except:
                print(f'cant fit {signal}')
                continue
            row = '{}'.format(signal) + (len(popt) * ',{:.5f}').format(*popt) + '\n'
            csv_file.write(row)

            fitted_force = fit_func(t, *popt)
            f, ax = plt.subplots(1, 1)
            ax.plot(t, force, 'k', label='force')
            ax.plot(t, fitted_force, 'g', label=f'regressed force deg {degree}\nc:{str_coef}')
            ax.legend()
            
            plt.savefig(save_path)
            plt.close(f)

        csv_file.close()
    coeff_data = pd.read_csv(csv_path)
    number_cols = list(set(coeff_data.columns).difference(['signal_number']))
    scaler = StandardScaler()
    scaler.fit(coeff_data[number_cols])

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
