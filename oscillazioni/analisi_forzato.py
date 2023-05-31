import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# configuration parameters
MASS = 140.87 #g
SIGMA_MASS = 0.49 #g
PERIOD = 0.803683 #s
SIGMA_PERIOD = 0.01 #s
PULSATION = 7.817988 #rad/s
SIGMA_PULSATION = 0.097277 #rad/s
FREQUENCY = 1.244271 #Hz
SIGMA_FREQUENCY = 0.015482 #Hz
GAMMA = 0.002524 #1/s
SIGMA_GAMMA = 0.000722 #1/s

# function which finds index of last element of array less then given value
def find_last_index_greater_than(arr, target):
    result = -1

    for i in range(len(arr)):
        if arr[i] > target:
            result = i

    return result

# function which finds index of first element of array less then given value
def find_first_index_greater_than(arr, target):
    result = -1

    for i in range(len(arr)):
        if arr[i] > target:
            result = i
            break

    return result

def max(array):
    
    max_value = array[0]
    
    for i in range(1, len(array)):
        
        if array[i] > max_value:
            
            max_value = array[i]
    
    return max_value

# functions which calculates the mean of the elements of a dictionary
def dic_mean(dic):
    
    n = 0
    sum = 0
    
    for key in dic:
        
        sum += dic[key]
        n += 1
    
    return (sum / n)

def chi_q(arr_obs, arr_th, c):
    
    chi = 0
    
    for i in range(len(arr_obs)):
        
        chi += ((arr_obs[i] - arr_th[i])**2) / arr_th[i]
    
    return (chi / (len(arr_obs) - c - 1))

# analysis function

def analysis(path_lorentz, lim_min, lim_max):
    
    # import csv file for amplitudes
    lorentziana = pd.read_csv(path_lorentz, sep=';')
    
    # convert DataFrames to numpy array
    amplitudes = lorentziana['amplitude'].to_numpy()
    frequencies = lorentziana['frequency'].to_numpy()

    ampl = amplitudes[::-1]
    freq = frequencies[::-1]

    err_ampl = 0.003
    err_freq = 0.001
    
    # evaluate the force acted on the spring
    amp_max = max(ampl)
    i_max, = np.where(ampl == amp_max)
    force = 2 * PULSATION * MASS * GAMMA * amp_max
    
    #defining lorentz's distribution
    def lorentz(freq):
        return (force / MASS) / (2 * PULSATION * np.sqrt((2 * np.pi * freq - PULSATION)**2 + GAMMA**2))
    
    # FWHM
    half_width = amp_max / 2

    print(half_width)

    i_half_1 = find_first_index_greater_than(ampl, half_width)
    i_half_2 = find_last_index_greater_than(ampl, half_width)

    print(i_half_1)
    print(i_half_2)

    freq_fwhm_1 = (freq[i_half_1] + freq[i_half_1-1]) / 2
    freq_fwhm_2 = (freq[i_half_2] + freq[i_half_2+1]) / 2

    fwhm = 2 * np.pi * (freq_fwhm_2 - freq_fwhm_1)

    gamma_fwhm = fwhm / (2 * np.sqrt(3))
    force_fwhm = 2 * PULSATION * MASS * gamma_fwhm * amp_max

    print("\nGamma fwhm:", round(gamma_fwhm,6), "\n")

    #defining lorentz's distribution with new gamma
    def lorentz_fwhm(freq):
        return (force_fwhm / MASS) / (2 * PULSATION * np.sqrt((2 * np.pi * freq - PULSATION)**2 + gamma_fwhm**2))

    # linspace for plotting
    freq_space = np.linspace(1.15, 1.3, 1000)
    
    # theoretical values of distribution
    ampl_theo = lorentz(freq)
    
    # normalization
    n = {}
    
    for i in range(1, len(ampl)):
            for j in range(i):
                if (((i - j) >= (lim_min-1) and (i - j) <= (lim_max)) and (i_max > j and i_max < i)):
                    n[(j,i)] = sum(ampl_theo[j:i+1]) / sum(ampl[j:i+1])
    
    with open("analisi/forzato/normalization.log", 'w') as textfile:
        for key in n:
            print(key[0], "-", key[1],":\tN: ", n[key], file=textfile)
    
    chi = {}
    
    for key in n:
        chi[key] = chi_q(n[key] * ampl, ampl_theo, 1)
    
    with open("analisi/forzato/normalization_chi.log", 'w') as textfile:
        for key in chi:
            print(key[0], "-", key[1],":\tN: ", chi[key], file=textfile)
    
    key_norm = min(chi, key=chi.get)
    n_norm = n[key_norm]
    chi_norm = chi[key_norm]
    print("Normalization constant:", n_norm)
    print("Chi squared:", chi_norm)
    print("Interval:", key_norm)

    # theoretical values of distribution
    ampl_theo_fwhm = lorentz_fwhm(freq)
    
    # normalization fwhm
    print("\nFWHM:")
    n = {}
    
    for i in range(1, len(ampl)):
            for j in range(i):
                if (((i - j) >= (lim_min-1) and (i - j) <= (lim_max)) and (i_max > j and i_max < i)):
                    n[(j,i)] = sum(ampl_theo_fwhm[j:i+1]) / sum(ampl[j:i+1])
    
    with open("analisi/forzato/normalization_fwhm.log", 'w') as textfile:
        for key in n:
            print(key[0], "-", key[1],":\tN: ", n[key], file=textfile)
    
    chi = {}
    
    for key in n:
        chi[key] = chi_q(n[key] * ampl, ampl_theo_fwhm, 1)
    
    with open("analisi/forzato/normalization_chi_fwhm.log", 'w') as textfile:
        for key in chi:
            print(key[0], "-", key[1],":\tN: ", chi[key], file=textfile)
    
    key_norm_fwhm = min(chi, key=chi.get)
    n_norm_fwhm = n[key_norm_fwhm]
    chi_norm_fwhm = chi[key_norm_fwhm]
    print("Normalization constant:", n_norm_fwhm)
    print("Chi squared:", chi_norm_fwhm)
    print("Interval:", key_norm_fwhm)
    
    plt.plot(freq_space, lorentz(freq_space), label="th")
    plt.plot(freq_space, lorentz_fwhm(freq_space), label="th fwhm")
    plt.errorbar(freq, ampl, err_ampl, err_freq, fmt='.', c='green', label="obs")
    plt.errorbar(freq, n_norm*ampl, n_norm*err_ampl, err_freq, fmt='.', c='red', label="obs norm")
    plt.errorbar(freq, n_norm_fwhm*ampl, n_norm_fwhm*err_ampl, err_freq, fmt='.', c='black', label="obs norm fwhm")
    plt.xlabel("ω [rad/s]")
    plt.ylabel("A [mm]")
    plt.legend()
    plt.grid()
    plt.title("Lorentz's distribution")
    plt.savefig("analisi/forzato/plot_lorentz.png", dpi=1200)
    plt.close()

analysis('dati/forzato/lorentziana.csv', 5, 10)