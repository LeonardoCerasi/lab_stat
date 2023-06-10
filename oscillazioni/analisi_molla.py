import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

G = 9.80665

# functions which calculates the mean of the elements of an array
def arr_mean(array):
    
    sum = 0
    
    for i in range(len(array)):
        
        sum += array[i]
    
    return (sum / len(array))

# functions which calculates the standard deviation of the elements of an array
def arr_dev_st_c(array):
    
    sum = 0
    mean = arr_mean(array)
    
    for i in range(len(array)):
        
        sum += (array[i] - mean)**2
    
    return np.sqrt(sum / ((len(array) - 1) * len(array)))

# linear regression
def linear_regression(x, y, y_err):
    
    nu = len(x) - 2
    
    sum_p = 0
    sum_x = 0
    sum_xx = 0
    sum_y = 0
    sum_yy = 0
    sum_xy = 0
    
    for i in range(len(x)):
        
        sum_p += 1 / (y_err[i])**2
        sum_x += x[i] / (y_err[i])**2
        sum_xx += (x[i])**2 / (y_err[i])**2
        sum_y += y[i] / (y_err[i])**2
        sum_yy += (y[i])**2 / (y_err[i])**2
        sum_xy += x[i] * y[i] / (y_err[i])**2
    
    delta_x = sum_p * sum_xx - (sum_x)**2
    delta_y = sum_p * sum_yy - (sum_y)**2
    
    intercept = (sum_xx * sum_y - sum_x * sum_xy) / delta_x
    slope = (sum_p * sum_xy - sum_x * sum_y) / delta_x
    
    sigma_intercept = np.sqrt(sum_xx / delta_x)
    sigma_slope = np.sqrt(sum_p / delta_x)
    
    r = slope * np.sqrt(delta_x / delta_y)
    
    return {'slope': slope, 'inter': intercept, 's_slp': sigma_slope, 's_int': sigma_intercept, 'nu': nu, 'r': r}

# chi squared
def chi_q(x, y, y_err, regression):
    
    chi_squared = 0
    
    for i in range(len(x)):
        
        chi = ((y[i] - x[i] * regression['slope'] - regression['inter']) / y_err[i])**2
        
        chi_squared += chi
    
    return (chi_squared / regression['nu'])

# analysis function
def analysis_statico_calibro(path_masse, path_dati, int_configurazione, massa_gancio, massa_molla):
    
    print("\n\nAnalisi statico calibro:")

    print("\nConfiguration "+str(int_configurazione)+":")
    
    # import csv files for data as DataFrames
    heigths = pd.read_csv(path_dati, sep=';')
    masses = pd.read_csv(path_masse, sep=';')
    
    # convert DatFrames of data to numpy arrays
    h1 = heigths['h1'].to_numpy()
    h2 = heigths['h2'].to_numpy()
    h3 = heigths['h3'].to_numpy()
    mass = masses['mass'].to_numpy()
    
    # calculating average heigths
    heigth = []
    sigma_heigth = []
    for i in range(len(h1)):
        heigth.append(arr_mean([h1[i], h2[i], h3[i]]))
        sigma_heigth.append(arr_dev_st_c([h1[i], h2[i], h3[i]]))
    
    # calculating heigth and mass differences
    heigth_diff = []
    sigma_heigth_diff = []
    mass_diff = []
    
    if (int_configurazione == 1):
    
        for i in range(1,len(heigth)):
            heigth_diff.append(heigth[0] - heigth[i])
            sigma_heigth_diff.append(np.sqrt((sigma_heigth[0])**2 + (sigma_heigth[i])**2))
            mass_diff.append(sum(mass[0:i]))
    
    elif (int_configurazione == 2):
        
        for i in range(1,len(heigth)):
            heigth_diff.append(heigth[i] - heigth[0])
            sigma_heigth_diff.append(np.sqrt((sigma_heigth[0])**2 + (sigma_heigth[i])**2))
            mass_diff.append(sum(mass[1:i+1]))
    
    # calculating k test
    k_poss = {}
    
    if (int_configurazione == 1):
    
        for i in range(1, len(heigth_diff)):
            for j in range(0, i):
                if ((i - j) >= 5):
                    k_poss[(j+1,i+1)] = G * sum(mass[j+1:i+1]) / (heigth[j+1] - heigth[i+1])
    
    elif (int_configurazione == 2):
        
        for i in range(1, len(heigth_diff)):
            for j in range(0, i):
                if ((i - j) >= 4):
                    k_poss[(j+2,i+2)] = G * sum(mass[j+2:i+2]) / (heigth[i+1] - heigth[j+1])
    
    key_k = max(k_poss, key=k_poss.get)
    k_test = k_poss[key_k]
    print("\nTest k:", round(k_test,6), key_k)
    
    # calculate error on mass differences
    sigma_mass_diff = []
    
    for i in range(len(sigma_heigth_diff)):
        sigma_mass_diff.append((k_test / G) * sigma_heigth_diff[i])
        
    # linear regression
    regression = linear_regression(heigth_diff, mass_diff, sigma_mass_diff)
    
    print("\nLinear regression:")
    for key in regression:
        print(key, ":\t", regression[key])
        
    # chi squared analysis
    chi = chi_q(heigth_diff, mass_diff, sigma_mass_diff, regression)
    print("\nχ²:", chi)
    
    #evaluate k
    k = G * regression['slope']
    sigma_k = G * regression['s_slp']
    
    print("\nk:", round(k,6))
    print("sigma_k: ", round(sigma_k,6))
    
    # define domain of linear regression
    x = np.linspace(0, max(heigth_diff)+20, 1000)
    
    # define regression line
    def reg_lin(x):
        return (regression['slope'] * x + regression['inter'])
    
    # plot data and line
    plt.plot(x, reg_lin(x), color='blue', lw = 0.5)
    plt.errorbar(heigth_diff, mass_diff, sigma_mass_diff, sigma_heigth_diff, fmt='.', ecolor='red')
    plt.grid()
    plt.xlabel("Δh [mm]")
    plt.ylabel('Δm [g]')
    plt.title('Linear regression for configuration: statico calibro '+str(int_configurazione))
    plt.savefig("analisi/molla/plot_statico_calibro_"+str(int_configurazione)+".png", dpi=1200)
    plt.close()

analysis_statico_calibro('dati/molla/mass.csv', 'dati/molla/statico_calibro_1.csv', 1, 19.74, 12.43)
analysis_statico_calibro('dati/molla/mass.csv', 'dati/molla/statico_calibro_2.csv', 2, 19.74, 22.40)

# analysis function
def analysis_statico_sensore(path_masse, path_dati, int_configurazione, massa_gancio, massa_molla):
    
    print("\n\nAnalisi statico sensore:")

    print("\nConfiguration "+str(int_configurazione)+":")
    
    # import csv files for data as DataFrames
    heigths = pd.read_csv(path_dati, sep=';')
    masses = pd.read_csv(path_masse, sep=';')
    
    if (int_configurazione == 1):
    
        # convert DatFrames of data to numpy arrays
        h0 = heigths['h0'].to_numpy()
        h1 = heigths['h1'].to_numpy()
        h2 = heigths['h2'].to_numpy()
        h3 = heigths['h3'].to_numpy()
        h4 = heigths['h4'].to_numpy()
        h5 = heigths['h5'].to_numpy()
        h6 = heigths['h6'].to_numpy()
        h6p = heigths['h6p'].to_numpy()
        h7 = heigths['h7'].to_numpy()
        h8 = heigths['h8'].to_numpy()
        mass = masses['mass'].to_numpy()

        # calculating average heigths
        heigth_0 = 1000*arr_mean(h0)
        heigth_1 = 1000*arr_mean(h1)
        heigth_2 = 1000*arr_mean(h2)
        heigth_3 = 1000*arr_mean(h3)
        heigth_4 = 1000*arr_mean(h4)
        heigth_5 = 1000*arr_mean(h5)
        heigth_6 = 1000*arr_mean(h6)
        heigth_6p = 1000*arr_mean(h6p)
        heigth_7 = 1000*arr_mean(h7)
        heigth_8 = 1000*arr_mean(h8)

        sigma_heigth_0 = 1000*arr_dev_st_c(h0)
        sigma_heigth_1 = 1000*arr_dev_st_c(h1)
        sigma_heigth_2 = 1000*arr_dev_st_c(h2)
        sigma_heigth_3 = 1000*arr_dev_st_c(h3)
        sigma_heigth_4 = 1000*arr_dev_st_c(h4)
        sigma_heigth_5 = 1000*arr_dev_st_c(h5)
        sigma_heigth_6 = 1000*arr_dev_st_c(h6)
        sigma_heigth_6p = 1000*arr_dev_st_c(h6p)
        sigma_heigth_7 = 1000*arr_dev_st_c(h7)
        sigma_heigth_8 = 1000*arr_dev_st_c(h8)

        delta = heigth_6p - heigth_6
        heigth_0 += delta
        heigth_1 += delta
        heigth_2 += delta
        heigth_3 += delta
        heigth_4 += delta
        heigth_5 += delta

        # calculating heigth and mass differences
        heigth_diff = []
        sigma_heigth_diff = []
        mass_diff = []

        heigth_diff.append(heigth_0 - heigth_1)
        heigth_diff.append(heigth_0 - heigth_2)
        heigth_diff.append(heigth_0 - heigth_3)
        heigth_diff.append(heigth_0 - heigth_4)
        heigth_diff.append(heigth_0 - heigth_5)
        heigth_diff.append(heigth_0 - heigth_6p)
        heigth_diff.append(heigth_0 - heigth_7)
        heigth_diff.append(heigth_0 - heigth_8)

        sigma_heigth_diff.append(np.sqrt((sigma_heigth_0)**2 + (sigma_heigth_1)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_0)**2 + (sigma_heigth_2)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_0)**2 + (sigma_heigth_3)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_0)**2 + (sigma_heigth_4)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_0)**2 + (sigma_heigth_5)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_0)**2 + (sigma_heigth_6p)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_0)**2 + (sigma_heigth_7)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_0)**2 + (sigma_heigth_8)**2))

        for i in range(1,9):
                mass_diff.append(sum(mass[0:i]))

        # test k (from previous code)
        k_test = 4.733643
        print("\nTest k:", round(k_test,6))
    
    elif (int_configurazione == 2):
        
        # convert DatFrames of data to numpy arrays
        h1 = heigths['h1'].to_numpy()
        h2 = heigths['h2'].to_numpy()
        h3 = heigths['h3'].to_numpy()
        h4 = heigths['h4'].to_numpy()
        h5 = heigths['h5'].to_numpy()
        h6 = heigths['h6'].to_numpy()
        h7 = heigths['h7'].to_numpy()
        h8 = heigths['h8'].to_numpy()
        mass = masses['mass'].to_numpy()
        
        # calculating average heigths
        heigth_1 = 1000*arr_mean(h1)
        heigth_2 = 1000*arr_mean(h2)
        heigth_3 = 1000*arr_mean(h3)
        heigth_4 = 1000*arr_mean(h4)
        heigth_5 = 1000*arr_mean(h5)
        heigth_6 = 1000*arr_mean(h6)
        heigth_7 = 1000*arr_mean(h7)
        heigth_8 = 1000*arr_mean(h8)
        
        sigma_heigth_1 = 1000*arr_dev_st_c(h1)
        sigma_heigth_2 = 1000*arr_dev_st_c(h2)
        sigma_heigth_3 = 1000*arr_dev_st_c(h3)
        sigma_heigth_4 = 1000*arr_dev_st_c(h4)
        sigma_heigth_5 = 1000*arr_dev_st_c(h5)
        sigma_heigth_6 = 1000*arr_dev_st_c(h6)
        sigma_heigth_7 = 1000*arr_dev_st_c(h7)
        sigma_heigth_8 = 1000*arr_dev_st_c(h8)
        
        # calculating heigth and mass differences
        heigth_diff = []
        sigma_heigth_diff = []
        mass_diff = []
        
        heigth_diff.append(heigth_1 - heigth_2)
        heigth_diff.append(heigth_1 - heigth_3)
        heigth_diff.append(heigth_1 - heigth_4)
        heigth_diff.append(heigth_1 - heigth_5)
        heigth_diff.append(heigth_1 - heigth_6)
        heigth_diff.append(heigth_1 - heigth_7)
        heigth_diff.append(heigth_1 - heigth_8)
        
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_1)**2 + (sigma_heigth_2)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_1)**2 + (sigma_heigth_3)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_1)**2 + (sigma_heigth_4)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_1)**2 + (sigma_heigth_5)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_1)**2 + (sigma_heigth_6)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_1)**2 + (sigma_heigth_7)**2))
        sigma_heigth_diff.append(np.sqrt((sigma_heigth_1)**2 + (sigma_heigth_8)**2))
        
        for i in range(2,9):
                mass_diff.append(sum(mass[1:i]))
        
        # test k (from previous code)
        k_test = 7.724078
        print("\nTest k:", round(k_test,6))
    
    # check instrumental error
    for i in range(len(sigma_heigth_diff)):
        if (sigma_heigth_diff[i] < 0.2):
            sigma_heigth_diff[i] = 0.2
    
    # calculate error on mass differences
    sigma_mass_diff = []
    
    for i in range(len(sigma_heigth_diff)):
        sigma_mass_diff.append((k_test / G) * sigma_heigth_diff[i])
        
    # linear regression
    regression = linear_regression(heigth_diff, mass_diff, sigma_mass_diff)
    
    print("\nLinear regression:")
    for key in regression:
        print(key, ":\t", regression[key])
        
    # chi squared analysis
    chi = chi_q(heigth_diff, mass_diff, sigma_mass_diff, regression)
    print("\nχ²:", chi)
    
    #evaluate k
    k = G * regression['slope']
    sigma_k = G * regression['s_slp']
    
    print("\nk:", round(k,6))
    print("sigma_k: ", round(sigma_k,6))
    
    # define domain of linear regression
    x = np.linspace(0, max(heigth_diff)+20, 1000)
    
    # define regression line
    def reg_lin(x):
        return (regression['slope'] * x + regression['inter'])
    
    # plot data and line
    plt.plot(x, reg_lin(x), color='blue', lw = 0.5)
    plt.errorbar(heigth_diff, mass_diff, sigma_mass_diff, sigma_heigth_diff, fmt='.', ecolor='red')
    plt.grid()
    plt.xlabel("Δh [mm]")
    plt.ylabel('Δm [g]')
    plt.title('Linear regression for configuration: statico sensore '+str(int_configurazione))
    plt.savefig("analisi/molla/plot_statico_sensore_"+str(int_configurazione)+".png", dpi=1200)
    plt.close()

analysis_statico_sensore('dati/molla/mass.csv', 'dati/molla/statico_sensore_1.csv', 1, 19.74, 12.43)
analysis_statico_sensore('dati/molla/mass.csv', 'dati/molla/statico_sensore_2.csv', 2, 19.74, 22.40)

# analysis function
def analysis_dinamico_sensore(path_dati, int_configurazione):

    print("\n\nAnalisi dinamico sensore:")
    
    print("\nConfiguration "+str(int_configurazione)+":")
    
    # import csv files for data as DataFrames
    dati = pd.read_csv(path_dati, sep=';')
    
    # convert DatFrames of data to numpy arrays
    omega = dati['omega'].to_numpy()
    sigma_omega = dati['sigma_omega'].to_numpy()
    mass = dati['mass'].to_numpy()
    sigma_mass = dati['sigma_mass'].to_numpy()
    
    # linear regression
    regression = linear_regression(omega, mass, sigma_mass)
    
    print("\nLinear regression:")
    for key in regression:
        print(key, ":\t", regression[key])
    
    # chi squared analysis
    chi = chi_q(omega, mass, sigma_mass, regression)
    print("\nχ²:", chi)
    
    #evaluate k
    k = regression['slope'] / 1000
    sigma_k = regression['s_slp'] / 1000
    
    print("\nk:", round(k,6))
    print("sigma_k: ", round(sigma_k,6))
    
    # define domain of linear regression
    x = np.linspace(0, max(omega)+0.005, 1000)
    
    # define regression line
    def reg_lin(x):
        return (regression['slope'] * x + regression['inter'])
    
    # plot data and line
    plt.plot(x, reg_lin(x), color='blue', lw = 0.5)
    plt.errorbar(omega, mass, sigma_mass, sigma_omega, fmt='.', ecolor='red')
    plt.grid()
    plt.xlabel("1/ω²")
    plt.ylabel('m [g]')
    plt.title('Linear regression for configuration: dinamico '+str(int_configurazione))
    plt.savefig("analisi/molla/plot_dinamico_"+str(int_configurazione)+".png", dpi=1200)
    plt.close()

analysis_dinamico_sensore('dati/molla/dinamico_sensore_1.csv', 1)
analysis_dinamico_sensore('dati/molla/dinamico_sensore_2.csv', 2)