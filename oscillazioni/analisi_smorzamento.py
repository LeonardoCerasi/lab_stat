import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2, t

# function which returns the number of occurrences of each element of an array
def count_occurrences(array):
    
    unique_elements, counts = np.unique(array, return_counts=True)
    
    occurrences = dict(zip(unique_elements, counts))
    
    return occurrences

# find key of lowest value of a dictionary
def min_dic(dic):
    
    min_value = min(dic.values())

    for key, value in dic.items():

        if value == min_value:

            return key

def max_arr(array):
    
    max_value = array[0]
    
    for i in range(1, len(array)):
        
        if array[i] > max_value:
            
            max_value = array[i]
    
    return max_value

def min_arr(array):
    
    min_value = array[0]
    
    for i in range(1, len(array)):
        
        if array[i] < min_value:
            
            min_value = array[i]
    
    return min_value

# function which returns the local maximums positions and their time value
def local_max(array_x, array_y, fraction):
    
    max_pos = {}
    
    if (fraction == 0):
        delta = 1
    
    else:
        delta = int(len(array_y) / fraction)
    
    i = 1
    j = 1
    while i < (len(array_y)-1):
        
        i_min = i - delta
        i_max = i + delta
        
        if (i_min < 0):    
            i_min = 0
        
        if (i_max > len(array_y)):
            i_max = len(array_y)
        
        if (array_y[i] == max_arr(array_y[i_min:i_max+1])):
        
            max_pos[j] = [array_x[i], array_y[i]]
            j += 1
        
            i = i_max - 1
        
        i += 1
    
    return max_pos

# function which returns the local minimums positions and their time value
def local_min(array_x, array_y, fraction):
    
    min_pos = {}
    
    if (fraction == 0):
        delta = 1
    
    else:
        delta = int(len(array_y) / fraction)
    
    i = 1
    j = 1
    while i < (len(array_y)-1):
        
        i_min = i - delta
        i_max = i + delta
        
        if (i_min < 0):    
            i_min = 0
        
        if (i_max > len(array_y)):
            i_max = len(array_y)
        
        if (array_y[i] == min_arr(array_y[i_min:i_max+1])):
        
            min_pos[j] = [array_x[i], array_y[i]]
            j += 1
        
            i = i_max - 1
        
        i += 1
    
    return min_pos

# function which returns the matrix of all the possible periods of oscillations
def periods(array_extremes):
    
    p_osc = {}
    
    for key_1 in array_extremes:
        
        for key_2 in array_extremes:
        
            if (key_2 > key_1):
                
                p_osc[(key_1, key_2)] = ((array_extremes[key_2][0] - array_extremes[key_1][0]) / (key_2 - key_1))
    
    return p_osc

# functions which calculates the mean of the elements of a dictionary
def dic_mean(dic):
    
    n = 0
    sum = 0
    
    for key in dic:
        
        sum += dic[key]
        n += 1
    
    return (sum / n)

# functions which calculates the standard deviation of the elements of a dictionary
def dic_dev_st_c(dic):
    
    sum = 0
    mean = dic_mean(dic)
    
    for key in dic:
        
        sum += (dic[key] - mean)**2
    
    return np.sqrt(sum / ((len(dic.keys()) - 1) * len(dic.keys())))

# functions which calculates the mean of the elements of an array
def arr_mean(array):
    
    sum = 0
    
    for i in range(len(array)):
        
        sum += array[i]
    
    return (sum / len(array))

# functions which calculates the standard deviation of the elements of an array
def arr_dev_st(array):
    
    sum = 0
    mean = arr_mean(array)
    
    for i in range(len(array)):
        
        sum += (array[i] - mean)**2
    
    return np.sqrt(sum / (len(array) - 1))

# function which returns the highest n values of an array
def find_highest(arr, n):
    
    arr = np.array(arr)
    
    sorted_indices = np.argpartition(arr, -n)[-n:]
    
    highest_values = arr[sorted_indices]
    
    return highest_values

# function which finds index of last element of array less then given value
def find_last_index_less_than(arr, target):
    result = -1

    for i in range(len(arr)):
        if arr[i] < target:
            result = i

    return result

# function which finds index of first element of array less then given value
def find_first_index_less_than(arr, target):
    result = -1

    for i in range(len(arr)):
        if arr[i] < target:
            result = i
            break

    return result

# find index of an element of an array
def first_index(array, value):
    
    i = 0
    
    while (i < len(array)):
        
        if (array[i] == value):
            
            return i
        
        i += 1
    
    return -1

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

# analysis funtion

def analysis(path_equilibrio, path_oscillazioni, int_configurazione, frequency, plot_color, extr_color, n_bins, key_max_1, key_max_2, n_1, n_2):

    # import csv file for equilibrium position as DataFrames
    equilibrio = pd.read_csv(path_equilibrio, sep=';')
    # import csv file for oscillating positions as DataFrames
    oscillazione = pd.read_csv(path_oscillazioni, sep=';')

    # convert DataFrames of equilibrium positions to numpy arrays
    eq_time = equilibrio['time'].to_numpy()
    eq_pos = equilibrio['position'].to_numpy()
    
    print("\n\n\nConfiguration "+str(int_configurazione)+":")

    # plot equilibrium positions
    plt.scatter(eq_time, eq_pos, marker='.', s=1, c=plot_color)
    plt.grid()
    plt.xlabel('t [s]')
    plt.ylabel('h [m]')
    plt.savefig('analisi/smorzamento/'+str(int_configurazione)+'/plot_equilibrio_'+str(int_configurazione)+'.png', dpi=1200)
    plt.close()

    # plot histogran of equilibrium positions
    fig, axs= plt.subplots(1, 1, tight_layout = True)
    axs.hist(eq_pos, bins = 200, color=plot_color)
    plt.grid()
    plt.xlabel('h [m]')
    plt.ylabel('counts')
    plt.savefig('analisi/smorzamento/'+str(int_configurazione)+'/plot_equilibrio_'+str(int_configurazione)+'_hist.png', dpi=1200)
    plt.close()


    # print number of equilibrium positions
    eq_pos_dic = count_occurrences(eq_pos)
    with open('analisi/smorzamento/'+str(int_configurazione)+'/output_equilibrio_'+str(int_configurazione)+'_hist.log','w') as textfile:
        print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)
        for pos in eq_pos_dic:
            print("x = ", pos, "\t n: ", eq_pos_dic[pos], file=textfile)

    # mean equilibrium position
    total_counts = 0
    mean_eq_pos_counts = 0
    for pos in eq_pos_dic:
        total_counts += eq_pos_dic[pos]
        mean_eq_pos_counts += pos * eq_pos_dic[pos]
    
    mean_eq_pos = mean_eq_pos_counts / total_counts
    print("\nMean equilibrium position:", round(mean_eq_pos, 6))

    dev_eq_pos_counts = 0
    for pos in eq_pos_dic:
        dev_eq_pos_counts += eq_pos_dic[pos] * (pos - mean_eq_pos)**2
    
    dev_eq_pos = np.sqrt(dev_eq_pos_counts / total_counts)
    print("Error on mean eq. pos.:", round(dev_eq_pos, 6))

    # convert DatFrames of oscillating positions to numpy arrays
    osc_time = oscillazione['time'].to_numpy()
    osc_pos_s = oscillazione['position'].to_numpy()
    osc_pos = osc_pos_s - mean_eq_pos

    ## minimum separation between oscillating positions
    #diff = {}
    #diff_0 = {}
    #for i in range(1, len(osc_time)):
    #    if ((osc_time[i] <= 40.000) and ((osc_pos[i] - osc_pos[i-1]) != 0)):
    #        diff[(i-1, i)] = abs(osc_pos[i] - osc_pos[i-1])
    #    elif ((osc_time[i] <= 40.000) and ((osc_pos[i] - osc_pos[i-1]) == 0)):
    #        diff_0[(i-1, i)] = abs(osc_pos[i] - osc_pos[i-1])
    #
    #with open('analisi/smorzamento/'+str(int_configurazione)+'/output_oscillazioni_'+str(int_configurazione)+'_diff.log', 'w') as textfile:
    #    print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)

    #    for key in diff:
    #        print(key, "\t ", diff[key], file=textfile)
    #
    #with open('analisi/smorzamento/'+str(int_configurazione)+'/output_oscillazioni_'+str(int_configurazione)+'_diff_0.log', 'w') as textfile:
    #    print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)

    #    for key in diff_0:
    #        print(key, "\t ", diff_0[key], file=textfile)
    #
    #count = 0
    #for key in diff:
    #    if (diff[key] < 0.000021):
    #        count += 1
    #
    #print("\nMinimum separation:", diff[min_dic(diff)])
    #print("Problematic diff:", count)
    #print("Zero diff:", len(diff_0.keys()))
    #print("Mean diff:", round(dic_mean(diff), 6))


    # print maximum positions
    with open('analisi/smorzamento/'+str(int_configurazione)+'/output_oscillazioni_'+str(int_configurazione)+'_max.log', 'w') as textfile:
        print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)
        osc_max_dic = local_max(osc_time, osc_pos, 200)

        for key in osc_max_dic:
            print(key, "\t x: ", osc_max_dic[key][1], "\t t: ", osc_max_dic[key][0], file=textfile)
    
    # print minimum positions
    with open('analisi/smorzamento/'+str(int_configurazione)+'/output_oscillazioni_'+str(int_configurazione)+'_min.log', 'w') as textfile:
        print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)
        osc_min_dic = local_min(osc_time, osc_pos, 200)

        for key in osc_min_dic:
            print(key, "\t x: ", osc_min_dic[key][1], "\t t: ", osc_min_dic[key][0], file=textfile)
    
    # amplitude of oscillation
    amplitude = (osc_max_dic[1][1] - osc_min_dic[1][1]) / 2

    # dictionary of every extreme position
    osc_extr = {}
    
    if (len(osc_max_dic.keys()) > len(osc_min_dic.keys())):
        for key in osc_min_dic:
            osc_extr[2*key-1] = osc_max_dic[key]
            osc_extr[2*key] = osc_min_dic[key]
        osc_extr[2*len(osc_min_dic.keys())+1] = osc_max_dic[len(osc_min_dic.keys())+1]
    elif (len(osc_max_dic.keys()) < len(osc_min_dic.keys())):
        for key in osc_max_dic:
            osc_extr[2*key-1] = osc_min_dic[key]
            osc_extr[2*key] = osc_max_dic[key]
        osc_extr[2*len(osc_max_dic.keys())+1] = osc_min_dic[len(osc_max_dic.keys())+1]
    
    # print minimum positions
    with open('analisi/smorzamento/'+str(int_configurazione)+'/output_oscillazioni_'+str(int_configurazione)+'.log', 'w') as textfile:
        print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)
        
        for key in osc_extr:
            print(key, "\t x: ", osc_extr[key][1], "\t t: ", osc_extr[key][0], file=textfile)

    # evaluate period of oscillations
    with open('analisi/smorzamento/'+str(int_configurazione)+'/output_periodi_'+str(int_configurazione)+'.log', 'w') as textfile:
        print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)
        
        # using maximum positions
        print("(using maxima)\n", file=textfile)
        p_osc_max = periods(osc_max_dic)
        n_max = len(p_osc_max.keys())

        for i in range(1, len(osc_max_dic)-1):
            for j in range(i+1, len(osc_max_dic)):
                print(i, "-", j, "\t T: ", p_osc_max[(i, j)], file=textfile)

        osc_period_max = dic_mean(p_osc_max)
        err_osc_period_max = dic_dev_st_c(p_osc_max)
        print("\n\n\nThe mean period of oscillation is: ", osc_period_max, file=textfile)
        print("\n\n\nThe error is: ", err_osc_period_max, file=textfile)
        
        # using minimum positions
        print("\n(using minima)\n", file=textfile)
        p_osc_min = periods(osc_min_dic)
        n_min = len(p_osc_min.keys())

        for i in range(1, len(osc_min_dic)-1):
            for j in range(i+1, len(osc_min_dic)):
                print(i, "-", j, "\t T: ", p_osc_min[(i, j)], file=textfile)

        osc_period_min = dic_mean(p_osc_min)
        err_osc_period_min = dic_dev_st_c(p_osc_min)
        print("\n\n\nThe mean period of oscillation is: ", osc_period_min, file=textfile)
        print("\n\n\nThe error is: ", err_osc_period_min, file=textfile)
        
        # using both
        print("\n(half periods)", file=textfile)
        p_osc_extr = periods(osc_extr)

        for i in range(1, len(osc_extr)-1):
            for j in range(i+1, len(osc_extr), 2):
                print(i, "-", j, "\t T/2: ", p_osc_extr[(i, j)], file=textfile)

        osc_period_extr = dic_mean(p_osc_extr)
        err_osc_period_extr = dic_dev_st_c(p_osc_extr)
        print("\n\n\nThe mean period of oscillation is: ", 2*osc_period_extr, file=textfile)
        print("\n\n\nThe error is: ", 2*err_osc_period_extr, file=textfile)
    
    print("\n(max)")
    print("The mean period of oscillation is: ", osc_period_max)
    print("The error is: ", err_osc_period_max)
    print("\n(min)")
    print("The mean period of oscillation is: ", osc_period_min)
    print("The error is: ", err_osc_period_min)
    print("\n(mm)")
    print("The mean period of oscillation is: ", 2*osc_period_extr)
    print("The error is: ", 2*err_osc_period_extr)

    period = (osc_period_max + osc_period_min + (2*osc_period_extr))/3
    sigma_period = np.sqrt(err_osc_period_max**2 + err_osc_period_min**2 + (2*err_osc_period_extr)**2)/3

    period = (osc_period_max + osc_period_min)/2
    sigma_period = np.sqrt(err_osc_period_max**2 + err_osc_period_min**2)/2

    err_period = (sigma_period if (sigma_period >= 1/frequency) else (1/frequency))

    print("\n\n\n", sigma_period, "\n\n\n")

    t_s = np.sqrt(((n_max-1)*err_osc_period_max**2 + (n_min-1)*err_osc_period_min**2) / (n_max + n_min - 2))
    t_n = np.sqrt(((1/n_max) + (1/n_min)))
    t_d = np.abs(osc_period_max - osc_period_min)
    t_student = t_d / (t_n * t_s)
    t_nu = n_max + n_min - 2

    print("t_s:", t_s)
    print("t_n:", t_n)
    print("t_d:", t_d)
    print("t:", t_student)
    print("t_nu:", t_nu)
    print("P:", t.cdf(t_student, t_nu), "\n\n\n")
    
    ## evaluate equilibrium position
    #with open('analisi/smorzamento/'+str(int_configurazione)+'/output_equilibrio_'+str(int_configurazione)+'.log', 'w') as textfile:
    #    print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)
    #    
    #    x_eq = []
    #    
    #    for i in range(1, len(osc_extr)-1):
    #        x = (osc_extr[i][1] * osc_extr[i+2][1] - (osc_extr[i+1][1])**2) / (2 * osc_extr[i+1][1] + osc_extr[i][1] + osc_extr[i+2][1])
    #        x_eq.append(x)
    #        print(i, "-", (i+1), "-", (i+2), "\t x: ", x, file=textfile)
    #    
    #print("\nEquilibrium poisition of configuration "+str(int_configurazione)+":", round(arr_mean(x_eq),6))
    #print("Error on equilibrium positions: ", round(arr_dev_st(x_eq),6))
    #
    ## evaluate dumping
    #with open('analisi/smorzamento/'+str(int_configurazione)+'/output_smorzamento_'+str(int_configurazione)+'.log', 'w') as textfile:
    #    print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)
    #    
    #    s_mean = []
    #    
    #    for i in range(1, len(osc_extr)-1):
    #        s = (osc_extr[i+2][1] + osc_extr[i+1][1]) / (osc_extr[i+1][1] + osc_extr[i][1])
    #        gamma = - period / (2 * np.log(s))
    #        s_mean.append(s)
    #        print(i, "-", (i+1), "-", (i+2), "\t s: ", s, "\t gamma:", gamma, file=textfile)
    #    
    #print("\nDumping:", round(arr_mean(s_mean),6))
    #print("Error on dumping: ", round(arr_dev_st(s_mean),6))
    #gamma = - period / (2 * np.log(arr_mean(s_mean)))
    #print("Friction costant of configuration "+str(int_configurazione)+":", round(gamma,6))

    # plot oscillating positions
    plt.scatter(osc_time, osc_pos, marker='.', s=1, c=plot_color)
    for key in osc_extr:
        plt.scatter(osc_extr[key][0], osc_extr[key][1], marker='.', s=3, c=extr_color)
    plt.grid()
    plt.xlabel('t [s]')
    plt.ylabel('x [m]')
    plt.savefig('analisi/smorzamento/'+str(int_configurazione)+'/plot_oscillazione_'+str(int_configurazione)+'.png', dpi=1200)
    plt.close()

    plt.scatter(osc_time, osc_pos_s, marker='.', s=1, c=plot_color)
    plt.grid()
    plt.xlabel('t [s]')
    plt.ylabel('h [m]')
    plt.savefig('analisi/smorzamento/'+str(int_configurazione)+'/plot_oscillazione_'+str(int_configurazione)+'_s.png', dpi=1200)
    plt.close()
    
    # analysis of oscillating positions
    pos_hist, pos_bins = np.histogram(osc_pos, n_bins)
    
    def position(i):
        return ((pos_bins[i]+pos_bins[i+1])/2)
    
    max_hist = local_max(np.array(range(0, len(pos_hist))), pos_hist, 0)
    
    with open('analisi/smorzamento/'+str(int_configurazione)+'/output_oscillazioni_'+str(int_configurazione)+'_hist.log', 'w') as textfile:
        print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)
        
        for i in range(len(pos_hist)):           
            print(round(pos_bins[i],6), "-", round(pos_bins[i+1],6), "\tx:", round(position(i),6), "\tn:", pos_hist[i], file=textfile)
        
        print("\n\nThe local maxima of the histogram are at:", file=textfile)
        
        for key in max_hist:
            print(round(pos_bins[max_hist[key][0]],6), "-", round(pos_bins[max_hist[key][0]+1],6), "\tx:", round(position(max_hist[key][0]),6), "\tn:", pos_hist[max_hist[key][0]], file=textfile)
    
    # plot histogran of oscillating positions
    fig, axs= plt.subplots(1, 1, tight_layout = True)
    N, bins, patches = axs.hist(osc_pos, bins = n_bins, color=plot_color)
    plt.grid()
    plt.xlabel('h [m]')
    plt.ylabel('counts')
    plt.savefig('analisi/smorzamento/'+str(int_configurazione)+'/plot_oscillazione_'+str(int_configurazione)+'_hist.png', dpi=1200)
    plt.close()
    
    # evaluation of the error of oscillating positions
    i_max_1 = max_hist[key_max_1][0]
    i_max_2 = max_hist[key_max_2][0]
    max_1 = position(i_max_1)
    max_2 = position(i_max_2)
    n_max_1 = pos_hist[i_max_1]
    n_max_2 = pos_hist[i_max_2]
    max_mean = (max_1 * n_max_1 + max_2 * n_max_2) / (n_max_1 + n_max_2)
    n_max_mean = (n_max_1 + n_max_2) / 2
    
    i_half_1 = find_last_index_less_than(pos_hist[0:i_max_1+1], n_max_mean/2)
    i_half_2 = find_first_index_less_than(pos_hist[i_max_2:len(pos_hist)], n_max_mean/2) + i_max_2
    half_1 = position(i_half_1)
    half_2 = position(i_half_2)
    
    hwhm = (position(i_half_2) - position(i_half_1)) / 2
    hwfm = (max_2 - max_1) / 2
    
    error_p = ((half_2 - half_1) - (max_2 - max_1)) / 2
    print("\nSpatial error:", round(error_p,6), "(previous)")
    print("Centroide:", round(max_mean,6))

    error_d = np.pi * amplitude / (frequency * period)
    print("\nHeight error:", round(error_d, 6), "(dynamic)")
    print("Amplitude:", round(amplitude, 6))

    error = np.sqrt(dev_eq_pos**2 + error_d**2)
    print("\nHeight error:", round(error, 6))

    # linear regression
    print("\nLinear regressions:")

#################################################################################################################################################################
#
#    reg_dic_1 = {}
#    reg_dic_2 = {}
#
#    def func_1(int):
#
#        x_func_1 = []
#        y_func_1 = []
#        y_err_func_1 = []
#
#        i = 1
#        while i <= len(osc_max_dic.keys()):
#            x_func_1.append((i-1))
#            y_func_1.append(np.log(osc_max_dic[1][1] / osc_max_dic[i][1]))
#            y_err_func_1.append(error * np.sqrt((1.0 / osc_max_dic[i][1])**2 + (1.0 / osc_max_dic[1][1])**2))
#            i += int
#
#        return chi_q(x_func_1, y_func_1, y_err_func_1, linear_regression(x_func_1, y_func_1, y_err_func_1))
#
#    def func_2(int):
#
#        x_func_2 = []
#        y_func_2 = []
#        y_err_func_2 = []
#
#        i = 1
#        while i <= len(osc_max_dic.keys()):
#            x_func_2.append((i-1))
#            y_func_2.append(np.log(osc_max_dic[1][1] / osc_max_dic[i][1]))
#            y_err_func_2.append(error * np.sqrt((1.0 / osc_max_dic[i][1])**2 + (1.0 / osc_max_dic[1][1])**2))
#            i += int
#
#        reg = linear_regression(x_func_2, y_func_2, y_err_func_2)
#        return reg['inter']
#
#    for i in range(1,1+5):
#
#        reg_dic_1[i] = func_1(i)
#        reg_dic_2[i] = func_2(i)
#
#
#    n_1 = min_dic(reg_dic_1)
#    n_2 = min_dic(reg_dic_2)
#
#    print("n_1:", n_1)
#    print("n_2:", n_2)
#    print("\n")
#
#################################################################################################################################################################

    # c2 = 0
    x_2 = []
    y_2 = []
    y_err_2 = []

    x_p_2 = []
    t_p_2 = []
    
    i = 1
    count = 0
    while (i <= len(osc_max_dic.keys()) and count < 10):
        x_2.append((i-1))
        y_2.append(np.log(osc_max_dic[1][1] / osc_max_dic[i][1]))
        y_err_2.append(error * np.sqrt((1.0 / osc_max_dic[i][1])**2 + (1.0 / osc_max_dic[1][1])**2))

        x_p_2.append(osc_max_dic[i][1])
        t_p_2.append(osc_max_dic[i][0])
        i += n_2
        count += 1
        
    with open('analisi/smorzamento/'+str(int_configurazione)+'/output_regressione_c2_'+str(int_configurazione)+'.log', 'w') as textfile:
        print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)

        for i in range(len(x_2)):
            print("x:", x_2[i], "\ty:", y_2[i], "\tsigma:", y_err_2[i], file=textfile)

        print("\n\n\nMeasures used:\n", file=textfile)

        for i in range(len(x_p_2)):
            print("x:", x_p_2[i], "\tt:", t_p_2[i], file=textfile)
    
    regression_c2 = linear_regression(x_2, y_2, y_err_2)
    
    chi_c2 = chi_q(x_2, y_2, y_err_2, regression_c2)
    nu2 = regression_c2['nu']
    p2 = chi2.cdf(chi_c2, nu2)
    print("χ² (c2=0):", chi_c2)
    print("ν  (c2=0):", nu2)
    print("P  (c2=0):", round((1 - p2)*100,3))
    
    # define domain of linear regression
    n_space_2 = np.linspace(0, max_arr(x_2)+3, 1000)
    
    # define regression line
    def reg_lin_c2(x_2):
        return (regression_c2['slope'] * x_2 + regression_c2['inter'])
    
    # plot data and line
    plt.plot(n_space_2, reg_lin_c2(n_space_2), color='blue', lw = 0.5)
    plt.errorbar(x_2, y_2, y_err_2, fmt='.', ecolor='red')
    plt.grid()
    plt.xlabel("η")
    plt.ylabel('ξ')
    plt.savefig('analisi/smorzamento/'+str(int_configurazione)+'/plot_regressione_c2_'+str(int_configurazione)+'.png', dpi=1200)
    plt.close()
    
    # c1 = 0    
    x_1 = []
    y_1 = []
    y_err_1 = []

    x_p_1 = []
    t_p_1 = []
    
    i = 1
    count = 0
    while (i <= len(osc_max_dic.keys()) and count < 10):
        x_1.append(osc_max_dic[i][0])
        y_1.append(1 / osc_max_dic[i][1])
        y_err_1.append(error / (osc_max_dic[i][1])**2)

        x_p_1.append(osc_max_dic[i][1])
        t_p_1.append(osc_max_dic[i][0])
        i += n_1
        count += 1
        
    with open('analisi/smorzamento/'+str(int_configurazione)+'/output_regressione_c1_'+str(int_configurazione)+'.log', 'w') as textfile:
        print("Configuration "+str(int_configurazione)+": ", "\n", file=textfile)

        for i in range(len(x_1)):
            print("x:", x_1[i], "\ty:", y_1[i], "\tsigma:", y_err_1[i], file=textfile)
        
        print("\n\n\nMeasures used:\n", file=textfile)

        for i in range(len(x_p_1)):
            print("x:", x_p_1[i], "\tt:", t_p_1[i], file=textfile)
    
    regression_c1 = linear_regression(x_1, y_1, y_err_1)
    
    chi_c1 = chi_q(x_1, y_1, y_err_1, regression_c1)
    nu1 = regression_c1['nu']
    p1 = chi2.cdf(chi_c1, nu1)
    print("\nχ² (c1=0):", chi_c1)
    print("ν  (c1=0):", nu1)
    print("P  (c1=0):", round((1 - p1)*100,3))
    
    # define domain of linear regression
    n_space_1 = np.linspace(0, max_arr(x_1)+3*period, 1000)
    
    # define regression line
    def reg_lin_c1(x):
        return (regression_c1['slope'] * x + regression_c1['inter'])
    
    # plot data and line
    plt.plot(n_space_1, reg_lin_c1(n_space_1), color='blue', lw = 0.5)
    plt.errorbar(x_1, y_1, y_err_1, fmt='.', ecolor='red')
    plt.grid()
    plt.xlabel("η [s]")
    plt.ylabel('ξ [1/m]')
    plt.savefig('analisi/smorzamento/'+str(int_configurazione)+'/plot_regressione_c1_'+str(int_configurazione)+'.png', dpi=1200)
    plt.close()
    
    with open('analisi/smorzamento/'+str(int_configurazione)+'/output_configurazione_'+str(int_configurazione)+'_.log', 'w') as textfile:
        
        print("\nMaxima of distribution:", file=textfile)
        print(round(pos_bins[i_max_1],6), "-", round(pos_bins[i_max_1+1],6), "\tx:", round(position(i_max_1),6), "\tn:", pos_hist[i_max_1], file=textfile)
        print(round(pos_bins[i_max_2],6), "-", round(pos_bins[i_max_2+1],6), "\tx:", round(position(i_max_2),6), "\tn:", pos_hist[i_max_2], file=textfile)

        print("\nHalf maximum:", round(n_max_mean/2,6), file=textfile)

        print("\nFWHM limits:", file=textfile)
        print(round(pos_bins[i_half_1],6), "-", round(pos_bins[i_half_1+1],6), "\tx:", round(position(i_half_1),6), "\tn:", pos_hist[i_half_1], file=textfile)
        print(round(pos_bins[i_half_2],6), "-", round(pos_bins[i_half_2+1],6), "\tx:", round(position(i_half_2),6), "\tn:", pos_hist[i_half_2], file=textfile)

        print("\nHWHM for configuration "+str(int_configurazione)+":", round(hwhm,6), file=textfile)
        print("HWFM for configuration "+str(int_configurazione)+":", round(hwfm,6), file=textfile)
        
        print("\nLinear regression", file=textfile)
        
        print("\n(c2 = 0)", file=textfile)
        for key in regression_c2:
            print(key, ":\t", regression_c2[key], file=textfile)
            
        print("\n(c1 = 0)", file=textfile)
        for key in regression_c1:
            print(key, ":\t", regression_c1[key], file=textfile)
    
    gamma = regression_c2["slope"] / period
    err_gamma = np.sqrt((regression_c2["s_slp"])**2 + (gamma * err_period)**2) / period
    
    omega = 2 * np.pi / period
    err_omega = 2 * np.pi * err_period / (period**2)

    nu = 1 / period
    err_nu = err_period / (period**2)

    with open('analisi/smorzamento/'+str(int_configurazione)+'/output_'+str(int_configurazione)+'.log', 'w') as textfile:

        print("\nPeriod:", round(period, 6), file=textfile)
        print("Error on period:", round(err_period, 6), file=textfile)

        print("\nGamma:", round(gamma,6), file=textfile)
        print("Error on gamma:", round(err_gamma,6), file=textfile)

        print("\nOmega:", round(omega,6), file=textfile)
        print("Error on omega:", round(err_omega,6), file=textfile)

        print("\nResonance:", round(nu,6), file=textfile)
        print("Error on resonance:", round(err_nu,6), file=textfile)


analysis('dati/smorzamento/equilibrio_1.csv', 'dati/smorzamento/oscillazione_1.csv', 1, 250, 'blue', 'red', 50, 5, 6, 4, 4)
analysis('dati/smorzamento/equilibrio_2.csv', 'dati/smorzamento/oscillazione_2.csv', 2, 100, 'red', 'blue', 50, 2, 5, 5, 5)
analysis('dati/smorzamento/equilibrio_3.csv', 'dati/smorzamento/oscillazione_3.csv', 3, 200, 'green', 'red', 50, 3, 4, 12, 12)