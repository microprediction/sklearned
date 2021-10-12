
import numpy as np




def remove_surrogate_outliers(d):
    for thing in ['train','val','test']:
        d_avg_last_five = np.mean(d['x_'+thing][:, 0, -5:], axis=1)
        innovation = d['y_'+thing][:, 0] - d_avg_last_five
        innovation_std = np.std(innovation)
        wild_fraction = np.mean(np.abs(innovation) > 3 * innovation_std)
        n = len(innovation)
        conservative_y = [y[0] if abs(inn) < 3 * innovation_std else avg_last_five for y, avg_last_five, inn in
                              zip(d['y_'+thing], d_avg_last_five, innovation)]
        new_y = np.zeros(shape=(n,1))
        new_y[:,0] = conservative_y
        d['y_'+thing] = new_y
        if True:
            print('Wild fraction='+str(wild_fraction))
    return d



if __name__=='__main__':
    skater_name = 'tsa_aggressive_d0_ensemble'
    baseline = 'thinking_slow_and_fast'
    n_input = 160
    k = 1
    from sklearned.challenging.surrogatechallenge import cached_skater_surrogate_data
    d = cached_skater_surrogate_data(skater_name=skater_name, k=k, n_samples=150, n_warm=290, n_input=n_input)
    for thing in ['test','val','train']:
       print(np.shape(d['y_'+thing]))

    d = remove_surrogate_outliers(d)

    for thing in ['test','val','train']:
       print(np.shape(d['y_'+thing]))
