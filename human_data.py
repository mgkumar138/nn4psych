
'''Nassar et al. 2021 - Figure 6
#replicate Nassar2021 fig6a with RNN's behav model fits
#get human model fits for area between curves
'''
#%% setup data

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


#matlab structure is condition.[patient number, 115, 2]
sz_pat = 102 #first 102 / 134 in the model data are patients
#115 is trial num? use all for mean / std
#2 - nassar uses 2nd index always but idk what it is

mod_data = sio.loadmat('data/nassar2021/slidingWindowFits_model_23-Nov-2021.mat')
sub_data = sio.loadmat('data/nassar2021/slidingWindowFits_subjects_23-Nov-2021.mat')

mod_data = np.asarray(mod_data['binRegData'])
sub_data = np.asarray(sub_data['binRegData'])

#so raw python array structure is-
#data[0][0][CP/OB][pat / control ][:][1]
#TO-DO clean up this structure.
cp = 0
ob = 1

#subj data
mean_sub_cp_pat = np.mean(sub_data[0][0][cp][:sz_pat][:][:], axis=0) 
mean_sub_odd_pat = np.mean(sub_data[0][0][ob][:sz_pat][:][:], axis=0)
mean_sub_cp_control = np.mean(sub_data[0][0][cp][sz_pat:][:][:], axis=0)
mean_sub_odd_control = np.mean(sub_data[0][0][ob][sz_pat:][:][:], axis=0)

sd_sub_cp_pat = np.std(sub_data[0][0][cp][:sz_pat][:][:], axis=0)
sd_sub_odd_pat = np.std(sub_data[0][0][ob][:sz_pat][:][:], axis=0)
sd_sub_cp_control = np.std(sub_data[0][0][cp][sz_pat:][:][:], axis=0)
sd_sub_odd_control = np.std(sub_data[0][0][ob][sz_pat:][:][:], axis=0)

#model data
mean_mod_cp_pat = np.mean(mod_data[0][0][cp][:sz_pat][:][:], axis=0)
mean_mod_odd_pat = np.mean(mod_data[0][0][ob][:sz_pat][:][:], axis=0)
mean_mod_cp_control = np.mean(mod_data[0][0][cp][sz_pat:][:][:], axis=0)
mean_mod_odd_control = np.mean(mod_data[0][0][ob][sz_pat:][:][:], axis=0)

sd_mod_cp_pat = np.std(mod_data[0][0][cp][:sz_pat][:][:], axis=0)
sd_mod_odd_pat = np.std(mod_data[0][0][ob][:sz_pat][:][:], axis=0)
sd_mod_cp_control = np.std(mod_data[0][0][cp][sz_pat:][:][:], axis=0)
sd_mod_odd_control = np.std(mod_data[0][0][ob][sz_pat:][:][:], axis=0)


#%% calcs

#calc the differences line
diff_sub_pat = np.mean(np.array([mean_sub_cp_pat[:,1], mean_sub_odd_pat[:,1]]), axis=0)
diff_sub_control = np.mean(np.array([mean_sub_cp_control[:,1], mean_sub_odd_control[:,1]]), axis=0)
diff_mod_pat = np.mean(np.array([mean_mod_cp_pat[:,1], mean_mod_odd_pat[:,1]]), axis=0)
diff_mod_control = np.mean(np.array([mean_mod_cp_control[:,1], mean_mod_odd_control[:,1]]), axis=0)

#difference score(?) between the two lines
# Calculate area between CP and OB learning rate curves using numerical integration
area_sub_pat = np.trapz(np.abs(mean_sub_cp_pat[:,1] - mean_sub_odd_pat[:,1]), dx=1)
area_sub_control = np.trapz(np.abs(mean_sub_cp_control[:,1] - mean_sub_odd_control[:,1]), dx=1)
area_mod_pat = np.trapz(np.abs(mean_mod_cp_pat[:,1] - mean_mod_odd_pat[:,1]), dx=1)
area_mod_control = np.trapz(np.abs(mean_mod_cp_control[:,1] - mean_mod_odd_control[:,1]), dx=1)

#standard error of rad ( cp^2 + ob^2)
sem_sub_pat = np.sqrt((sd_sub_cp_pat[:,1]**2 + sd_sub_odd_pat[:,1]**2) / sz_pat)
sem_sub_control = np.sqrt((sd_sub_cp_control[:,1]**2 + sd_sub_odd_control[:,1]**2) / (len(sub_data[0][0][cp]) - sz_pat))

print("Subject Patients Total Difference:", area_sub_pat)
print("Subject Controls Total Difference:", area_sub_control)
print("Model Patients Total Difference:", area_mod_pat)
print("Model Controls Total Difference:", area_mod_control)

print("Subject Patients SEM:", sem_sub_pat)
print("Subject Controls SEM:", sem_sub_control)

#%%plots

#TO-DO: add error bars (right now, only controls have error bars representative of the paper figure, so must have been some cleaning on SZ group)
def plot_fig6a_no_error(diff_score):
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1)
    plt.plot(x_axis, mean_sub_cp_pat[:,1], label='sub CP pat', color='orange')
    plt.plot(x_axis, mean_sub_odd_pat[:,1], label='sub OB pat', color='brown')
    plt.plot(x_axis, mean_sub_cp_control[:,1], label='sub CP control', color='blue')
    plt.plot(x_axis, mean_sub_odd_control[:,1], label='sub OB control', color='green')
    
    
    if diff_score:
        plt.plot(x_axis, diff_sub_pat, label='sub CP-OB pat', color='red')
        plt.plot(x_axis, diff_sub_control, label='sub CP-OB control', color='purple')
    plt.xlabel('Relative Error')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.savefig('plots/nassarfig6a.png')
    plt.show()


def plot_fig6b(diff_score):
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1)
    plt.plot(x_axis, mean_mod_cp_pat[:,1], label='mod CP pat', color='orange')
    plt.plot(x_axis, mean_mod_odd_pat[:,1], label='mod OB pat', color='brown')
    plt.plot(x_axis, mean_mod_cp_control[:,1], label='mod CP control', color='blue')
    plt.plot(x_axis, mean_mod_odd_control[:,1], label='mod OB control', color='green')
    if diff_score:
        plt.plot(x_axis, diff_mod_pat, label='mod CP-OB pat', color='red')
        plt.plot(x_axis, diff_mod_control, label='mod CP-OB control', color='purple')
    plt.xlabel('Relative Error')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.savefig('plots/nassarfig6b.png')
    plt.show()

def plot_bar_graph_area_between_curves():
    plt.figure(figsize=(10, 6))
    plt.bar(['Sub pat', 'Sub con', 'Mod pat', 'Mod con'], [area_sub_pat, area_sub_control, area_mod_pat, area_mod_control], color=['orange', 'brown', 'blue', 'green'])
    plt.ylabel('Area between curves')
    plt.title('Area between CP and OB learning rate curves')
    plt.savefig('plots/area_between_curves.png')
    plt.show()



#decide cutoff on x-axis boundaries ("relativity" in relative error)
x_axis = [i for i in range(115)]




plot_fig6a_no_error(diff_score = True)
plot_fig6b(diff_score = True)
plot_bar_graph_area_between_curves()


