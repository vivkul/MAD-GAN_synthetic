#!/usr/bin/env python
import numpy as np
from scipy.stats import chisquare
from scipy.stats import entropy
dirs = ['out/BEGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001/out/', 'out/DCGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001/out/', 'out/GoGAN_specs_gen1_0_toydisc_toydisc/WClip=0.1_LR=0.0001/out/', 'out/MADGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001_NGEN=4/out/', 'out/MODEGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001/out/', 'out/TRIVIAL_specs_gen1_0_toydisc_toydisc/LR=0.0001_NGEN=4/out/', 'out/WGAN_specs_gen1_0_toydisc_toydisc/WClip=0.1_LR=0.0001/out/', 'out/UNROLLEDGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001/out/']

filenames = ['BEGAN.out', 'DCGAN.out', 'GGAN_2nd.out', 'MADGAN.out', 'MODEGAN.out', 'TRIVIAL.out', 'WGAN.out', 'UNROLLEDGAN.out']

index = [190, 190, 190, 190, 190, 190, 190, 120]

data_pre = 'data_samples_0'
gen_pre = 'gen_samples_0'

# data_samples = 'out/MADGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001_NGEN=4/out/data_samples_0190_MADGAN.out'
# gen_samples = 'out/MADGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001_NGEN=4/out/gen_samples_0190_MADGAN.out'

binsize = 0.5

f = open('chisquare_res.out', 'a+')
f.write('\nbinsize = ' + str(binsize) +'\n')
f.close()

f = open('kl_res.out', 'a+')
f.write('\nbinsize = ' + str(binsize) +'\n')
f.close()

for itera in range(len(dirs)):
	data_samples = dirs[itera] + data_pre + str(index[itera]) + '_' + filenames[itera]
	gen_samples = dirs[itera] + gen_pre + str(index[itera]) + '_' + filenames[itera]

	data = np.loadtxt(data_samples, delimiter=',')
	gen = np.loadtxt(gen_samples, delimiter=',')

	data_hist = np.histogram(data, bins=np.arange(-10,130,binsize))
	gen_hist = np.histogram(gen, bins=np.arange(-10,130,binsize))

	data_hist_non_zero_freq = []
	gen_hist_non_zero_freq = []

	for i in range(len(data_hist[0])):
		if data_hist[0][i] != 0:
			data_hist_non_zero_freq.append(data_hist[0][i])
			gen_hist_non_zero_freq.append(gen_hist[0][i])

	chisquare_res = chisquare(np.array(gen_hist_non_zero_freq), np.array(data_hist_non_zero_freq))
	any_freq_zero = len(data_hist[0]) - len(data_hist_non_zero_freq)

	f = open('chisquare_res.out', 'a+')
	f.write(filenames[itera] + ': ' + str(chisquare_res) + ' ' + str(any_freq_zero) +'\n')
	f.close()

	data_hist = np.histogram(data, bins=np.arange(-10,130,binsize), density=True)
	gen_hist = np.histogram(gen, bins=np.arange(-10,130,binsize), density=True)

        data_hist_non_zero_freq = []
	gen_hist_non_zero_freq = []

	for i in range(len(data_hist[0])):
		if data_hist[0][i] != 0:
			data_hist_non_zero_freq.append(data_hist[0][i])
			gen_hist_non_zero_freq.append(gen_hist[0][i])

	entropy_res=entropy(gen_hist_non_zero_freq, data_hist_non_zero_freq)

	f = open('kl_res.out', 'a+')
	f.write(filenames[itera] + ': ' + str(entropy_res) + ' ' + str(any_freq_zero) +'\n')
	f.close()


