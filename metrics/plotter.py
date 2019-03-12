#!/usr/bin/env python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dirs = ['out/BEGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001/out/', 'out/DCGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001/out/', 'out/GoGAN_specs_gen1_0_toydisc_toydisc/WClip=0.1_LR=0.0001/out/', 'out/MADGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001_NGEN=4/out/', 'out/MODEGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001/out/', 'out/TRIVIAL_specs_gen1_0_toydisc_toydisc/LR=0.0001_NGEN=4/out/', 'out/WGAN_specs_gen1_0_toydisc_toydisc/WClip=0.1_LR=0.0001/out/', 'out/UNROLLEDGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001/out/']

filenames = ['BEGAN', 'DCGAN', 'GGAN_2nd', 'MADGAN', 'MODEGAN', 'TRIVIAL', 'WGAN', 'UNROLLEDGAN']

index = [190, 190, 190, 190, 190, 190, 190, 190]

data_pre = 'data_samples_'
gen_pre = 'gen_samples_'

# data_samples = 'out/MADGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001_NGEN=4/out/data_samples_0190_MADGAN.out'
# gen_samples = 'out/MADGAN_specs_gen1_0_toydisc_toydisc/LR=0.0001_NGEN=4/out/gen_samples_0190_MADGAN.out'

binsize = 0.1

for itera in range(len(dirs)):
        itera += 3
	filename = '0' + str(index[itera]) + '_' + filenames[itera]
	if filenames[itera] == 'MODEGAN':
		filename = 'reg' + filename

	data_samples = dirs[itera] + data_pre + filename + '.out'
	gen_samples = dirs[itera] + gen_pre + filename + '.out'

	data = np.loadtxt(data_samples, delimiter=',')
	gen = np.loadtxt(gen_samples, delimiter=',')

	fig, axs = plt.subplots()

	sns.distplot(gen,ax=axs,bins=np.arange(-10, 130, 0.1),norm_hist=True,kde=False) #sns.distplot(gen,ax=axs,bins=np.arange(-10, 130, 0.1))
	sns.distplot(data,ax=axs,bins=np.arange(-10, 130, 0.1),norm_hist=True,kde=False)
	plt.axis([-10, 130, 0, 0.1])
	plt.savefig('outFig/' + filenames[itera] + '_1D.png')

	break
