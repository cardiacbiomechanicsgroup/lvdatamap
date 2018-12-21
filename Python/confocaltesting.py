from cardiachelpers import confocalhelper
import numpy as np
import matplotlib.pyplot as mplt
from mpl_toolkits.mplot3d import Axes3D

test_tif = 'H:\ConfocalImages\DHETrialExport-062218\Slice3Contour.tif'

test_list = confocalhelper.openModelImage(test_tif)

if not isinstance(test_list, list):
	test_list = [test_list]

endo_traces = [None]*len(test_list)
epi_traces = [None]*len(test_list)
slice_gap = 10

for im_num, im_slice in enumerate(test_list):
	skeleton_image = confocalhelper.contourMaskImage(im_slice)
	endo_path, epi_path, labelled_arr = confocalhelper.splitImageObjects(skeleton_image)
	endo_traces[im_num] = confocalhelper.orderPathTrace(endo_path)
	epi_traces[im_num] = confocalhelper.orderPathTrace(epi_path)

slice_gaps = [i*slice_gap for i in range(len(endo_traces))]

endo_x, endo_y = confocalhelper.formatContourForModel(endo_traces)
epi_x, epi_y = confocalhelper.formatContourForModel(epi_traces)

avg_endo_x = np.nanmean(endo_x)
avg_endo_y = np.nanmean(endo_y)

for i in range(len(endo_traces)):
	endo_traces[i][:, 0] = np.subtract(endo_traces[i][:, 0], avg_endo_x)
	endo_traces[i][:, 1] = np.subtract(endo_traces[i][:, 1], avg_endo_y)
	epi_traces[i][:, 0] = np.subtract(epi_traces[i][:, 0], avg_endo_x)
	epi_traces[i][:, 1] = np.subtract(epi_traces[i][:, 1], avg_endo_y)

endo_x -= avg_endo_x
epi_x -= avg_endo_x
endo_y -= avg_endo_y
epi_y -= avg_endo_y

fig = mplt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(endo_traces)):
	ax.plot(endo_traces[i][:, 0], endo_traces[i][:, 1], [slice_gaps[i]]*endo_traces[i].shape[0])
	ax.plot(epi_traces[i][:, 0], epi_traces[i][:, 1], [slice_gaps[i]]*epi_traces[i].shape[0])