%matplotlib inline

import os
import numpy as np
from scipy import io as sio
import matplotlib.pylab as plt
from collections import OrderedDict

from metrics import *

dataset = ["50Salads", "JIGSAWS", "MERL", "GTEA"][0]
base_dir = os.path.expanduser("~/TCN_release/predictions/{}/".format(dataset))
dirs = np.sort(os.listdir(base_dir))

# Manually set the background class
bg_class = 0
# If using Bharat's predictions background class is 5. Else 0.
if dataset is "MERL":
	bg_class = 5
elif "JIGSAWS" in base_dir:
	bg_class = None


P_all = {}
Y_all = {}
f1_scores = OrderedDict()
acc_scores = OrderedDict()
edit_scores = OrderedDict()

# Go through each predictions folder and output stats
for d in dirs:
	files = np.sort(os.listdir(base_dir+d))

	# Load each split in the given folder
	S, P, Y = [], [], []
	for f in files:
		data = sio.loadmat("/".join([base_dir+d,f]))
		S +=[s for s in data['S'][0]]
		P +=[np.squeeze(p) for p in data['P'][0]]
		Y +=[np.squeeze(y) for y in data['Y'][0]]

	# MERL stats are computed as if the dataset was one long sequence. 
	if dataset == "MERL":
		P = [np.hstack(P)]
		Y = [np.hstack(Y)]
		S = [np.vstack(S)]

	P_all[d] = P
	Y_all[d] = Y
	n_classes = np.hstack(Y).max()+1
	print(d)

	# Compute metrics
	CM = ComputeMetrics(overlap=.25, n_classes=n_classes, bg_class=bg_class)
	CM.add_predictions(1, P, Y)
	CM.print_scores()

	overlaps = [overlap_f1(P, Y, n_classes, bg_class, overlap=o) for o in [.1, .25, .5]]
	print("F1@{1,25,50}", ",".join(["{:.03}".format(o) for o in overlaps]))

	f1_scores[d] = overlaps[1]
	acc_scores[d] = list(CM.scores["accuracy"].values())[0]
	edit_scores[d] = list(CM.scores["edit_score"].values())[0]


	# Mean Average Precision (midpoint and @k)
	gt_file = []
	gt_labels = []
	gt_intervals = []
	for i,y in enumerate(Y):
		gt_labels += [utils.segment_labels(y)]
		gt_intervals += [np.array(utils.segment_intervals(y))]
		gt_file += [[i]*len(gt_labels[-1])]
	gt_file = np.hstack(gt_file)
	gt_intervals = np.vstack(gt_intervals)
	gt_labels = np.hstack(gt_labels)

	det_file = []
	det_labels = []
	det_intervals = []
	det_scores = []
	for i,y in enumerate(P):
		det_labels += [utils.segment_labels(y)]
		det_intervals += [np.array(utils.segment_intervals(y))]
		det_file += [[i]*len(det_labels[-1])]
		
		det_scores += [S[i][inter[0]:inter[1]][:,label].max() for inter,label in zip(det_intervals[-1], det_labels[-1])]

	det_file = np.hstack(det_file)
	det_intervals = np.vstack(det_intervals)
	det_labels = np.hstack(det_labels)
	det_scores = np.hstack(det_scores)

	pr, ap, mAP_1 = IoU_mAP(gt_file, det_file, gt_labels, det_labels, 
							gt_intervals, det_intervals, det_scores, .1, bg_class=bg_class)
	pr, ap, mAP_5 = IoU_mAP(gt_file, det_file, gt_labels, det_labels, 
							gt_intervals, det_intervals, det_scores, .5, bg_class=bg_class)
	pr, ap, mAP_mid = midpoint_mAP(gt_file, det_file, gt_labels, det_labels, 
							gt_intervals, det_intervals, det_scores, bg_class=bg_class)
	print("mAP(.1, .5, mid): {:.04}, {:.04}, {:.04}".format(mAP_1*100, mAP_5*100, mAP_mid*100))

	print()


# ----------------- Various visualizations of predictions ------------
# Performance per trial
if 0:
	from pylab import cm
	accs = np.array([metrics.accuracy(p,y) for p,y in zip(P,Y)])
	edits = np.array([metrics.edit_score(p,y) for p,y in zip(P,Y)])
	f1s = np.array([overlap_f1(p, y, n_classes, bg_class, overlap=.25) for p,y in zip(P,Y)])
	pearson = np.corrcoef(edits, accs)	
	pearson = np.corrcoef(edits, f1s)	
	pearson = np.corrcoef(accs, f1s)
	order = np.argsort(edits)
	# order = np.argsort(accs)
	# order = np.argsort(order)
	n_trials = len(order)
	plt.scatter(np.arange(len(order)), accs[order], color='blue', linewidth=5, label="Acc")
	plt.scatter(np.arange(len(order)), edits[order], color='red', linewidth=5, label="Edit")
	plt.scatter(np.arange(len(order)), f1s[order], color='green', linewidth=5, label="Seg F1")
	# plt.scatter(np.arange(len(order)), accs[order], color=cm.jet(order/n_trials)[:,:3], s=60, label="Acc")
	# plt.scatter(np.arange(len(order)), edits[order], marker='o', facecolors='none', edgecolors=cm.jet(order/n_trials)[:,:3], s=60, linewidth=3, label="Edit")
	plt.xlabel("Trials (sorted by edit)", fontsize=16)
	plt.xticks(fontsize=14) 
	plt.yticks(fontsize=14) 
	dataset = "JIGSAWS" if "JIGSAWS" in base_dir else "50 Salads"
	plt.title("Performance on {} (p={:.02})".format(dataset,pearson[0,1]), fontsize=16)
	plt.legend(loc=4, fontsize=14)
	plt.axis([-1,n_trials+1,min(accs.min(), edits.min())-2, 102])



if 0: # ECCV or ICRA receptive field
	ds = np.array([int(k.split("_")[-1]) for k in f1_scores])
	order = np.argsort(ds)
	ds = ds[order]
	accs = np.array(list(acc_scores.values()))[order]
	edits = np.array(list(edit_scores.values()))[order]
	f1s = np.array(list(f1_scores.values()))[order]

	rate = 30 / 5. if "JIGSAWS" in base_dir else 30 / 10.
	plt.plot(ds/rate, accs, color='blue', linewidth=5, label="Acc")
	plt.plot(ds/rate, edits, color='red', linewidth=5, label="Edit")
	#plt.plot(ds, f1s, color='green', linewidth=5, label="Seg F1")
	plt.xlabel("Filter length (seconds)", fontsize=16)
	plt.xticks(fontsize=14) 
	plt.yticks(fontsize=14) 
	dataset = "JIGSAWS" if "JIGSAWS" in base_dir else "50 Salads"
	plt.title("Performance on {}".format(dataset), fontsize=16)
	plt.legend(loc=4, fontsize=14)

if 0: # ED-TCN receptive field
	ds = np.array([int(k.split("conv")[-1]) for k in f1_scores])
	Ls = np.array([int(k.split("relu_")[-1].split("_")[0]) for k in f1_scores])
	f1s = np.array(list(f1_scores.values()))

	for l in range(1, 5):
		order = np.argsort(ds)
		ds = ds[order]
		Ls = Ls[order]
		f1s = f1s[order]
		plt.plot(ds[Ls==l], f1s[Ls==l], '-', linewidth=5, label="L={}".format(l))
	# plt.xlabel("Receptive Field Duration", fontsize=18)
	plt.xlabel("Filter Duration (seconds)", fontsize=18)
	plt.ylabel("F1@25 Score", fontsize=18)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.title("ED-TCN", fontsize=20, fontstyle="normal")
	plt.legend()

if 0: # Dilated TCN receptive field
	Bs = np.array([int(k[-1]) for k in f1_scores])
	Ls = np.array([int(k.split("_")[5][1]) for k in f1_scores])
	f1s = np.array(list(f1_scores.values()))

	for l in range(1, 5):
		order = np.argsort(Ls)
		Bs = Bs[order]
		Ls = Ls[order]
		f1s = f1s[order]
		plt.plot(Ls[Bs==l], f1s[Bs==l], '-', linewidth=5, label="B={}".format(l))
	plt.xlabel("Number of layers ", fontsize=18)
	plt.ylabel("F1@25 Score", fontsize=18)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.title("Dilated TCN", fontsize=20, fontstyle="normal")
	plt.legend()


if 0:
	# bg_class = 0
	# n_classes = 18
	# # base_dir = '/home/colin/data/50Salads/predictions/'
	# base_dir = '/home/colin/data/50Salads/predictions/accel/'


	# dirs = os.listdir(base_dir)
	# dirs = np.sort([f for f in dirs if "TCN_mid" in f])
	# dirs = np.sort([f for f in dirs if "eval" in f])


	idx = 1
	# plt.figure(figsize=(12,8))
	# y0 = Y_all["TCN_mid_SVM__conv18"][idx]
	# plt.subplot(5,1,1)
	# plt.xticks([])
	# imshow_(y0)
	# for i,t in enumerate(["TCN_mid_SVM__conv18", "TCN_mid_WaveNet_wavenet_conv18", "TCN_mid_ECCV__conv36", "TCN_mid_TCN_norm_relu_conv18"]):
		# p = P_all[t][idx]
		# plt.subplot(5,1,i+2)
		# imshow_(p)
		# if i != 4:
		# 	plt.xticks([])
		# print(np.mean(p==y0))
		# acc = np.mean(p==y0)
		# plt.ylabel("{:.04}".format(acc*100), fontsize=18)

	for idx in range(1, 39, 3):
		# plt.figure(figsize=(12,8))
		plt.figure(figsize=(12,6))
		y0 = list(Y_all.values())[0][idx]
		plt.subplot(len(P_all)+1,1,1)
		imshow_(y0, vmin=0, vmax=n_classes)
		plt.ylabel("Truth", fontsize=16)
		plt.xticks([])
		# for i,t in enumerate(list(P_all.keys())):	
		for i,t in enumerate([list(P_all.keys())[-1]]):	
			p = P_all[t][idx]
			plt.subplot(len(P_all)+1,1,i+2)
			imshow_(p, vmin=0, vmax=n_classes)

			# if i != 4:
			# 	plt.xticks([])

			# plt.xticks([])
			# print(np.mean(p==y0))
			acc = np.mean(p==y0)
			plt.ylabel("{:.04}".format(acc*100), fontsize=16)
			if "ICRA" in t:
				plt.ylabel("ICRA", fontsize=16)
			else:
				plt.ylabel("ED\nTCN", fontsize=16)




	plt.figure(figsize=(12,12))
	for idx in range(10):
		y0 = list(Y_all.values())[0][idx*5]
		p = P_all[list(P_all.keys())[0]][idx*5]
		plt.subplot(20,1,idx+1)
		imshow_(np.vstack([y0, p]), vmin=0, vmax=n_classes)

		# if i != 4:
		# 	plt.xticks([])

		plt.yticks([])
		if idx!=10-1:
			plt.xticks([])
		# print(np.mean(p==y0))
		acc = int(np.mean(p==y0)*100)
		acc = int(metrics.edit_score(p, y0, bg_class=bg_class))
		
		plt.ylabel("{}".format(acc), fontsize=16)



	scores = np.sort([metrics.edit_score(p, y0, bg_class=bg_class) for p,y0 in zip(list(P_all.values())[0], list(Y_all.values())[0])])
	plt.scatter(range(len(scores)), scores, color="b", linewidth=6, label="Edit")
	plt.xlabel("Trial number (sorted)", fontsize=16)
	#plt.ylabel("Edit Score", fontsize=16)
	plt.yticks(fontsize=14)

	scores = np.sort([metrics.accuracy(p, y0, bg_class=bg_class) for p,y0 in zip(list(P_all.values())[0], list(Y_all.values())[0])])
	plt.scatter(range(len(scores)), scores, color="g", linewidth=6, label="Acc")
	plt.xlabel("Trial number (sorted)", fontsize=16)
	#$plt.ylabel("Accuracy Score", fontsize=16)
	plt.yticks(fontsize=16)


	scores = np.array([metrics.edit_score(p, y0, bg_class=bg_class) for p,y0 in zip(list(P_all.values())[0], list(Y_all.values())[0])])
	order = np.argsort(scores)
	scores = scores[order]
	plt.scatter(range(len(scores)), scores, color="r", linewidth=6, label="Edit")

	scores = np.array([metrics.accuracy(p, y0, bg_class=bg_class) for p,y0 in zip(list(P_all.values())[0], list(Y_all.values())[0])])
	scores = scores[order]
	plt.scatter(range(len(scores)), scores, color="b", linewidth=6, label="Acc")
	plt.xlabel("Trial number (sorted)", fontsize=18)
	#$plt.ylabel("Accuracy Score", fontsize=16)
	plt.yticks(fontsize=18)
	plt.legend()



# MERL predictions
if 0:
	t_start = 840
	t_stop = t_start+155

	y0 = list(Y_all.values())[0][0]
	y0 = y0.copy() + 1
	y0[y0==6]=0
	y0 = y0[t_start:t_stop]
	plt.figure(figsize=(12,8))
	plt.subplot(len(P_all)+1,1,1)
	plt.xticks([])
	imshow_(y0, vmin=0, vmax=n_classes-1)
	plt.ylabel("Truth", fontsize=18)
	# imshow_(y0, vmin=0, vmax=n_classes)
	
	#for i,t in enumerate(list(P_all.keys())):
	for i,t in enumerate(["MERL_sparse", "MERL_dense", "wavenet_B3_convL1", "TCN_norm_relu_2_conv3"]):
		# if "MERL" in t:
		# 	p = np.array(P_all[t])[0]
		# else:
		p = np.array(P_all[t])[0].copy()
		if "sparse" not in t:
			p += 1
			p[p==6]=0
			#p_ = p.copy()
			#p_[p==4] = 5
			#p_[p==5] = 4
			#p = p_
		plt.subplot(len(P_all)+1,1,i+2)
		if "sparse" in t:
			p = np.array(Y_all[t])[0].copy()
			p = p[t_start*5.98299:t_stop*5.98299]
			#p = p[t_start*5.992:t_stop*5.992]
			#p_tmp = p[t_start:t_stop*6.2956]
			imshow_(p, vmin=0, vmax=n_classes-1)
		else:
			p = p[t_start:t_stop]
			imshow_(p, vmin=0, vmax=n_classes-1)
		# imshow_(p, vmin=0, vmax=n_classes)
		#print(t, p.shape)

		if i != 4:
			plt.xticks([])
		# print(np.mean(p==y0))
		plt.ylabel("{:.04}".format(acc*100), fontsize=18)
		if "TCN" in t:
			plt.ylabel("ED-TCN", fontsize=18)
		elif "wavenet" in t:
			plt.ylabel("D-TCN", fontsize=18)
		elif "dense" in t:
			plt.ylabel("Dense", fontsize=18)
		elif "sparse" in t:
			plt.ylabel("Sparse", fontsize=18)						
		else:
			plt.ylabel(t, fontsize=18)

		idxs = np.linspace(0, y0.shape[0], p.shape[0], endpoint=False).astype(np.int)
		acc = np.mean(p==y0[idxs])*100
		print(t, "Acc {:.03}, {:.03}".format(acc, metrics.overlap_f1(p, y0, n_classes, bg_class, overlap=.25)))



# 50 Salads predictions
if 0:
	idx = 40
	if 1:
	#for idx in range(50):
		y0 = list(Y_all.values())[0][idx]
		plt.figure(idx, figsize=(12,8))
		plt.subplot(len(P_all)+1,1,1)
		plt.xticks([])
		imshow_(y0, vmin=0, vmax=n_classes-1)
		plt.ylabel("Truth", fontsize=18)
		# imshow_(y0, vmin=0, vmax=n_classes)
		
		#for i,t in enumerate(list(P_all.keys())):
		for i,t in enumerate(["TCN_mid_SVM__conv18", "TCN_mid_ECCV__conv36", "TCN_mid_WaveNet_wavenet_B2_L5_B2", "TCN_mid_TCN_norm_relu_2_conv15"]):
			p = np.array(P_all[t])[idx]
			yy = np.array(Y_all[t])[idx]
			
			plt.subplot(len(P_all)+1,1,i+2)
			imshow_(p, vmin=0, vmax=n_classes-1)

			if i != 4:
				plt.xticks([])
			# print(np.mean(p==y0))
			plt.ylabel("{:.04}".format(acc*100), fontsize=18)
			if "norm_relu_" in t:
				plt.ylabel("ED-TCN", fontsize=18)
			elif "wavenet" in t:
				plt.ylabel("D-TCN", fontsize=18)
			elif "ECCV" in t:
				plt.ylabel("ST-CNN", fontsize=18)
			elif "SVM" in t:
				plt.ylabel("SVM", fontsize=18)						
			else:
				plt.ylabel(t, fontsize=18)
			#plt.title(idx)

			acc = np.mean(p==yy)*100
			print(idx, "Acc {:.03}, {:.03}".format(acc, metrics.overlap_f1(p, yy, n_classes, bg_class, overlap=.25)), t)

		print()

