import argparse
import os

import numpy as np
from PIL import Image

from utils.metrics import iou_stats


# tp_fn = np.zeros(args.num_classes, dtype=np.float64)
# tp_fp = np.zeros(args.num_classes, dtype=np.float64)
# tp = np.zeros(args.num_classes, dtype=np.float64)
#
# for dirpath, dirnames, filenames in os.walk(args.pred_dir):
# 	for filename in filenames:
# 		predname = os.path.join(dirpath, filename)
# 		gtname = predname.replace(args.pred_dir, args.gt_dir)
# 		if args.string_replace != '':
# 			stra, strb = args.string_replace.split(',')
# 			gtname = gtname.replace(stra, strb)
#
# 		pred = np.asarray(
# 			Image.open(predname).convert(mode='L'),
# 			dtype=np.uint8)
#
# 		gt = np.asarray(
# 			Image.open(gtname).convert(mode='P'),
# 			dtype=np.uint8)
#
# 		_tp_fn, _tp_fp, _tp = iou_stats(
# 			pred,
# 			gt,
# 			num_classes=args.num_classes,
# 			background=0)
#
# 		tp_fn += _tp_fn
# 		tp_fp += _tp_fp
# 		tp += _tp
#
# iou = tp / (tp_fn + tp_fp - tp + 1e-12) * 100.0
#
# class_names = ['Background', 'Aero', 'Bike', 'Bird', 'Boat',
# 			   'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow',
# 			   'Table', 'Dog', 'Horse', 'MBike', 'Person',
# 			   'Plant', 'Sheep', 'Sofa', 'Train', 'TV']
#
# for i in range(args.num_classes):
# 	print('class {:10s}: {:02d}, acc: {:4.4f}%'.format(
# 		class_names[i], i, iou[i]))
# mean_iou = iou.sum() / args.num_classes
# print('mean IOU: {:4.4f}%'.format(mean_iou))
#
# mean_pixel_acc = tp.sum() / (tp_fp.sum() + 1e-12)
# print('mean Pixel Acc: {:4.4f}%'.format(mean_pixel_acc))


def calcu_voc_mIou(pred_dir, gt_dir):
	assert os.path.isdir(pred_dir)
	assert os.path.isdir(gt_dir)
	
	print('......')
	
	n_class = 21
	tp_fn = np.zeros(n_class, dtype=np.float64)
	tp_fp = np.zeros(n_class, dtype=np.float64)
	tp = np.zeros(n_class, dtype=np.float64)
	
	for parent, dirs, files in os.walk(pred_dir):
		for file in files:
			pred_img_file = os.path.join(parent, file)
			gt_img_file = pred_img_file.replace(pred_dir, gt_dir)
			# if args.string_replace != '':
			# 	stra, strb = args.string_replace.split(',')
			# 	gtname = gtname.replace(stra, strb)
			pred = np.asarray(
				Image.open(pred_img_file).convert(mode='L'),
				dtype=np.uint8)
			gt = np.asarray(
				Image.open(gt_img_file).convert(mode='P'),
				dtype=np.uint8)
			_tp_fn, _tp_fp, _tp = iou_stats(
				pred,
				gt,
				num_classes=n_class,
				background=0)
			
			tp_fn += _tp_fn
			tp_fp += _tp_fp
			tp += _tp
	
	iou = tp / (tp_fn + tp_fp - tp + 1e-12) * 100.0
	
	class_names = ['Background', 'Aero', 'Bike', 'Bird', 'Boat',
				   'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow',
				   'Table', 'Dog', 'Horse', 'MBike', 'Person',
				   'Plant', 'Sheep', 'Sofa', 'Train', 'TV']
	
	for i in range(n_class):
		print('class {:10s}: {:02d}, acc: {:4.4f}%'.format(class_names[i], i, iou[i]))
	
	mean_iou = iou.sum() / n_class
	print('mean IOU: {:4.4f}%'.format(mean_iou))
	
	mean_pixel_acc = tp.sum() / (tp_fp.sum() + 1e-12)
	print('mean Pixel Acc: {:4.4f}%'.format(mean_pixel_acc))


def calcu_cityscapes_mIou(pred_dir, gt_dir):
	assert os.path.isdir(pred_dir)
	assert os.path.isdir(gt_dir)
	n_class = 19
	tp_fn = np.zeros(n_class, dtype=np.float64)
	tp_fp = np.zeros(n_class, dtype=np.float64)
	tp = np.zeros(n_class, dtype=np.float64)
	
	for parent, dirs, files in os.walk(pred_dir):
		for file in files:
			pred_img_file = os.path.join(parent, file)
			gt_img_file = pred_img_file.replace(pred_dir, gt_dir)
			gt_img_file = gt_img_file.replace('leftImg8bit', 'gtFineId_labelIds')
			pred = np.asarray(
				Image.open(pred_img_file).convert(mode='L'),
				dtype=np.uint8)
			gt = np.asarray(
				Image.open(gt_img_file).convert(mode='L'),
				dtype=np.uint8)
			_tp_fn, _tp_fp, _tp = iou_stats(
				pred,
				gt,
				num_classes=n_class,
				background=0)
			
			tp_fn += _tp_fn
			tp_fp += _tp_fp
			tp += _tp
	
	iou = tp / (tp_fn + tp_fp - tp + 1e-12) * 100.0
	
	class_names = ['Background', 'Aero', 'Bike', 'Bird', 'Boat',
				   'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow',
				   'Table', 'Dog', 'Horse', 'MBike', 'Person',
				   'Plant', 'Sheep', 'Sofa', 'Train', 'TV']
	
	for i in range(n_class):
		print('class {:10s}: {:02d}, acc: {:4.4f}%'.format(class_names[i], i, iou[i]))
	
	mean_iou = iou.sum() / n_class
	print('mean IOU: {:4.4f}%'.format(mean_iou))
	
	mean_pixel_acc = tp.sum() / (tp_fp.sum() + 1e-12)
	print('mean Pixel Acc: {:4.4f}%'.format(mean_pixel_acc))


def get_arguments():
	
	parser = argparse.ArgumentParser(
		description='Benchmark segmentation predictions'
	)
	
	parser.add_argument('--dataset',type=str,default='voc',
						help='dataset')
	parser.add_argument('--pred-dir', type=str, default='',
						help='/path/to/prediction.')
	parser.add_argument('--gt-dir', type=str, default='',
						help='/path/to/ground-truths')
	parser.add_argument('--string-replace', type=str, default=',',
						help='replace the first string with the second one')
	
	return parser.parse_args()

def main():
	args = get_arguments()
	
	if args.dataset.lower()=='voc':
		calcu_voc_mIou(args.pred_dir,args.gt_dir)
	elif args.dataset.lower()=='cityscapes':
		calcu_cityscapes_mIou(args.pred_dir,args.gt_dir)
	else:
		pass

if __name__ == '__main__':
	main()