import h5py
import open3d as o3d
import os
import argparse
import glob
import numpy as np
import matplotlib

def labels_to_cm(labels):
	cm = matplotlib.cm.get_cmap('viridis')
	unique_labels = sorted(list(np.unique(labels)))
	# norm_labels = np.array([unique_labels.index(i)/len(unique_labels) for i in labels])
	norm_labels = np.where(labels==3, 0.99, 0)
	return cm(norm_labels)[:,:3]

def read_pcd(fn):
	gt_pcd = o3d.geometry.PointCloud()
	pred_pcd = o3d.geometry.PointCloud()
	
	with h5py.File(fn,'r') as f:
		n_pts = f['data']['P'].shape[0]
		points = f['data']['P'][()]
		labels = f['data']['I_gt'][()]
		pred_labels = np.argmax(f['pred_result']['instance_per_point'][()], axis=1)
		normals = f['data']['normal_gt'][()]
		pred_normals = f['pred_result']['normal_per_point']

		gt_pcd.points = o3d.utility.Vector3dVector(points)
		gt_pcd.colors = o3d.utility.Vector3dVector(labels_to_cm(labels))
		gt_pcd.normals = o3d.utility.Vector3dVector(normals)

		pred_pcd.points = o3d.utility.Vector3dVector(points)
		pred_pcd.colors = o3d.utility.Vector3dVector(labels_to_cm(pred_labels))
		pred_pcd.normals = o3d.utility.Vector3dVector(pred_normals)

	return gt_pcd, pred_pcd
	
def generate_mesh(fn, args):
	gt_pcd, pred_pcd = read_pcd(fn)
	radii = [0.01, 0.02, 0.05, 0.1, 0.2]
	gt_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(gt_pcd, o3d.utility.DoubleVector(radii))
	pred_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pred_pcd, o3d.utility.DoubleVector(radii))

	return gt_mesh, pred_mesh

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--eval_dir', type=str)
	parser.add_argument('--output_dir', type=str)
	parser.add_argument('--visualize', action='store_true')
	args = parser.parse_args()

	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)

	for fn in glob.glob(os.path.join(args.eval_dir,'*.h5')):
		print(fn)
		gt, pred = generate_mesh(fn, args)
		if args.visualize:
			o3d.visualization.draw_geometries([pred])
		basename = os.path.split(fn)[-1][:-3]
		o3d.io.write_triangle_mesh(os.path.join(args.output_dir, f'{basename}_gt.ply'), gt)
		o3d.io.write_triangle_mesh(os.path.join(args.output_dir, f'{basename}_pred.ply'), pred)

if __name__ == '__main__': main()