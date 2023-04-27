import numpy as np
import h5py
import argparse
import open3d as o3d
import os
import cv2
import glob
from scipy import stats, optimize
import matplotlib.pyplot as plt
import pickle

from meshvis import read_pcd

MIN_PTS_THRESHOLD = 20

def load_params_from_file(fn, metadata=''):
	'''
	Input: 	fn 			-	.h5 filename
			metadata	-	.h5 filename for metadata file (optional, default metadata assumed otherwise)

	Output:	params_per_instance	-	dictionary[instance_id] = dictionary of parameters
			type_per_instance	-	dictionary[instance_id] = type_id
	'''
	type_per_instance = {}
	params_per_instance = {}

	if len(metadata) > 0:
		with h5py.File(metadata,'r') as g:
			name_to_id_dict = pickle.loads(g.attrs['name_to_id_dict'])
	else:
		name_to_id_dict = None

	with h5py.File(fn,'r') as f:
		all_params = f['pred_result']['parameters']
		instance_per_point = np.argmax(f['pred_result']['instance_per_point'], axis=1)
		type_per_point = f['pred_result']['type_per_point']
		index_to_name = pickle.loads(f.attrs['list_of_primitives'])

		# verify index to name if supplied
		if name_to_id_dict:
			for i,name in enumerate(index_to_name):
				assert name_to_id_dict[name] == i, f'Invalid primitive name to id correspondence'

		# find and validate type per instance
		for pt,instance in enumerate(instance_per_point):
			if instance not in type_per_instance.keys():
				type_per_instance[instance] = [type_per_point[pt]]
			else:
				type_per_instance[instance].append(type_per_point[pt])
		# select params per instance
		all_instances = sorted(type_per_instance.keys())
		for instance in all_instances:
			primtype, counts = stats.mode(type_per_instance[instance])
			if counts[0] < MIN_PTS_THRESHOLD:
				type_per_instance.pop(instance)
				continue
			primtype = primtype[0]
			type_per_instance[instance] = primtype
			obj_params = {}
			for paramname in all_params.keys():
				if index_to_name[primtype] in paramname:
					obj_params[paramname] = all_params[paramname][instance]

			# find extra params based on type
			member_pts = f['data']['P'][()][instance_per_point == instance,:]
			if index_to_name[primtype] == 'cylinder':
				all_dist = np.matmul(member_pts, obj_params['cylinder_axis'])
				max_height = np.amax(all_dist) - np.amin(all_dist)
				obj_params['cylinder_height'] = max_height
			elif index_to_name[primtype] == 'cone':
				all_dist = np.matmul(member_pts, obj_params['cone_axis'])
				max_height = np.amax(np.abs(all_dist - np.matmul(obj_params['cone_apex'], obj_params['cone_axis'])))
				bbox_axis_length = max_height - np.amin(np.abs(all_dist - np.matmul(obj_params['cone_apex'], obj_params['cone_axis'])))
				radius = max_height * np.tan(obj_params['cone_half_angle'])
				bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-radius, -radius, 0]), max_bound=np.array([radius, radius, bbox_axis_length]))
				obj_params['cone_height'] = max_height
				obj_params['cone_bbox'] = bbox
			elif index_to_name[primtype] == 'plane':
				obj_params['plane_alpha'] = 0.5
			obj_pcd = o3d.geometry.PointCloud()
			obj_pcd.points = o3d.utility.Vector3dVector(member_pts)
			obj_params['obj_pcd'] = obj_pcd
			obj_params['obj_npts'] = member_pts.shape[0]
			params_per_instance[instance] = obj_params

			# print(instance, index_to_name[primtype], obj_params)

	return params_per_instance, type_per_instance, index_to_name


def distance(pcd, mesh, n_pts=8000):
	'''
	Measure Chamfer distance between mesh and pcd

	Input:	pcd 	-	open3d PointCloud
			mesh 	-	open3d TriangleMesh
			n_pts 	-	points to sample from mesh for point cloud conversion (int)

	Output: mean (float), median (float)
	'''
	mesh_pcd = mesh.sample_points_uniformly(number_of_points=n_pts)
	dist = np.array(pcd.compute_point_cloud_distance(mesh_pcd))
	return np.mean(dist), np.median(dist)

def rotation_from_primitive_axis(axis):
	'''
	Input: axis 	-	central axis (unit vector)
	Output:	R 		- 	3x3 rotation matrix
	'''
	# construct rotation vector v = a x 0 for 0=[0,0,1]
	zero_vec = np.array([0,0,1])
	v = np.cross(axis, zero_vec)
	v = v / np.linalg.norm(v) * np.arccos(np.dot(axis, zero_vec) / np.linalg.norm(axis))
	result, _ = cv2.Rodrigues(v)
	return result

def generate_primitives(params_per_instance, type_per_instance, index_to_name):
	all_geometry = []
	cm = plt.cm.viridis
	weighted_mean, weighted_median, total_pts = 0, 0, 0
	for i,instance in enumerate(type_per_instance.keys()):
		# if instance != 3: continue
		primtype = type_per_instance[instance]
		params = params_per_instance[instance]
		if index_to_name[primtype] == 'cylinder':
			height = params['cylinder_height']
			geom = o3d.geometry.TriangleMesh.create_cylinder(radius=np.sqrt(params['cylinder_radius_squared']),
													 height=height)
			geom.rotate(rotation_from_primitive_axis(params['cylinder_axis']))
			geom.translate(params['cylinder_center'])
		elif index_to_name[primtype] == 'cone':
			height = params['cone_height']
			radius = height * np.tan(params['cone_half_angle'])
			rot = rotation_from_primitive_axis(params['cone_axis'])
			geom = o3d.geometry.TriangleMesh.create_cone(radius=radius,
												 		 height=height,
												 		 split=20)
			geom = geom.crop(params['cone_bbox'])
			geom.rotate(rot)
			tip = np.array(params['cone_apex']).reshape(3,1) - np.matmul(rot, np.array([0,0,height]).reshape(3,1))
			geom.translate(tip)
		elif index_to_name[primtype] == 'plane':
			geom = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(params['obj_pcd'], params['plane_alpha'])
		elif index_to_name[primtype] == 'sphere':
			geom = o3d.geometry.TriangleMesh.create_sphere(radius=np.sqrt(params['sphere_radius_squared']))
			geom.translate(params['sphere_center'])
		geom.compute_vertex_normals()
		geom.paint_uniform_color(cm(i/len(type_per_instance.keys()))[:3])
		all_geometry.append(geom)

		# compute constituent distances
		n_pts = params['obj_npts']
		total_pts += n_pts
		mean, median = distance(params['obj_pcd'], geom, n_pts=n_pts)
		# print(f'   {instance:>2}\t\tmean: {mean:>.4}\tmedian: {median:>.4}')
		weighted_mean += mean * n_pts
		weighted_median += median * n_pts
	print(f'weighted\tmean: {weighted_mean / total_pts:>.4}\tmedian: {weighted_mean / total_pts:>.4}')

	# DEBUG ONLY
	# base_geom = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
	# base_geom.paint_uniform_color([1,0,0])
	# all_geometry.append(base_geom)

	return all_geometry

def custom_draw_geometry_with_camera_trajectory(geometries, all_pos, output_path, output_folder="joint"):
	custom_draw_geometry_with_camera_trajectory.index = -1
	custom_draw_geometry_with_camera_trajectory.rot = all_pos
	custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()
	joint_path = os.path.join(output_path, output_folder)
	if not os.path.isdir(joint_path):
		os.makedirs(joint_path)

	def move_forward(vis):
		# This function is called within the o3d.visualization.Visualizer::run() loop
		# The run loop calls the function, then re-render
		# So the sequence in this function is to:
		# 1. Capture frame
		# 2. index++, check ending criteria
		# 3. Set camera
		# 4. (Re-render)
		ctr = vis.get_view_control()
		# ctr.rotate(10.0, 0.0)
		glb = custom_draw_geometry_with_camera_trajectory
		if glb.index >= 0:
			print("Capture image {:05d}".format(glb.index))
			image = vis.capture_screen_float_buffer(False)
			depth = vis.capture_depth_float_buffer(False)

			plt.figure()
			# plt.subplot(121)
			plt.axis('off')
			plt.imshow(np.asarray(image))
			# plt.subplot(122)
			# plt.axis('off')
			# plt.imshow(np.asarray(depth))
			plt.tight_layout()
			plt.savefig(os.path.join(joint_path, '{:05d}.png'.format(glb.index)))
			plt.close()
		glb.index = glb.index + 1
		if glb.index < len(glb.rot):
			ctr.rotate(glb.rot[glb.index], 0.0)
		else:
			custom_draw_geometry_with_camera_trajectory.vis.\
				register_animation_callback(None)
			exit()
		return False

	vis = custom_draw_geometry_with_camera_trajectory.vis
	vis.create_window()
	for g in geometries:
		vis.add_geometry(g)
	vis.register_animation_callback(move_forward)
	vis.run()
	vis.destroy_window()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--files', type=str, help='Files to load and render (glob style)')
	parser.add_argument('--metadata', type=str, default='', help='File from which to load metadata')
	parser.add_argument('--output', type=str, default='', help='Output 3D rotation rendering')
	parser.add_argument('--comp_alpha', action='store_true', help='Compute pcd distance from alpha shape')
	parser.add_argument('--render_interactive', action='store_true', help='Render interactive display')
	parser.add_argument('--render', action='store_true', help='Render visualization (non-interactive)')
	args = parser.parse_args()

	for fn in glob.glob(args.files):
		_, basename = os.path.split(fn)
		basename = basename[:-len('_bundle.h5')]
		print(basename)
		_, pcd = read_pcd(fn)
		ppi, tpi, itn = load_params_from_file(fn, args.metadata)
		primitives = generate_primitives(ppi, tpi, itn)

		if args.render_interactive:
			o3d.visualization.draw_geometries([pcd, *primitives])

		if args.comp_alpha:
			hull, _ = pcd.compute_convex_hull(joggle_inputs=True)
			alpha_mean, alpha_med = distance(pcd, hull)

			print('-'*25)
			print(f'  alpha\t\tmean: {alpha_mean:>.4}\tmedian: {alpha_med:>.4}')
			print('-'*25)
			print('-'*25)

			if args.render_interactive:
				o3d.visualization.draw_geometries([pcd, hull])

		if args.render and len(args.output) > 0:
			full_output = os.path.join(args.output, basename)
			if not os.path.isdir(full_output):
				os.makedirs(full_output)
			# rotation units pixels, 0.003 radians/pix
			d_deg = 5
			d_pix = float(d_deg) * np.pi/180 / 0.003
			camera_trajectory = [d_pix for _ in range(0,360,d_deg)]
			custom_draw_geometry_with_camera_trajectory([pcd, *primitives], camera_trajectory, full_output, output_folder='joint')
			# custom_draw_geometry_with_camera_trajectory([pcd], camera_trajectory, full_output, output_folder='pcd_singleprim')
			# custom_draw_geometry_with_camera_trajectory(primitives, camera_trajectory, full_output, output_folder='primitives')


if __name__ == '__main__': main()
