import glm
import random
import numpy as np
import cv2 as cv
import os

block_size = 1.0

def generate_grid(width, depth):
    # Generates the floor grid locations
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors
    


def load_camera_params_from_xml(cam_name):
    # Absolute path to the config.xml for the given camera
    config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../data/{cam_name}/config.xml")

    print(f"Trying to load config.xml from {config_file_path}")

    # Initialize a dictionary to store the parameters
    cam_params = {}

    # Check if the config.xml file exists
    if not os.path.exists(config_file_path):
        print(f"Error: config.xml not found for {cam_name} at {config_file_path}")
        return None

    # Open the XML config file
    fs = cv.FileStorage(config_file_path, cv.FILE_STORAGE_READ)
    
    # Check if the XML file is loaded successfully
    if not fs.isOpened():
        print(f"Error: Unable to open config.xml for {cam_name}")
        return None

    # Read the camera matrix (3x3)
    cam_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()

    # Read the rotation vector (3x1) and translation vector (3x1)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()

    # Store parameters in the dictionary
    cam_params["camera_matrix"] = cam_matrix
    cam_params["dist_coeffs"] = dist_coeffs
    cam_params["rvec"] = rvec
    cam_params["tvec"] = tvec

    return cam_params


def construct_lookup_table(width, height, depth):
    """
    Constructs a lookup table mapping 3D voxel positions to 2D image plane coordinates in each view.
    """
    lookup_table = {}  # Dictionary to store voxel -> (cam_id, xim, yim) mappings

    for cam_index, cam_name in enumerate(["cam1", "cam2", "cam3", "cam4"]):
        cam_params = load_camera_params_from_xml(cam_name)

        if cam_params is None:
            print(f"Failed to load parameters for {cam_name}")
            continue

        # Camera parameters
        cam_matrix = cam_params["camera_matrix"]
        dist_coeffs = cam_params["dist_coeffs"]
        rvec = cam_params["rvec"]
        tvec = cam_params["tvec"]

        # Convert rvec to rotation matrix
        R, _ = cv.Rodrigues(rvec)

        # Iterate over each voxel in the volume
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    # Compute the voxel's 3D world position
                    voxel_3d = np.array([[x * block_size - width / 2], 
                                         [y * block_size], 
                                         [z * block_size - depth / 2]], dtype=np.float32)

                    # Transform voxel to camera space
                    voxel_cam = R @ voxel_3d + tvec

                    if voxel_cam[2] <= 0:  # Ensure the voxel is in front of the camera
                        continue

                    # Project to 2D image plane
                    img_points, _ = cv.projectPoints(voxel_3d.T, rvec, tvec, cam_matrix, dist_coeffs)

                    if img_points is None or np.isnan(img_points).any() or np.isinf(img_points).any():
                        continue

                    # Ensure the 2D projection fits within the image plane
                    xim, yim = int(img_points[0][0][0]), int(img_points[0][0][1])

                    # Store the mapping in the lookup table
                    voxel_key = (x, y, z)  # Tuple representation of voxel
                    if voxel_key not in lookup_table:
                        lookup_table[voxel_key] = []  # Initialize list for voxel

                    lookup_table[voxel_key].append((cam_index, xim, yim))  # Store camera and 2D position

    return lookup_table


def set_voxel_positions(width, height, depth):
    """
    Uses the lookup table to determine visible voxels based on masks.
    """
    lookup_table = construct_lookup_table(width, height, depth)
    data, colors = [], []

    # Load masks for all cameras
    masks = {}
    for cam_name in ["cam1", "cam2", "cam3", "cam4"]:
        mask_path = os.path.abspath(os.path.join("..", "data", cam_name, "computed_mask.png"))
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Warning: Mask file for {cam_name} not found at {mask_path}. Skipping.")
            return [], []  # Return empty if any mask is missing
        
        print(f"✅ Mask loaded for {cam_name}, shape: {mask.shape}")
        print(f"Unique values in mask: {np.unique(mask)}")  # Should show [0, 255]

        _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)  # Convert to binary
        masks[cam_name] = mask

    # Iterate over the lookup table
    for voxel, projections in lookup_table.items():
        x, y, z = voxel
        visible_in_all_views = True  # Assume it's visible in all views

        for cam_index, xim, yim in projections:
            cam_name = f"cam{cam_index + 1}"  # Map index to camera name
            
            # Ensure integer coordinates
            xim, yim = int(xim), int(yim)

            # Check if coordinates are within image bounds
            if 0 <= xim < masks[cam_name].shape[1] and 0 <= yim < masks[cam_name].shape[0]:
                mask_value = masks[cam_name][yim, xim]  # Get mask value at projected location
                
                # Debugging: Print the mask value for this voxel projection
                #print(f"Voxel {voxel} -> Projects to ({xim}, {yim}) in {cam_name}, Mask Value: {mask_value}")

                # Visualizing Projections on the Mask
                """
                mask_colored = cv.cvtColor(masks[cam_name], cv.COLOR_GRAY2BGR)  # Convert mask to color image
                cv.circle(mask_colored, (xim, yim), 2, (0, 0, 255), -1)  # Draw red dot
                cv.imshow(f"Projections on {cam_name}", mask_colored)  # Show mask with projections
                cv.waitKey(1)  # Small delay to update image
                """
                if mask_value == 0:  # If background, mark as not visible
                    visible_in_all_views = False
                    break  # No need to check other views
            else:
                visible_in_all_views = False  # Out of bounds
                #print(f"Voxel {voxel} -> Projects to ({xim}, {yim}), but it's OUT OF BOUNDS in {cam_name}")
                break

        if visible_in_all_views:
            data.append([x * block_size - width / 2, y * block_size, z * block_size - depth / 2])
            colors.append([x / width, y / height, z / depth])  # Assign some color

    # Final count of voxels generated
    print(f"✅ Total voxels generated: {len(data)}")
    return data, colors


def get_cam_positions():
    """
    Extract camera positions (translation vectors) from the config.xml files of each camera.
    """
    cam_positions = []

    # Loop through all camera folders (assuming camera folders are named 'cam1', 'cam2', etc.)
    for cam_name in ["cam1", "cam2", "cam3", "cam4"]:
        # Load camera parameters from the respective config.xml
        cam_params = load_camera_params_from_xml(cam_name)

        # Extract the translation vector (tvec) which corresponds to the camera position
        tvec = cam_params["tvec"]
        
        # Extract the numerical values from the numpy arrays
        cam_positions.append([float(tvec[0]), float(tvec[1]), float(tvec[2])])

    print(f"Camera positions: {cam_positions}")
    cam_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

    return cam_positions, cam_colors


def get_cam_rotation_matrices():
    """
    Extract camera rotation matrices from the rotation vector (rvec) in each camera's config.xml.
    """
    cam_rotations = []

    # Loop through all camera folders (assuming camera folders are named 'cam1', 'cam2', 'cam3', 'cam4']:
    for cam_name in ["cam1", "cam2", "cam3", "cam4"]:
        # Load camera parameters from the respective config.xml
        cam_params = load_camera_params_from_xml(cam_name)

        # Extract the rotation vector (rvec)
        rvec = cam_params["rvec"]
        
        # Convert the rotation vector to a rotation matrix using Rodrigues' rotation formula
        rotation_matrix, _ = cv.Rodrigues(rvec)
        
        # Add the 3x3 rotation matrix as a 4x4 matrix with an additional row and column for homogeneous coordinates
        rotation_matrix_4x4 = np.eye(4)
        rotation_matrix_4x4[:3, :3] = rotation_matrix
        
        # Append the 4x4 rotation matrix to the list
        cam_rotations.append(rotation_matrix_4x4)

    return cam_rotations
