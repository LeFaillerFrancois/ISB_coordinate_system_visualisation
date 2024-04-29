# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:27:26 2024

@author: flefaill
"""
###############################################################################
###########################IMPORTING LIBRARIES#################################
###############################################################################

import numpy as np
import pandas as pd
# import xarray as xr # For EMGs, not analysed in this script

import statistics # Could serve for peak detection
from scipy.signal import find_peaks
from scipy import signal
from scipy.interpolate import interp1d

import scipy.io
# from pyomeca import Analogs # For EMGs, not analysed in this script
from scipy.spatial.transform import Rotation

# Visualisation libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation


###############################################################################
#####################LOADING FILES WITH CORRECT TIME###########################
###############################################################################

path = "data/punch_kinematics.mat"
path_force = "data/punch_force.txt"

force_frequency = 5000 #Hz
mocap_frequency = 200 #Hz
grf_frequency = 200 #Hz

# Loading file. File is in .mat because .c3d can only have 1 frequency for analogs
def load_mat_file(filename):
    file = scipy.io.loadmat(filename, simplify_cells=True)
    filename_in_dict=list(file.keys())[-1]
    data=file[filename_in_dict]
    return data

# Used to get the real start time of the acc signal as only the force signal was timed with the initial trigger
# But both ended with the end trigger. i.e. Synch purpose.
def get_time_of_trigger_start(path) :
    # Loading the force vector from the txt
    def load_force_from_txt (txt_path):
        force = pd.read_csv(txt_path, index_col=False, header=0, decimal=",", delimiter="\t")
        trigger=force["Trigger [V]"]
        
        # Make the trigger easier to detect
        trigger.replace('****', np.nan, inplace=True)
        index_trigger = trigger.isnull().idxmax() # Trigger event index
        
        force = force.iloc[index_trigger:len(force),:].reset_index(drop=True) # Crop before trigger
        return force[['Fx [N]','Fy [N]','Fz [N]']]
    
    force=load_force_from_txt(path)
    time=len(force)*1/force_frequency
    return time

# Converting time in frames to time in sec knowing the frequency of the device.
def reconstruct_time_vector(num_frames, acquisition_frequency):
    total_duration = num_frames / acquisition_frequency
    time_vector = np.linspace(0, total_duration, num=num_frames)
    return time_vector

# Low pass butterworth filter
def LP_filter(x,cut_off_frequency,recording_frequency):
    # Low pass filter, no phase shift
    b, a = signal.butter(4, cut_off_frequency, btype='low', analog=False, output='ba', fs=recording_frequency)
    y = signal.filtfilt(b, a, x, padlen=150)
    return y

time = get_time_of_trigger_start(path_force)
data = load_mat_file(path)

data_info = data["Trajectories"]["Labeled"]
marker_names = data_info["Labels"]
marker_data = data_info["Data"]
print(marker_names)

def interpolate_missing_values(col):
    #col = array.copy()
    col_non_missing_indices = np.where(np.isfinite(col))[0]
    if len(col_non_missing_indices) > 0:
        linear_quadra = interp1d(col_non_missing_indices, col[col_non_missing_indices], kind='quadratic', fill_value="extrapolate")
        col_interp = linear_quadra(np.arange(len(col)))
    else:
        col_interp = col
    return col_interp

# Getting a marker by its name. Synch it with the force sensor.
def select_marker(data_info, marker_name) :
    marker_data = data_info["Data"]
    marker_names = data_info["Labels"]
    marker = marker_data[marker_names == marker_name][0][0:3,:]
    
    marker_at_trigger = marker[:,marker.shape[1]-round(time*mocap_frequency):marker.shape[1]]

    # Filtering 
    marker_at_trigger_without_missing_values = np.apply_along_axis(interpolate_missing_values, axis=1, arr=marker_at_trigger)
    marker_filtered = LP_filter(marker_at_trigger_without_missing_values, 6, mocap_frequency)
    return marker_filtered

first_frame = marker_data.shape[2]-round(time*mocap_frequency)
# Actual number of frames 
random_marker = select_marker(data_info, "PAO")
frames = random_marker.shape[1]  # Number of frames

###############################################################################
###########################RETRIEVING MARKERS##################################
###############################################################################


# Pelvis
REIAS = select_marker(data_info, "REIAS")
LEIAS = select_marker(data_info, "LEIAS")
REIPS = select_marker(data_info, "REIPS")
LEIPS = select_marker(data_info, "LEIPS")

# Knee
RKNE = select_marker(data_info, "RKNE")
RKNI = select_marker(data_info, "RKNI")

LKNE = select_marker(data_info, "LKNE")
LKNI = select_marker(data_info, "LKNI")

# Ankle
RMALI = select_marker(data_info, "RMALI")
RMALE = select_marker(data_info, "RMALE")

LMALI = select_marker(data_info, "LMALI")
LMALE = select_marker(data_info, "LMALE")

# Foot/toe
RTOE = select_marker(data_info, "RTOE")
LTOE = select_marker(data_info, "LTOE")

# Thorax
IJ = select_marker(data_info, "IJ")
STER = select_marker(data_info, "STER")
C7 = select_marker(data_info, "C7")
T10 = select_marker(data_info, "T10")

# Shoulder
RACR = select_marker(data_info, "RACR")
LACR = select_marker(data_info, "LACR")

# Elbow
RHLE = select_marker(data_info, "RHLE")
RHME = select_marker(data_info, "RHME")

LHLE = select_marker(data_info, "LHLE")
LHME = select_marker(data_info, "LHME")

# Forearm
RUSP = select_marker(data_info, "RUSP")
RRSP = select_marker(data_info, "RRSP")
LUSP = select_marker(data_info, "LUSP")
LRSP = select_marker(data_info, "LRSP")

RGLOVE = select_marker(data_info,"RGLOVE") / 1000
LGLOVE = select_marker(data_info,"LGLOVE") / 1000

###############################################################################
###############CALCULATING JOINT CENTERS AND ROTATION MATRIX###################
###############################################################################

##############################LOWER BODY#######################################

# Lower body According to https://pubmed.ncbi.nlm.nih.gov/11934426/
# Wu, G., Siegler, S., Allard, P., Kirtley, C., Leardini, A., Rosenbaum, D.,
# Whittle, M., D'Lima, D. D., Cristofolini, L., Witte, H., Schmid, O.,
# Stokes, 
# I., & Standardization and Terminology Committee of the International Society of Biomechanics (2002). 
# ISB recommendation on definitions of joint coordinate system of various joints for the reporting of human joint motion--
# part I: ankle, hip, and spine. International Society of Biomechanics. 
# Journal of biomechanics, 35(4), 543–548. https://doi.org/10.1016/s0021-9290(01)00222-6

# Pelvis center
Op = (REIAS+REIPS+LEIAS+LEIPS)/4

# Knee center
Ork = (RKNE+RKNI)/2
Olk = (LKNE+LKNI)/2

# Ankle center
Ora = (RMALI+RMALE)/2
Ola = (LMALI+LMALE)/2

# Pelvis coordinate systeme
zrp = (REIAS-LEIAS)/np.linalg.norm(REIAS-LEIAS,axis=0)
xrp = (0.5*(REIAS+LEIAS)-0.5*(LEIAS+LEIPS)) / np.linalg.norm(0.5*(REIAS+LEIAS)-0.5*(LEIAS+LEIPS),axis = 0)
yrp=np.cross(zrp,xrp,axis=0)
xrp=np.cross(yrp,zrp,axis=0) 
oRrp = np.array([xrp,yrp,zrp])

zlp = (LEIAS-REIAS)/np.linalg.norm(LEIAS-REIAS,axis=0)
xlp = (0.5*(REIAS+LEIAS)-0.5*(LEIAS+LEIPS)) / np.linalg.norm(0.5*(REIAS+LEIAS)-0.5*(LEIAS+LEIPS),axis = 0)
ylp = np.cross(xlp,zlp,axis=0)
xlp = np.cross(ylp,-zlp,axis=0)
oRlp = np.array([xlp,ylp,zlp])

# Thigh center 
# /!\ ref missing
pOrt = np.array([-74,-46,-74]) # Coordinate of thigh in the pelvis coordinate # /!\ ref missing
pOlt = np.array([-74,-46,74]) # Coordinate of thigh in the pelvis coordinate # /!\ ref missing
Ort = np.zeros((3,oRrp.shape[2])) 
Olt = np.zeros((3,oRrp.shape[2])) 
for f in range(oRrp.shape[2]):
    Ort[:,f] = oRrp[:,:,f] @ pOrt + Op[:,f]
    Olt[:,f] = oRrp[:,:,f] @ pOlt + Op[:,f]

# Thigh coordinate systeme
yrt = (Ort-Ork) / np.linalg.norm(Ort-Ork,axis=0)
zrt = (RKNE-RKNI) / np.linalg.norm(RKNE-RKNI,axis=0)
xrt = np.cross(yrt,zrt,axis=0)
zrt = np.cross(yrt,-xrt,axis=0)
Rrt = np.array([xrt,yrt,zrt])

ylt = (Olt-Olk) / np.linalg.norm(Olt-Olk,axis=0)
zlt = (LKNE-LKNI) / np.linalg.norm(LKNE-LKNI,axis=0)
xlt = np.cross(zlt,ylt,axis=0)
zlt = np.cross(ylt,xlt,axis=0)
Rlt = np.array([xlt,ylt,zlt])

# # Knee coordinate system ?
# yrk = (Ork-Ora) / np.linalg.norm(Ork-Ora,axis=0)
# zrk = (RMALE-RMALI) / np.linalg.norm(RMALE-RMALI,axis=0)
# xrk = np.cross(yrk,zrk,axis=0)
# zrk = np.cross(yrk,-xrk,axis=0)
# Rrk = np.array([xrk,yrk,zrk])

# ylk = (Olk-Ola) / np.linalg.norm(Olk-Ola,axis=0)
# zlk = (LMALE-LMALI) / np.linalg.norm(LMALE-LMALI,axis=0)
# xlk = np.cross(zlk,ylk,axis=0)
# zlk = np.cross(ylk,xlk,axis=0)
# Rlk = np.array([xlk,ylk,zlk])
# #doesn't exist on ISB, but could be usefull

# Tibia/Fibula coordinate system
zra = (RMALE-RMALI) / np.linalg.norm(RMALE-RMALI,axis=0)
xra = np.cross(Ork-RMALI,RMALE-RMALI,axis=0) / np.linalg.norm(np.cross((Ork-RMALI),(RMALE-RMALI),axis=0),axis=0)
yra = np.cross(zra,xra,axis=0) 
Rra = np.array([xra,yra,zra])

zla = (LMALE-LMALI) / np.linalg.norm(LMALE-LMALI,axis=0)
xla = np.cross(LMALE-LMALI,Olk-LMALI,axis=0) / np.linalg.norm(np.cross((LMALE-LMALI),(Olk-LMALI),axis=0),axis=0)
yla = np.cross(zla,-xla,axis=0) 
Rla = np.array([xla,yla,zla])

# Calcaneus coordinate system
yrc = (Ork-Ora) / np.linalg.norm(Ork-Ora,axis=0)
xrc = np.cross(RKNE-RKNI,Ora-RKNI,axis=0) / np.linalg.norm(np.cross((RKNE-RKNI),(Ora-RKNI),axis=0),axis=0)
zrc = np.cross(xrc,yrc,axis=0)
xrc = np.cross(yrc,zrc,axis=0)
Rrc = np.array([xrc,yrc,zrc])

ylc = (Olk-Ola) / np.linalg.norm(Olk-Ola,axis=0)
xlc = np.cross(LKNE-LKNI,Ola-LKNI,axis=0) / np.linalg.norm(np.cross((LKNE-LKNI),(Ola-LKNI),axis=0),axis=0)
zlc = np.cross(xlc,ylc,axis=0)
xlc = np.cross(ylc,-zlc,axis=0)
Rlc = np.array([xlc,ylc,zlc])

##############################UPPER BODY#######################################

# Upper body according to : 
# Wu, G., van der Helm, F. C., Veeger, H. E., Makhsous, M., Van Roy, P., Anglin,
# C., Nagels, J., Karduna, A. R., McQuade, K., Wang, X., Werner, F. W., Buchholz, 
# B., & International Society of Biomechanics (2005). 
# ISB recommendation on definitions of joint coordinate systems of various joints
# for the reporting of human joint motion--Part II: shoulder, elbow, wrist and hand.
# Journal of biomechanics, 38(5), 981–992. 
# https://doi.org/10.1016/j.jbiomech.2004.05.042

# Thorax center
thorax_center_high = (IJ+C7) / 2
thorax_center_low = (STER+T10) / 2

# Thorax coordinate system
ythorax = (thorax_center_high-thorax_center_low) / np.linalg.norm(thorax_center_high-thorax_center_low, axis=0)
zthorax = np.cross(IJ-thorax_center_low,C7-thorax_center_low,axis=0) / np.linalg.norm(np.cross((IJ-thorax_center_low),(C7-thorax_center_low),axis=0),axis=0)
xthorax = np.cross(ythorax,zthorax,axis=0)
zthorax = np.cross(ythorax,-xthorax,axis=0)
Rthorax = np.array([xthorax,ythorax,zthorax])

# Clavicle coordinate system
zrclav = (RACR-IJ) / np.linalg.norm(RACR-IJ,axis=0)
xrclav = np.cross(ythorax,zrclav,axis=0)
yrclav = np.cross(zrclav,xrclav,axis=0)
Rrclav = np.array([xrclav,yrclav,zrclav])

zlclav = (LACR-IJ) / np.linalg.norm(LACR-IJ,axis=0)
xlclav = np.cross(ythorax,-zlclav,axis=0)
ylclav = np.cross(xlclav,zlclav,axis=0)
Rlclav = np.array([xlclav,ylclav,zlclav])

# Scapula is not possible to compute with our marker set
# Missing scapula
# Missing scapula
# Missing scapula
# Missing scapula

# Shoulder center
# /!\ ref missing
pOrshoulder = np.array([18.9,-3.9,-9.3]) # Coordinate of shoulder in the acromion coordinate # /!\ ref missing
pOlshoulder = np.array([18.9,-3.9,9.3]) # Coordinate of shoulder in the acromion coordinate # /!\ ref missing
Orshoulder = np.zeros((3,oRrp.shape[2])) 
Olshoulder = np.zeros((3,oRrp.shape[2])) 
for f in range(oRrp.shape[2]):
    Orshoulder[:,f] = pOrshoulder + RACR[:,f]
    Olshoulder[:,f] = pOlshoulder + LACR[:,f]

# Elbow center
Orelbow = (RHME+RHLE) / 2
Olelbow = (LHME+LHLE) / 2

# Humerus coordinate system
yrhum = (Orshoulder - Orelbow) / np.linalg.norm(Orshoulder - Orelbow, axis = 0)
xrhum = np.cross(RHLE-Orshoulder,RHME-Orshoulder,axis=0) / np.linalg.norm(np.cross((RHLE-Orshoulder),(RHME-Orshoulder),axis=0),axis=0)
zrhum = np.cross(yrhum,-xrhum,axis=0)
xrhum = np.cross(yrhum,zrhum,axis=0)
Rrhum = np.array([xrhum,yrhum,zrhum])

ylhum = (Olshoulder - Olelbow) / np.linalg.norm(Olshoulder - Olelbow, axis = 0)
xlhum = np.cross(LHME-Olshoulder,LHLE-Olshoulder,axis=0) / np.linalg.norm(np.cross((LHME-Olshoulder),(LHLE-Olshoulder),axis=0),axis=0)
zlhum = np.cross(ylhum,xlhum,axis=0)
xlhum = np.cross(ylhum,-zlhum,axis=0)
Rlhum = np.array([xlhum,ylhum,zlhum])

# Wrist center
Orw = (RUSP+RRSP) / 2
Olw = (LUSP+LRSP) / 2

# Forearm coordinate system
yrforearm = (Orelbow-RUSP) / np.linalg.norm(Orelbow-RUSP,axis=0)
xrforearm = np.cross(RRSP-Orelbow,RUSP-Orelbow,axis=0) / np.linalg.norm(np.cross((RRSP-Orelbow),(RUSP-Orelbow),axis=0),axis=0)
zrforearm = np.cross(yrforearm,-xrforearm,axis=0)
Rrforearm = np.array([xrforearm,yrforearm,zrforearm])

ylforearm = (Olelbow-LUSP) / np.linalg.norm(Olelbow-LUSP,axis=0)
xlforearm = -np.cross(LRSP-Olelbow,LUSP-Olelbow,axis=0) / np.linalg.norm(np.cross((LRSP-Olelbow),(LUSP-Olelbow),axis=0),axis=0)
zlforearm = np.cross(ylforearm,xlforearm,axis=0)
Rlforearm = np.array([xlforearm,ylforearm,zlforearm])


###############################################################################
###############GETTING GROUND REACTION FORCES AND COP##########################
###############################################################################

# Getting ground reaction force from file
def get_grf(mat_path):
    grf = load_mat_file(mat_path)
    grf = grf["Force"]
    left_grf = grf[0] # Left grf plate
    right_grf = grf[1] # Right grf plate
    return left_grf,right_grf

# Getting grf and cop, and synchronizing them with the force sensor
def work_flow_get_grf_and_cop(mat_path,force_path):
    left_grf, right_grf = get_grf(mat_path)
    left_cop=get_grf(mat_path)[0]["COP"]
    right_cop=get_grf(path)[1]["COP"]
    left_grf, right_grf = left_grf["Force"],right_grf["Force"]
    time = get_time_of_trigger_start(force_path)
    left_grf_trigger = left_grf[:,left_grf.shape[1]-round(time*grf_frequency):left_grf.shape[1]]
    left_cop_trigger = left_cop[:,left_cop.shape[1]-round(time*grf_frequency):left_cop.shape[1]]
    right_grf_trigger = right_grf[:,left_grf.shape[1]-round(time*grf_frequency):right_grf.shape[1]]
    right_cop_trigger = right_cop[:,right_cop.shape[1]-round(time*grf_frequency):right_cop.shape[1]]
    return left_grf_trigger, right_grf_trigger, left_cop_trigger, right_cop_trigger

left_grf, right_grf, left_cop, right_cop = work_flow_get_grf_and_cop(path,path_force)

left_grf_norm=np.linalg.norm(left_grf, axis=0)
right_grf_norm=np.linalg.norm(right_grf, axis=0)

###############################################################################
##############VISUALISATION OF MODEL COORDINATE SYSTEM#########################
###############################################################################

# Class to plot arrows on a 3D plot. Used to visualise coordinate systems.
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

# Plotting the coordinate systems by plotting 3 arrows, knowing their origin and rotation matrix
def plot_joint_coordinate_system(joint_pose, joint_matrix, ax, frame=0):
    xx = joint_matrix[0, 0, frame] / 5 # / 5 is the size of the coordinate frame 
    xy = joint_matrix[0, 1, frame] / 5 
    xz = joint_matrix[0, 2, frame] / 5    

    yx = joint_matrix[1, 0, frame] / 5
    yy = joint_matrix[1, 1, frame] / 5
    yz = joint_matrix[1, 2, frame] / 5

    zx = joint_matrix[2, 0, frame] / 5
    zy = joint_matrix[2, 1, frame] / 5
    zz = joint_matrix[2, 2, frame] / 5

    xO = joint_pose[0, frame] / 1000 # / 1000 => meters
    yO = joint_pose[1, frame] / 1000
    zO = joint_pose[2, frame] / 1000
    
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', shrinkA=0, shrinkB=0)
    x_axis = Arrow3D([xO, xO+xx], [yO, yO+xy], [zO, zO+xz], **arrow_prop_dict, color='r')
    ax.add_artist(x_axis)
    y_axis = Arrow3D([xO, xO+yx], [yO, yO+yy], [zO, zO+yz], **arrow_prop_dict, color='g')
    ax.add_artist(y_axis)
    z_axis = Arrow3D([xO, xO+zx], [yO, yO+zy], [zO, zO+zz], **arrow_prop_dict, color='b')
    ax.add_artist(z_axis)

# Plotting a bone between 2 3D points.
def plot_bone(point1, point2, ax, color="black", frame = 0):
    x_line = np.array([point1[0, frame], point2[0, frame]]) / 1000
    y_line = np.array([point1[1, frame], point2[1, frame]]) / 1000
    z_line = np.array([point1[2, frame], point2[2, frame]]) / 1000
    
    ax.plot(x_line, y_line, z_line, color=color)

# Plotting every labelised markers in the file.
def plot_markers(data_info, marker_names,ax, frame = 0):
    for marker_name in marker_names : 
        marker_temp = select_marker(data_info,marker_name) / 1000
        x = marker_temp[0,frame]
        y = marker_temp[1,frame]
        z = marker_temp[2,frame]
        ax.scatter(x, y, z,color="grey")
        
# Plotting the ground reaction force, still working on it.          
def plot_grf(cop, grf, ax, frame=0, foot="left"):
    # There is still a problem with this one (COP), cant find out
    if foot == "left":
        yoffset = 1800 # Location of the force plate
        xoffset = 300 # Location of the force plate
        xx = grf[0, frame] / 1000
        xy = grf[1, frame] / 1000
        xz = grf[2, frame] / 1000  

        xO = (xoffset+cop[0, frame] /10) /1000
        yO = (yoffset+cop[1, frame] /10) /1000
        zO = cop[2, frame] / 1000 # Actually is 0 all the way, z being up
        
    if foot == "right":
        yoffset = 600 # Location of the force plate
        xoffset = 300 # Location of the force plate
        xx = grf[0, frame] / 1000
        xy = grf[1, frame] / 1000 * -1 # This force plate is rotated 90°
        xz = grf[2, frame] / 1000 

        xO = (xoffset+ cop[0, frame] /10) / 1000
        yO = (yoffset+ cop[1, frame] /10) / 1000
        zO = cop[2, frame] / 1000 # Actually is 0 all the way, z being up
    
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', shrinkA=0, shrinkB=0)
    a = Arrow3D([xO, xO+xx], [yO, yO+xy], [zO, zO+xz], **arrow_prop_dict, color='yellow')
    ax.add_artist(a)
        
###############################MODEL VISUALISATION#############################  

# Creating the plot
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

# Function to update the plot of the model for each frame
# Chosing to plot or not to plot the markers, bones, coordinate systems and ground reaction forces
def update(frame, markers = False, bones = True, coordinate_systems = True, grf = True):
    ax1.clear()
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_xlim(-0.4, 1)
    ax1.set_ylim(0, 1.4)
    ax1.set_zlim(0, 1.4)
    ax1.set_title('3D Joint coordinates system visualisation')
    
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', shrinkA=0, shrinkB=0)
    x_axis = Arrow3D([0, 0.5], [0,0], [0,0], **arrow_prop_dict, color='r',label="x")
    ax1.add_artist(x_axis)
    y_axis = Arrow3D([0,0], [0,0.5], [0,0], **arrow_prop_dict, color='g',label="y")
    ax1.add_artist(y_axis)
    z_axis = Arrow3D([0,0], [0,0], [0,0.5], **arrow_prop_dict, color='b',label="z")
    ax1.add_artist(z_axis)
    
    ax1.legend()
    
    if markers :
        # Plotting the markers
        plot_markers(data_info, marker_names, ax1, frame = frame)
    
    if coordinate_systems :
        # Plotting pelvis coordinates
        # plot_joint_coordinate_system(Ort, oRrp, ax1, frame=frame)
        # plot_joint_coordinate_system(Olt, oRlp, ax1, frame=frame)
        plot_joint_coordinate_system(Op, oRrp, ax1, frame=frame) # Visualisation purpose only, ISB are the two above
        # Plotting thighs coordinates
        plot_joint_coordinate_system(Ort, Rrt, ax1, frame=frame)
        plot_joint_coordinate_system(Olt, Rlt, ax1, frame=frame)
        # Plotting tibia fibula coordinates
        # plot_joint_coordinate_system(Ora, Rra, ax1, frame=frame)
        # plot_joint_coordinate_system(Ola, Rla, ax1, frame=frame)
        plot_joint_coordinate_system(Ork, Rra, ax1, frame=frame) # Visualisation purpose only, ISB are the two above
        plot_joint_coordinate_system(Olk, Rla, ax1, frame=frame) # Visualisation purpose only, ISB are the two above
        # Plotting calcaneus coordinates
        plot_joint_coordinate_system(Ora, Rrc, ax1, frame=frame)
        plot_joint_coordinate_system(Ola, Rlc, ax1, frame=frame)
        
        # Plotting thorax coordinates
        plot_joint_coordinate_system(IJ, Rthorax, ax1, frame=frame)
        # Plotting clavicles coordinates
        plot_joint_coordinate_system(IJ, Rrclav, ax1, frame=frame)
        plot_joint_coordinate_system(IJ, Rlclav, ax1, frame=frame)
        # Plotting humerus coordinates
        plot_joint_coordinate_system(Orshoulder, Rrhum, ax1, frame=frame)
        plot_joint_coordinate_system(Olshoulder, Rlhum, ax1, frame=frame)
        # Plotting forearm coordinates
        # plot_joint_coordinate_system(RUSP, Rrforearm, ax1, frame=frame)
        # plot_joint_coordinate_system(LUSP, Rlforearm, ax1, frame=frame)
        plot_joint_coordinate_system(Orelbow, Rrforearm, ax1, frame=frame) # Visualisation purpose only, ISB are the two above
        plot_joint_coordinate_system(Olelbow, Rlforearm, ax1, frame=frame) # Visualisation purpose only, ISB are the two above
    
    if bones : 
        # Plotting right leg
        plot_bone(Ort, Op, ax1, color="r", frame=frame)
        plot_bone(Ort, Ork, ax1, color="r", frame=frame)
        plot_bone(Ork, Ora, ax1, color="r", frame=frame)
        plot_bone(Ora, RTOE, ax1, color="r", frame=frame)
        # Plotting left leg
        plot_bone(Olt, Op, ax1, color="g", frame=frame)
        plot_bone(Olt, Olk, ax1, color="g", frame=frame)
        plot_bone(Olk, Ola, ax1, color="g", frame=frame)
        plot_bone(Ola, LTOE, ax1, color="g", frame=frame)
        # Plotting thorax
        plot_bone(Op, C7, ax1, color="black", frame=frame)
        # Plotting clavicles
        plot_bone(IJ, RACR, ax1, color="r", frame=frame)
        plot_bone(IJ, LACR, ax1, color="g", frame=frame)
        # Plotting humerus
        plot_bone(Orshoulder, Orelbow, ax1, color="r", frame=frame)
        plot_bone(Olshoulder, Olelbow, ax1, color="g", frame=frame)
        # Plotting forearms
        plot_bone(Orw, Orelbow, ax1, color="r", frame=frame)
        plot_bone(Olw, Olelbow, ax1, color="g", frame=frame)
        
    if grf : 
        plot_grf(left_cop, left_grf, ax1, frame=frame, foot="left")    
        plot_grf(right_cop, right_grf, ax1, frame=frame, foot="right")    

    # Hands
    ax1.scatter(RGLOVE[0,frame], RGLOVE[1,frame], RGLOVE[2,frame],color="r",s=20)
    ax1.scatter(LGLOVE[0,frame], LGLOVE[1,frame], LGLOVE[2,frame],color="g",s=20)

# Create the animation
ani = FuncAnimation(fig1, update, frames=frames, interval=1/mocap_frequency) # To animate

# To plot a particular frame
#update(frame=1000, markers=True, bones = True, coordinate_systems = True, grf = True) # To plot a particular frame

# Show the animation/frame
plt.show()

###############################################################################
###########################VISUALISATION OF PUNCHS#############################
###############################################################################

# Derivative fonction for a serie (getting speed and acceleration)
def diff2(x,dt):
    y=np.zeros(len(x))
    n=len(x)-1
    y[0]=(-x[2]+4*x[1]-3*x[0])/(2*dt)
    y[n]=(3*x[n]-4*x[n-1]+x[n-2])/(2*dt) 
    for i in range(1,n-1):
        y[i]=(x[i+1]-x[i-1])/(2*dt)
    return y

# Derivative fonction for joint coordinates
def diff_vec3(x,dt):
    y=np.zeros(x.shape)
    y[0,:]=diff2(x[0,:],dt)
    y[1,:]=diff2(x[1,:],dt)
    y[2,:]=diff2(x[2,:],dt)
    return y

# Find the peaks in the data, return their index
def find_peaks_in_speed(data_series):
    # Use find_peaks to find the peaks
    peaks, _ = find_peaks(data_series,prominence=1,height=2,distance=100)#, distance=20, prominence=100)
    return peaks

RGLOVE_speed = diff_vec3(RGLOVE, 1/mocap_frequency)
RGLOVE_speed = LP_filter(RGLOVE_speed, 6, mocap_frequency)
RGLOVE_acc = diff_vec3(RGLOVE_speed, 1/mocap_frequency)
RGLOVE_acc = LP_filter(RGLOVE_acc, 6, mocap_frequency)
RGLOVE_speed_norm = np.linalg.norm(RGLOVE_speed,axis=0)
RGLOVE_acc_norm = np.linalg.norm(RGLOVE_acc,axis=0)

# To verify if peaks were correctly detected
def plot_peaks(serie,peaks):
    fig,ax=plt.subplots()
    ax.plot(serie)
    ax.scatter(peaks, [serie[k] for k in peaks], color='blue', marker='x', label='peak')
    ax.set_title("Verification of the peaks detection in the speed norm")
    ax.grid()
    ax.set_xlabel("Time [Frames]")
    ax.set_ylabel("Speed [m/s]")
    ax.legend()
    return

peaks=find_peaks_in_speed(RGLOVE_speed_norm)
plot_peaks(RGLOVE_speed_norm,peaks) # Checking if the peaks were correctly detected
plt.show()

# Create a numpy array of shape (len(peaks),3 (xyz coordinates),180 (number of frames))
# Represent the motion of each punch for around 1 sec
def create_df_pos_punchs(serie,peaks,plot=False):
    li=[] # List
    # Itterate through the list of the peaks     
    for peak in peaks :
        serie_punch = serie[:,peak-60:peak+120] # 0.3 sec before to 0.6 sec after the peak speed of a hit
        li.append(serie_punch)
    df=np.array(li)
    return df

pos_df = create_df_pos_punchs(RGLOVE, peaks)
pos_df_mean = np.mean(pos_df,axis=0) # Create the motion of a mean punch

# Plot a particular marker for a number of frames
def plot_marker(marker, ax, frame =slice(None), **kwargs):
    x = marker[0,frame]
    x = x-x[0]
    y = marker[1,frame]
    y = y-y[0]
    z = marker[2,frame]
    z = z-z[0]
    ax.scatter(x, y, z, **kwargs)
        
# Create a static plot of all the punchs that were thrown with the mean
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Mean 3D punch trajectory visualisation')
for i in range(pos_df.shape[0]):
    plot_marker(pos_df[i,:,:], ax,alpha=0.2,s=2,label=f"Hit n° {i+1}") # color="grey"
plot_marker(pos_df_mean, ax, label="Mean",color="r")
fig.legend()
plt.show()

# Create an animation of the mean punch
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
def update2(frame):
    ax2.clear()
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_zlabel('Z [m]')
    ax2.set_title('Mean 3D punch trajectory animation')
    plot_marker(pos_df_mean, ax2,frame=slice(0,1+frame), label="Mean",color="r")
    plt.legend()

# Create the animation
ani2 = FuncAnimation(fig2, update2, frames=frames, interval=1/mocap_frequency) # To animate
plt.show()

###############################################################################
#########################CALCULATING ANGLES####################################
###############################################################################

# Inverse of a matrix accross time
def inverse_matrix(matrix):
    inverse = np.empty_like(matrix)
    for i in range(matrix.shape[2]):
        inverse[:, :, i] = np.linalg.inv(matrix[:, :, i])
    return inverse

# Rotation matrix between Pelvis and thigh
oRrp_inv = inverse_matrix(oRrp)
pRrt = np.empty_like(oRrp)
for i in range(oRrp.shape[2]):
    pRrt[:,:,i] = oRrp_inv[:,:,i] @ Rrt[:,:,i]

# Calculating euler angles from the resulting rotation matrix across time
angles_euler = np.zeros((3,oRrp.shape[2]))
for i in range (pRrt.shape[2]):
    pRrt_rot = Rotation.from_matrix(pRrt[:,:,i])
    angles_euler[:,i] = pRrt_rot.as_euler('zyx', degrees=True)

# Plotting the hip angles
x = reconstruct_time_vector(frames, mocap_frequency)
plt.figure()
plt.title("Euler Angles of the hip accross time")
plt.plot(x,angles_euler[0,:],label="x")
plt.plot(x,angles_euler[1,:],label="y")
plt.plot(x,angles_euler[2,:],label="z")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Angle [°]")
plt.grid()
plt.show()