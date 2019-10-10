import numpy as np
from Vicon import Markers
from Utilities import Mean_filter




def leastsq_method(markers, offset=0):

    axises = []
    centers = []
    shank_markers = markers.get_rigid_body("ben:RightShank")[0:1000]
    T_Th = markers.get_frame("ben:RightThigh")
    adjusted = Markers.transform_markers(np.linalg.inv(T_Th), shank_markers)
    shank_markers = markers.get_rigid_body("ben:RightShank")[0:]
    thigh_markers = markers.get_rigid_body("ben:RightThigh")[0:]
    cor_filter = Mean_filter.Mean_Filter(20)
    aor_filter = Mean_filter.Mean_Filter(20)

    for frame in xrange(offset, len(adjusted[0]) - offset):

        m1 = adjusted[0][frame:frame + 2]
        m2 = adjusted[1][frame:frame + 2]
        m3 = adjusted[2][frame:frame + 2]
        m4 = adjusted[3][frame:frame + 2]
        data = [m1, m2, m3, m4]

        core = cor_filter.update(Markers.calc_CoR(data))
        axis = aor_filter.update(Markers.calc_AoR(data))

        thigh = Markers.calc_mass_vect([thigh_markers[0][frame],
                                        thigh_markers[1][frame],
                                        thigh_markers[2][frame],
                                        thigh_markers[3][frame]])

        shank = Markers.calc_mass_vect([shank_markers[0][frame],
                                        shank_markers[1][frame],
                                        shank_markers[2][frame],
                                        shank_markers[3][frame]])

        sol = Markers.minimize_center([thigh, shank], axis=axis, initial=(core[0][0], core[1][0], core[2][0]))

        centers.append(sol.x)
        axises.append(axis)

    return centers, axises


def rotation_method(markers,offset=10):

    shank_markers = markers.get_rigid_body("ben:RightShank")[0:]
    T_Th = markers.get_frame("ben:RightThigh")
    adjusted = Markers.transform_markers(np.linalg.inv(T_Th), shank_markers)

    shank_markers = markers.get_rigid_body("ben:RightShank")[0:]
    thigh_markers = markers.get_rigid_body("ben:RightThigh")[0:]

    T_Th = markers.get_frame("ben:RightThigh")[0:]
    T_Sh = markers.get_frame("ben:RightShank")[0:]

    centers = []
    axises = []

    for frame in xrange(offset, len(adjusted[0]) - offset):

        T_TH_SH_1 = np.dot(np.linalg.pinv(T_Th[frame]), T_Sh[frame])  # Markers.get_all_transformation_to_base(T_Th, T_Sh)[frame]
        T_TH_SH_2 = np.dot(np.linalg.pinv(T_Th[frame + offset]), T_Sh[frame + offset])
        R1 = T_TH_SH_1[:3, :3]
        R2 = T_TH_SH_2[:3, :3]
        R1_2 = np.dot(np.transpose(R2), R1)

        rp_1 = Markers.calc_mass_vect(
            [shank_markers[0][frame], shank_markers[1][frame], shank_markers[2][frame], shank_markers[3][frame]])
        rp_2 = Markers.calc_mass_vect(
            [shank_markers[0][frame + offset], shank_markers[1][frame + offset], shank_markers[2][frame + offset],
             shank_markers[3][frame + offset]])

        rd_1 = Markers.calc_mass_vect(
            [thigh_markers[0][frame], thigh_markers[1][frame], thigh_markers[2][frame], thigh_markers[3][frame]])
        rd_2 = Markers.calc_mass_vect(
            [thigh_markers[0][frame + offset], thigh_markers[1][frame + offset], thigh_markers[2][frame + offset],
             thigh_markers[3][frame + offset]])

        rdp1 = np.dot(T_Sh[frame][:3, :3], rd_1 - rp_1)
        rdp2 = np.dot(T_Sh[frame + offset][:3, :3], rd_2 - rp_2)

        P = np.eye(3) - R1_2
        Q = rdp2 - np.dot(R1_2, rdp1)

        rc = np.dot(np.linalg.pinv(P), Q)

        thigh = Markers.calc_mass_vect([thigh_markers[0][frame],
                                        thigh_markers[1][frame],
                                        thigh_markers[2][frame],
                                        thigh_markers[3][frame]])

        shank = Markers.calc_mass_vect([shank_markers[0][frame],
                                        shank_markers[1][frame],
                                        shank_markers[2][frame],
                                        shank_markers[3][frame]])

        axis, angle = Markers.R_to_axis_angle(T_TH_SH_1[0:3, 0:3])
        Rc = rp_1 + np.dot(np.transpose(T_Sh[frame][:3, :3]), rc)

        sol = Markers.minimize_center([thigh, shank], axis=axis, initial=(Rc[0], Rc[1], Rc[2]))
        centers.append(sol.x)
        axises.append(axis)

    return centers, axises

def sphere_method(markers, offset=10):

    shank_markers = markers.get_rigid_body("ben:RightShank")[0:]
    T_Th = markers.get_frame("ben:RightThigh")
    adjusted = Markers.transform_markers(T_Th, shank_markers)

    centers = []
    axises = []
    CoM = []
    CoM_fixed = []

    # Get all the mass centers of the frame
    for ii in xrange(len(adjusted[0])):
        temp = Markers.calc_mass_vect([adjusted[0][ii],
                                       adjusted[1][ii],
                                       adjusted[2][ii],
                                       adjusted[3][ii]])
        CoM.append(np.asarray(temp))

    fixed = CoM[0]
    CoM_fixed.append(fixed)
    frame_index = []
    thresh = 0.0
    for ii, center in enumerate(CoM):
        dist = np.sqrt(np.sum(np.power(fixed-center,2)))
        if dist >= thresh:
            frame_index.append(ii)
            fixed = center
            CoM_fixed.append(fixed)

    for ii in xrange(len(CoM_fixed)-5):
        raduis, C = Markers.sphereFit(CoM_fixed[ii:ii+5])
        C = np.row_stack((C,[1]))
        C = np.dot(np.linalg.pinv(T_Th[frame_index[ii]]),C)
        centers.append(C[0:3])
    print frame_index
    return centers, axises, frame_index