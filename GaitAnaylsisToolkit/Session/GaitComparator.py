#!/usr/bin/env python
# //==============================================================================
# /*
#     Software License Agreement (BSD License)
#     Copyright (c) 2020, AIMVicon
#     (www.aimlab.wpi.edu)

#     All rights reserved.

#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions
#     are met:

#     * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.

#     * Neither the name of authors nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#     FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#     COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#     INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#     BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#     ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#     POSSIBILITY OF SUCH DAMAGE.

#     \author    <http://www.aimlab.wpi.edu>
#     \author    <ajlewis@wpi.edu>
#     \author    Alek Lewis
#     \version   0.1
# */
# //==============================================================================

import GaitAnaylsisToolkit.Session.ViconGaitingTrial as vg
import GaitCore.Core as Core
import Vicon.Markers.Markers as Markers


def dist(A, B):
    return ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2) ** 0.5


class Comparator:
    def __init__(self, cal, dat, verbose=False):
        if verbose:
            print("Starting...")
        self.cal_file = cal
        self.cal_gait = vg.ViconGaitingTrial(cal)
        self.cal = self.cal_gait._vicon
        if verbose:
            print("Read calibration data!")

        self.dat_files = dat
        self.dat = []
        self.dat_gait = []
        for file in self.dat_files:
            g = vg.ViconGaitingTrial(file)
            self.dat_gait.append(g)
            self.dat.append(g._vicon)
            if verbose:
                print("Read data from " + file)

        self.frames = {"Root": [Core.Point.Point(0, 14, 0),
                                Core.Point.Point(56, 0, 0),
                                Core.Point.Point(14, 63, 0),
                                Core.Point.Point(56, 63, 0)], "L_Foot": [Core.Point.Point(0, 0, 0),
                                                                         Core.Point.Point(70, 0, 0),
                                                                         Core.Point.Point(28, 70, 0),
                                                                         Core.Point.Point(70, 63, 0)],
                       "L_Tibia": [Core.Point.Point(0, 0, 0),
                                   Core.Point.Point(0, 63, 0),
                                   Core.Point.Point(70, 14, 0),
                                   Core.Point.Point(35, 49, 0)], "L_Femur": [Core.Point.Point(0, 0, 0),
                                                                             Core.Point.Point(70, 0, 0),
                                                                             Core.Point.Point(0, 42, 0),
                                                                             Core.Point.Point(70, 56, 0)],
                       "R_Foot": [Core.Point.Point(0, 0, 0),
                                  Core.Point.Point(56, 0, 0),
                                  Core.Point.Point(0, 49, 0),
                                  Core.Point.Point(42, 70, 0)], "R_Tibia": [Core.Point.Point(0, 0, 0),
                                                                            Core.Point.Point(42, 0, 0),
                                                                            Core.Point.Point(7, 49, 0),
                                                                            Core.Point.Point(63, 70, 0)],
                       "R_Femur": [Core.Point.Point(7, 0, 0),
                                   Core.Point.Point(56, 0, 0),
                                   Core.Point.Point(0, 70, 0),
                                   Core.Point.Point(42, 49, 0)]}

        self.joint_to_parent_frame = {"L_Hip": "Root", "R_Hip": "Root",
                                      "L_Knee": "L_Femur", "R_Knee": "R_Femur",
                                      "L_Ankle": "L_Tibia", "R_Ankle": "R_Tibia"}
        self.joint_to_child_frame = {"L_Hip": "L_Femur", "R_Hip": "R_Femur",
                                     "L_Knee": "L_Tibia", "R_Knee": "R_Tibia",
                                     "L_Ankle": "L_Foot", "R_Ankle": "R_Foot"}

        self.cal_markers = self.cal.get_markers()
        self.cal_markers.smart_sort()
        self.cal_markers.auto_make_transform(self.frames)
        if verbose:
            print("Calculating joints...")
        self.cal_markers.calc_joints()
        if verbose:
            print("Joints calculated!")

        self.dat_markers = []
        for vi in self.dat:
            m = vi.get_markers()
            m.smart_sort()
            m.auto_make_transform(self.frames)
            self.dat_markers.append(m)

        if verbose:
            print("Mapping joints from calibration to data...")
        self.set_data_joints(verbose=verbose)
        # for m in self.dat_markers:
        #     m.calc_joints()
        if verbose:
            print("Joint locations mapped!")

    def set_data_joints(self, verbose=False):
        for i in range(len(self.dat)):
            j_rel_parent = self.cal_markers.get_joints_rel()
            j_rel_child = self.cal_markers.get_joints_rel_child()
            joints_by_parent = {}  # The joint centers, mapped from the calibration data through the parent body
            joints_by_child = {}  # The joint centers, mapped from the calibration data through the child body
            mapping_error = {}  # The mapping error for each joint

            for j in j_rel_parent:
                joints_by_parent[j] = []
                joints_by_child[j] = []
                mapping_error[j] = []
                nf = len(self.dat_markers[i].get_frame(self.joint_to_parent_frame[j]))
                for f in range(nf):  # Populate joints_by_parent, joints_by_child, and mapping_error
                    p = Core.Point.Point(j_rel_parent[j][0][0], j_rel_parent[j][1][0], j_rel_parent[j][2][0])
                    frame = self.dat_markers[i].get_frame(self.joint_to_parent_frame[j])[f]
                    p2 = Markers.local_point_to_global(frame, p)
                    joints_by_parent[j].append([p2.x, p2.y, p2.z])

                    p = Core.Point.Point(j_rel_child[j][0][0], j_rel_child[j][1][0], j_rel_child[j][2][0])
                    frame = self.dat_markers[i].get_frame(self.joint_to_child_frame[j])[f]
                    p3 = Markers.local_point_to_global(frame, p)
                    joints_by_child[j].append([p3.x, p3.y, p3.z])

                    mapping_error[j].append(dist([p2.x, p2.y, p2.z], [p3.x, p3.y, p3.z]))

            joints = {}
            for j in mapping_error:
                joints[j] = []
            has_misaligned = False  # Have we met any misaligned bodies?
            has_aligned = False
            has_ambiguous = False  # Are any bodies not conclusively aligned or misaligned?
            has_unsure = False  # Are any joints not adjacent to any definitely aligned bodies?
            has_forced_misalignment = False  # Are any joints surrounded by misaligned bodies?
            for f in range(nf):  # Using the mapping error, determine whether to use joints_by_parent or joints_by_child
                bodies = {"Root": 0, "R_Femur": 0, "L_Femur": 0, "R_Tibia": 0, "L_Tibia": 0, "R_Foot": 0,
                          "L_Foot": 0}  # Keep track of our body status
                adjacent = {
                    "R_Ankle": ("R_Tibia", "R_Foot"),
                    "R_Knee": ("R_Femur", "R_Tibia"),
                    "R_Hip": ("Root", "R_Femur"),
                    "L_Ankle": ("Root", "L_Femur"),
                    "L_Knee": ("L_Femur", "L_Tibia"),
                    "L_Hip": ("L_Tibia", "L_Foot")
                }
                for j in mapping_error:
                    err = mapping_error[j]
                    if -50 <= err[f] <= 50:  # TODO: Make this configurable
                        bodies[adjacent[j][0]] = 1
                        bodies[adjacent[j][1]] = 1

                has_aligned = False
                for b in bodies:
                    if bodies[b] == 1:
                        has_aligned = True

                if has_aligned:
                    for j in mapping_error:
                        err = mapping_error[j]
                        if not (-50 <= err[f] <= 50):  # If a joint is misaligned...
                            if bodies[adjacent[j][0]] == 1 and bodies[adjacent[j][1]] == 0:
                                # And one of the adjacent bodies is definitely aligned...
                                bodies[adjacent[j][1]] = -1  # The other body must be misaligned
                            if bodies[adjacent[j][1]] == 1 and bodies[adjacent[j][0]] == 0:
                                bodies[adjacent[j][0]] = -1

                for b in bodies:
                    if bodies[b] == 0:
                        has_ambiguous = True  # We weren't able to tell if some bodies were aligned or not!
                    if bodies[b] == -1:
                        has_misaligned = True  # At least one body is definitely misaligned!

                for j in mapping_error:
                    adj = (bodies[adjacent[j][0]], bodies[adjacent[j][1]])
                    if 0 in adj and 1 not in adj:
                        has_unsure = True  # At least one joint doesn't have any definitely aligned bodies!
                    elif -1 in adj and 0 not in adj and 1 not in adj:
                        has_forced_misalignment = True  # At least one joint is totally surrounded by misaligned bodies!

                    # Order of preference for final joints:
                        # 1. Aligned joint_by_parent
                        # 2. Aligned joint_by_child
                        # 3. Ambiguous joint_by_parent
                        # 4. Ambiguous joint_by_child
                        # 5. Definitely misaligned joint_by_parent
                    if adj[0] == 1:  # Parent frame is aligned!
                        joints[j].append(joints_by_parent[f])
                    elif adj[1] == 1:  # Child frame is aligned!
                        joints[j].append(joints_by_child[f])
                    elif adj[0] == 0:  # Parent frame is ambiguous
                        joints[j].append(joints_by_parent[f])
                    elif adj[1] == 0:  # Child frame is ambiguous
                        joints[j].append(joints_by_child[f])
                    else:  # Parent frame is definitely misaligned, but so is the child frame
                        joints[j].append(joints_by_parent)

            self.dat_markers[i].set_joints(joints)
            if verbose:
                print("I Exist")  # TODO: use flags for warnings

    def play(self, ind, joints=True):
        if ind == -1:
            self.cal_markers.play(joints=joints)
        else:
            self.dat_markers[ind].play(joints=joints, center=True)

    def calc_gait_cycles(self):
        for g in self.dat_gait:
            g.create_index_seperators()

    def get_gait_cycles(self, ind):
        return self.dat_gait[ind].vicon_set_points

    def print_gait_cycles(self):
        total = 0
        avg = 0
        for g in self.dat_gait:
            cycles = g.vicon_set_points
            total += len(cycles)
            for c in cycles:
                avg += (c[1] - c[0])
        avg /= total

        print("Across all data sources, there are a total of " + str(
            total) + " gait cycles, which have an average length of " + str(avg) + " frames.")
        print("Breakdown by individual data source:")
        for g in self.dat_gait:
            print(g._vicon._file_path + ":")
            t = len(g.vicon_set_points)
            print("\t" + str(t) + " gait cycles")
            a = 0
            for c in g.vicon_set_points:
                a += (c[1] - c[0])
            a /= t
            print("\tAverage length is " + str(a) + " frames")

    def total_cycles(self):
        total = 0
        for g in self.dat_gait:
            cycles = g.vicon_set_points
            total += len(cycles)
        return total

    def avg_cycle_len(self):
        total = 0
        avg = 0
        for g in self.dat_gait:
            cycles = g.vicon_set_points
            total += len(cycles)
            for c in cycles:
                avg += (c[1] - c[0])
        avg /= total
        return avg

    def total_cycle_ind(self, ind):
        g = self.dat_gait[ind]
        return len(g.vicon_set_points)

    def avg_cycle_len_ind(self, ind):
        g = self.dat_gait[ind]
        t = len(g.vicon_set_points)
        a = 0
        for c in g.vicon_set_points:
            a += (c[1] - c[0])
        a /= t
        return a
