# Description:
# 	Writes 3D volumetric data to Ceetron GLview native format (.vtf).
#
# Arguments:
#	p		point cloud (nx3 matrix of n (x,y,z)-coordinates).
#	tetr	Tetrahedral elements. Index to the four corners of element i given in row i.
#	<name>	Name of result set, a descriptive string.
#	value	Values of that particular results.
#
# The interpretation of the result values are based on their size:
#	n x 1		Vectors are scalar results (ex: temperature field).
#	n x 3		Matrices are vector results (ex: displacement field).
#	n x m		Matrices are scalar time results (ex: time-dependent temperature).
#	n x 3 x m	3D-matrices are multiple vector results (ex: eigenmodes).
#
# Except for the following special <name> tags:
#	'FileName' - name of file to store results into.
#	'Time'     - scalar values of the time iteration.
#
# Returns:
#	.vtf-file for GLview.
#
#
# Examples:
# # Store a scalar field.
# Write_VTF(p,tetr,Solution=u,FileName="Solution.vtf")
#
# # Store a time-dependent scalar field.
# Write_VTF(p,tetr,Solution=ut,Time=t,FileName="Solution.vtf")
#
# # Store a vector field.
# Write_VTF(p,tetr,Solution=U,FileName="Solution.vtf")
#
# # Eigenmodes
# allU = np.zeros((n,3,3))
# allU(:,:,1) = np.hstack([ux,uy,uz])
# allU(:,:,2) = np.hstack([ux,uy,uz])
# allU(:,:,3) = np.hstack([ux,uy,uz])
# Write_VTF(p,tetr,Eigenmodes=allU,FileName="Eigenmodes.vtf")
#
# # von Mises Stress?
# Write_VTF(p,tetr,Displacement=U,Von_Mises_Stress=sigma,FileName="Eigenmodes.vtf")
#
#
# Author: Kjetil A. Johannessen, Abdullah Abdulhaque
# Last edit: 07-10-2019


import numpy as np
from datetime import datetime


def write_vtf(p, tetr, *args, **kwargs):
    FILENAME = ""
    TimeSteps = []

    if "FileName" in kwargs:
        FILENAME = kwargs["FileName"]
    if "Time" in kwargs:
        FILENAME = kwargs["Time"]
    if FILENAME == "":
        FILENAME = "Output.vtf"
    if len(TimeSteps) == 0:
        TimeSteps = np.arange(1, np.shape(p)[0] + 1)

    OUTPUT = open(FILENAME, 'w')
    OUTPUT.write("*VTF-1.00 \n\n")
    OUTPUT.write("*INTERNALSTRING 40001 \n")
    OUTPUT.write("VTF Writer Version info: \n")
    OUTPUT.write("APP_INFO: GLview Express Writer: 1.1-12 \n")
    OUTPUT.write("GLVIEW_API_VER: 2.1-22 \n")
    DATE = datetime.now().strftime("%Y %h %d %H:%M:%S")
    OUTPUT.write("EXPORT_DATE: " + DATE + "\n\n")

    OUTPUT.write("*NODES 1 \n")
    for i in range(0, np.shape(p)[0]):
        a = str(format(p[i, 0], '.15f'))
    b = str(format(p[i, 1], '.15f'))
    c = str(format(p[i, 2], '.15f'))
    OUTPUT.write(a + " " + b + " " + c + "\n")
    OUTPUT.write("\n")

    OUTPUT.write("*ELEMENTS 1 \n")
    OUTPUT.write("%NODES #1 \n")
    OUTPUT.write("%NAME \"Patch 1\" \n")
    OUTPUT.write("%NO_ID \n")
    OUTPUT.write("%MAP_NODE_INDICES \n")
    OUTPUT.write("%PART_ID \n")

    OUTPUT.write("%TETRAHEDRONS \n")
    for i in range(0, np.shape(tetr)[0]):
        a = str(format(tetr[i, 0], '.15f'))
        b = str(format(tetr[i, 1], '.15f'))
        c = str(format(tetr[i, 2], '.15f'))
        d = str(format(tetr[i, 3], '.15f'))

    OUTPUT.write(a + " " + b + " " + c + " " + d + "\n")
    OUTPUT.write("\n")

    res_id = 2
    for i in kwargs:
        if ("FileName" in kwargs) or ("Time" in kwargs):
            continue
        u = kwargs[i]
        n = np.shape(u)

        # Time vector field.
        if len(n) > 2:
            for j in range(0, n[2]):
                OUTPUT.write("*RESULTS " + str(res_id) + "\n")
                OUTPUT.write("%NO_ID \n")
                OUTPUT.write("%DIMENSION 3 \n")
                OUTPUT.write("%PER_NODE #1 \n")

                u_temp = u[:, :, j]
                for k in range(0, np.shape(u_temp)[0]):
                    a = str(format(u_temp[k, 0], '.15f'))
                b = str(format(u_temp[k, 1], '.15f'))
                c = str(format(u_temp[k, 2], '.15f'))
                OUTPUT.write(a + " " + b + " " + c + "\n")
                OUTPUT.write("\n")
                res_id += 1

        # Time scalar field.
        elif n[1] > 3:
            for j in range(0, n[1]):
                OUTPUT.write("*RESULTS " + int(res_id) + "\n")
                OUTPUT.write("%NO_ID \n")
                OUTPUT.write("%DIMENSION 1 \n")
                OUTPUT.write("%PER_NODE #1 \n")

                u_temp = u[:, j]
                for k in range(0, np.shape(u_temp)[0]):
                    a = str(format(u_temp[k, 0], '.15f'))
                OUTPUT.write(a + "\n")
                OUTPUT.write("\n")
                res_id += 1

        # Vector field.
        elif n[1] == 3:
            for j in range(0, n[1]):
                OUTPUT.write("*RESULTS " + int(res_id) + "\n")
                OUTPUT.write("%NO_ID \n")
                OUTPUT.write("%DIMENSION 3 \n")
                OUTPUT.write("%PER_NODE #1 \n")

                u_temp = u
                for k in range(0, np.shape(u_temp)[0]):
                    a = str(format(u_temp[k, 0], '.15f'))
                b = str(format(u_temp[k, 1], '.15f'))
            c = str(format(u_temp[k, 2], '.15f'))
            OUTPUT.write(a + " " + b + " " + c + "\n")
            OUTPUT.write("\n")
            res_id += 1

        # Scalar field.
        elif n[1] == 1:
            for j in range(0, n[1]):
                OUTPUT.write("*RESULTS " + int(res_id) + "\n")
                OUTPUT.write("%NO_ID \n")
                OUTPUT.write("%DIMENSION 1 \n")
                OUTPUT.write("%PER_NODE #1 \n")

                u_temp = u
                for k in range(0, np.shape(u_temp)[0]):
                    a = str(format(u_temp[k, 0], '.15f'))
                OUTPUT.write(a + "\n")
                OUTPUT.write("\n")
                res_id += 1

        # Wrong input.
        else:
            info = "Nonvalid dimensions (" + n[0] + "," + n[1] + ") of solution field " + i  # TODO wut??
            raise RuntimeError("Write_VTF", info)
            OUTPUT.close()
            return
        OUTPUT.write("\n")

    OUTPUT.write("*GLVIEWGEOMETRY 1 \n")
    OUTPUT.write("%STEP 1 \n")
    OUTPUT.write("%ELEMENTS \n")
    OUTPUT.write("1 \n\n")

    res_id = 2
    c = 1
    for i in kwargs:
        if ("FileName" in kwargs) or ("Time" in kwargs):
            continue
        u = kwargs[i]
        n = np.shape(u)

        # Time vector field.
        if len(n) > 2:
            OUTPUT.write("*GLVIEWVECTOR " + str(c) + "\n")
            OUTPUT.write("%NAME " + i + "\n")
            for j in range(0, n[2]):
                OUTPUT.write("%STEP " + str(j + 1) + "\n")
                OUTPUT.write("%STEPNAME " + str(TimeSteps[j]) + "\n")
                OUTPUT.write(str(res_id) + "\n")
                res_id += 1
            OUTPUT.write("\n")

        # Time scalar field.
        elif n[1] > 3:
            OUTPUT.write("*GLVIEWSCALAR " + str(c) + "\n")
            OUTPUT.write("%NAME " + i + "\n")
            for j in range(0, n[2]):
                OUTPUT.write("%STEP " + str(j + 1) + "\n")
                OUTPUT.write("%STEPNAME " + str(TimeSteps[j]) + "\n")
                OUTPUT.write(str(res_id) + "\n")
                res_id += 1
            OUTPUT.write("\n")

        # Vector field.
        elif n[1] == 3:
            OUTPUT.write("*GLVIEWVECTOR " + str(c) + "\n")
            OUTPUT.write("%NAME " + i + "\n")
            OUTPUT.write("%STEP 1 \n")
            OUTPUT.write(str(res_id) + "\n\n")
            res_id += 1

        # Scalar field.
        elif n[1] == 1:
            OUTPUT.write("*GLVIEWSCALAR " + str(c) + "\n")
            OUTPUT.write("%NAME " + i + "\n")
            OUTPUT.write("%STEP 1 \n")
            OUTPUT.write(str(res_id) + "\n\n")
            res_id += 1
        c += 1
    OUTPUT.close()
