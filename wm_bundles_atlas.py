#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 20th 17:28:17 2021

@author: yaleman
"""
"""
% Syntax :
% 
% Description
%
%
% See also: 
%__________________________________________________
% Authors: Yasser Aleman Gomez
% Radiology Department
% CHUV, Lausanne
% Created on %(date)s
% Version $1.0

%% ======================= Importing Libraries ============================== %
"""
import numpy as np
import nibabel as nib
import h5py
import time
import os
from pathlib import Path
import argparse
import sys
import csv

class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def _build_args_parser():
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=52)

    from argparse import ArgumentParser

    p = argparse.ArgumentParser(formatter_class=SmartFormatter, description='\n Help \n')

    requiredNamed = p.add_argument_group('Required arguments')
    requiredNamed.add_argument('--method', '-m', action='store', choices=['connectivity', 'consfile'], required=True,
                metavar='METHOD', type=str, nargs=1, default='connectivity',
                help="R|Method selection (connectivity or consfile). Another file should be supplied if connectivity is \n"
                     "                                             selected as method. This file should be supplied \n"
                     "                                             through the flags --map -p. \n"
                     "\n"
                     "   connectivity: Compute the mean value of a supplied scalar map along all the bundles included in the \n"
                     "                 atlas. It outputs a connectivity matrix where the connection strength is the mean \n"
                     "                 value of the scalar map along each bundle. \n"
                     " \n"
                     "    consfile   : Extract and save some especific bundles. The selected bundles should be supplied \n"
                     "                 through a nifti-1 file or a csv text file using the flags --bfile -bf. If a scalar \n"
                     "                 map is supplied, a table with the mean values along these bundles is stored as well.\n"
                     "                 - If the file is a Nifti image the bundles intercepting its non-zero values will \n"
                     "                 be extracted and saved. \n"
                     "                 - If the file is a readable text file, each row of the txt file should contain \n"
                     "                 the source and target regions separated by a comma (ie. 10,57). \n"
                     "\n")
    requiredNamed.add_argument('--out', '-o', action='store', required=True, metavar='OUT', type=str, nargs=1,
                help= "R| Output basename. \n"
                      "                    If connectivity was selected as method then the output filename will be: \n"
                      "                    <outbasename>-connmat-scaleId' \n"
                      "\n"
                      "                    Otherwise, it will be: \n"
                      "                    <outbasename>-<sourceId_targetId>-scaleId')\n")
    requiredNamed.add_argument('--scale', '-s', action='store', required=False, choices=['scale1', 'scale2', 'scale3', 'scale4'],
               metavar='SCALE', type=str, nargs=1,
               help= "R| Scale Id (scale1, scale2, scale3 or scale4).\n", default=['scale1'])
    requiredNamed.add_argument('--bfile', '-bf', action='store', required=False, metavar='FILE', type=str, nargs=1,
                help="R| It can be either a binary mask or a Comma Separated text file.\n"
                     "     - If this file is a Nifti image the bundles intercepting its non-zero values will be \n"
                     "       extracted and saved \n"
                     "     - If the flie is a text file, each row of the txt file should contain the source and target \n"
                     "       regions separated by a comma (ie. 10,57). The first row of this file will be ignored. \n"
                     "\n", default='None')
    requiredNamed.add_argument('--map', '-p', action='store', required=False,
                metavar='MAP', type=str, nargs=1,
                help="R| Scalar map to compute its mean value along the selected bundles.\n"
                    "\n" , default='None')
    requiredNamed.add_argument('--subth', '-st', action='store', required=False,
                               metavar='SUBTHRESH', type=float, nargs=1,
                               help="R| Subject-level consistency threshold in percentage (0-100). Bundles appearing in \n"
                                    " a percentage of subjects lower than the selected threshold will not be taken into \n"
                                    " account (default = 0). \n"
                                    "\n", default=[0])
    requiredNamed.add_argument('--voxth', '-vt', action='store', required=False,
                               metavar='VOXTHRESH', type=float, nargs=1,
                               help="R| Voxel-level consistency threshold. Voxels with probability (0-1) values lower \n"
                                    " than the supplied threshold will be set to 0 in the selected bundles (default = 0).\n"
                                    "\n", default=[0])
    requiredNamed.add_argument('--extract', '-e', action='store_true', required=False,
                               help="R| Save individual bundles in Nifti-1 format. \n"
                                    "\n")
    requiredNamed.add_argument('--all', '-a', action='store_true', required=False,
                               help="R| Save individual bundles intercepting any of the regions of interest inside the mask\n"
                                    "\n")
    requiredNamed.add_argument('--collapse', '-c', action='store_true', required=False,
                               help="R| Collapse the bundles into a 4D color-coded Nifti-1 file. \n"
                                    "\n")
    requiredNamed.add_argument('--force', '-f', action='store_true', required=False,
                               help="R| Overwrite the results. \n"
                                    "\n")
    p.add_argument('--verbose', '-v', action='store', required=False,
                type=int, nargs=1,
                help='verbosity level: 1=low; 2=debug')

    args   = p.parse_args()
    method = args.method[0]
    map    = args.map[0]
    file   = args.bfile[0]

    # boolmap = bool(map) + bool(event_id) + bool(broker_id) + bool(product_id) + bool(event_keyword) + bool(metadata)
    if method == 'connectivity' and not os.path.isfile(map):
        print("\n")
        print("Please, supply a valid scalar map using the flags --map or -p.")
        p.print_help()
        sys.exit()
    elif method != 'connectivity' and not os.path.isfile(file):
        print(" ")
        print(" ")
        print("Please, supply a valid txt or binary mask to select the bundles. Use the flags --bfile or -f.")
        print(" ")
        print(" ")

        p.print_help()
        sys.exit()

    return p

def save_matrix(connMat, matFilename, lnames, regCodes, regCoords, regColors):
    """
    Save the connectivity matrix to a HDF5 file.
    @params:
        connMat        - Required  : Numpy array with the connectivity matrix (Float)
        matFilename    - Required  : Output filename for the connectivity matrix (Str)
        lnames         - Required  : Region names (Str)
        regCodes       - Required  : Region codes (position in the matrix) (Int)
        regCoords      - Required  : Coordinates of the regions center-of mass in mm (MNI space)(Float)
        regColors      - Required  : Region RGB (red, green and blue) colors triplets. It can be used for networks visualization.(Float)
    """

    # import networkx as nx
    import h5py
    # G = nx.from_numpy_matrix(connMat)
    # nx.write_gpickle(G, matFilename)

    indnan = np.isnan(connMat)
    connMat[indnan] = 0

    hm = h5py.File(matFilename, 'w')
    # Saving the Connectivity Matrix
    g1 = hm.create_group('connmat')
    g1.create_dataset('matrix', data=connMat)
    g1.create_dataset('gmregions', data=np.array(lnames, dtype='S'))
    g1.create_dataset('gmcodes', data=regCodes)
    g1.create_dataset('gmcoords', data=regCoords)
    g1.create_dataset('gmcolors', data=regColors)
    hm.close()

    return matFilename

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def bundles_from_mask(hfName, maskFilename, outBasename, scaleId, mapFilename, voxth=0, subth=0, boolall=False, boolbund=False, boolcoll=False):
    """
    Compute mean, median and a standard deviation of a certain scalar map along the bundles reaching specified ROIs
    @params:
        hfName            - Required  : h5 filename (Str)
        maskFilename      - Required  : Nifti-1 filename containing the ROIs. Each ROI should have a different value (Str)
        outBasename       - Required  : Output basename (Str)
        scaleId           - Required  : ScaleId: scale1, scale2, scale3 or scale4  (Str)
        mapFilename       - Required  : Scalar map filename (Int)
        voxth             - Optional  : Voxel-wise consistency threshold (Float)
        subth             - Optional  : Subject-wise consistency threshold (Float)
        boolall           - Optional  : Boolean variable to save all the bundles reaching any of the selected ROIs(Bool)
        boolbund          - Optional  : Boolean variable to save the selected bundles (Bool)
        boolcoll          - Optional  : Boolean variable to collapse the selected bundles (Bool)
    """

    outDir = os.path.dirname(outBasename)
    fname  = os.path.basename(outBasename)

    # Creating the folder if it does not exist
    if not os.path.isdir(outDir):
        path = Path(outDir)
        try:
            path.mkdir()
        except OSError:
            print("Failed to make nested directory")
        else:
            print("Nested directory made")

    # Reading the mask image
    maskI = nib.load(maskFilename)
    dataP = maskI.get_fdata()

    if boolall:
        dataP[dataP != 0] = 1

    nrois_mask = np.unique(dataP)
    nrois_mask = nrois_mask[nrois_mask != 0]

    # Reading the HDF5 file
    hf        = h5py.File(hfName, 'r')
    tempVar   = hf.get('header/affine')
    affine    = np.array(tempVar)
    tempVar   = hf.get('header/dim')
    dim       = np.array(tempVar)
    tempVar   = hf.get('header/gmregions')
    stnames   = np.array(tempVar)
    tempVar   = hf.get('header/nsubjects')
    nsubj     = np.array(tempVar)
    tempVar   = hf.get('header/gmcolors')
    gmcolors  = np.array(tempVar)

    # Reading the consistency matrix. It can be used for thresholding
    tempVar   = hf.get('matrices/consistency')
    consistM  = np.array(tempVar)/nsubj

    # tempVar = hf.get('header/scale')
    # scaleId = np.array(tempVar)
    if boolcoll:
        collMask  = np.zeros((dim[0], dim[1], dim[2]), dtype='float')  # Collapsing all the bundles
        collMaskc = np.zeros((dim[0], dim[1], dim[2], 3), dtype='float')



    # Bundles to be used
    bundloc   = np.argwhere(np.triu(consistM, 1) >= subth/100)
    X         = bundloc[:, 0] + 1
    Y         = bundloc[:, 1] + 1


    # Reading the scalar map
    booltab     = False
    if os.path.isfile(mapFilename):
        mapI    = nib.load(mapFilename)
        dataM   = mapI.get_fdata()
        booltab = True

    tabList   = [['BundleId', 'SourceROI', 'TargetROI', 'Consistency' , 'MeanValue', 'StdValue', 'MedianValue']]
    tic       = time.perf_counter()

    # Reading the bundles keys
    # tempVar       = hf.get('atlas')
    # bundIds       = list(tempVar.keys())
    # tbund         = len(bundIds)
    tbund     = len(X)
    # Loop along the bundles
    for i in np.arange(0, tbund - 1):
        # tempVar = hf.get("atlas/" + bundIds[i])
        tempS      = str(X[i]) + "_" + str(Y[i])
        tempVar    = hf.get("atlas/" + tempS)
        voxIds     = np.array(tempVar)

        bdcolor    = np.mean(gmcolors[bundloc[i, :],:],axis=0)/255

        # Creating the empty volume
        array_data = np.zeros((dim[0], dim[1], dim[2]), dtype=float)

        # Assigning the probability
        array_data[voxIds[:, 0], voxIds[:, 1], voxIds[:, 2]] = voxIds[:, 3] / nsubj

        # Thresholding using the concurrency value
        if voxth > 0:
            array_data[array_data < voxth] = 0

        # Interception between the mask and the probabilistic atlas
        tempM      = np.multiply(array_data != 0, dataP)

        nrois_bund = np.unique(tempM)
        nrois_bund = nrois_bund[nrois_bund != 0]

        savebund = False
        if len(nrois_mask) == 1 and len(nrois_bund) == 1:
            savebund = True
        elif len(nrois_mask) != 1 and len(nrois_bund) > 1:
            savebund = True

        if savebund:
            if boolcoll:
                collMask            = collMask  + np.asarray(array_data >0)
                collMaskc[:,:,:,0]  = collMaskc[:,:,:,0] + array_data*bdcolor[0]
                collMaskc[:,:,:,1]  = collMaskc[:,:,:,1] + array_data*bdcolor[1]
                collMaskc[:,:,:,2]  = collMaskc[:,:,:,2] + array_data*bdcolor[2]

            # Saving the bundle image
            if boolbund:
                bundFilename   = os.path.join(outDir, fname + "-fromMask-" + tempS + "-" + scaleId + ".nii.gz")
                array_img      = nib.Nifti1Image(array_data, affine)
                nib.save(array_img, bundFilename)

            # Creating the table
            if booltab:
                # tempS = bundIds[i]
                # roiIds = tempS.split('_')

                mVal    = np.mean(dataM[array_data >= voxth])
                sVal    = np.std(dataM[array_data >= voxth])
                dVal    = np.median(dataM[array_data >= voxth])
                sourROI = stnames[int(X[i])-1].decode('UTF-8')
                targROI = stnames[int(Y[i])-1].decode('UTF-8')
                cVal    = consistM[X[i]-1, Y[i]-1]*100
                tabList.append([tempS, sourROI, targROI, cVal, mVal, sVal, dVal])

        printProgressBar(i + 1, tbund , 'Processing bundle: ' + '(' + str(i+1) + '/' + str(tbund) + ')')

    # Saving the table
    if booltab:
        tabFilename     = outBasename + "-table-fromMask-" + scaleId + ".csv"
        with open(tabFilename, mode='w') as bundTab:
            bund = csv.writer(bundTab, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in np.arange(len(tabList)):
                bund.writerow(tabList[i])
    # Saving the collapsed image
    if boolcoll:
        collFilename           = os.path.join(outDir, fname + "-fromMask-bundcollap-" + scaleId + ".nii.gz")
        # collMask        = collMask != 0
        # array_img              = nib.Nifti1Image(collMask.astype(int), affine)
        # nib.save(array_img, collFilename)
        collMaskc[:, :, :, 0] = np.divide(collMaskc[:, :, :, 0], collMask, where=True)
        collMaskc[:, :, :, 1] = np.divide(collMaskc[:, :, :, 1], collMask, where=True)
        collMaskc[:, :, :, 2] = np.divide(collMaskc[:, :, :, 2], collMask, where=True)
        collMaskc[np.isnan(collMaskc)] = 0

        imgcoll               = nib.Nifti1Image(collMaskc, affine)
        nib.save(imgcoll, collFilename)

    # Timer
    toc                 = time.perf_counter()
    print(f"Processed in {toc - tic:0.4f} seconds")

def bundles_from_textfile(hfName, textFilename, outBasename, scaleId, mapFilename, voxth=0, subth=0, boolbund=False, boolcoll=False):
    """
    Compute mean, median and a standard deviation of a certain scalar map along the bundles specified in a csv
    file.
    @params:
        hfName            - Required  : h5 filename (Str)
        textFilename      - Required  : Comma Separated Value file with the bundles of interest (i.e. 10_57, bundle between regions 10 and 57) (Str)
        outBasename       - Required  : Output basename (Str)
        scaleId           - Required  : ScaleId: scale1, scale2, scale3 or scale4  (Str)
        mapFilename       - Required  : Scalar map filename (Int)
        voxth             - Optional  : Voxel-wise consistency threshold (range: 0-1)(Float)
        subth             - Optional  : Subject-wise consistency threshold in precentage (range: 0-100)(Float)
        boolbund          - Optional  : Boolean variable to save the selected bundles (Bool)
        boolcoll          - Optional  : Boolean variable to collapse the selected bundles (Bool)
    """

    outDir = os.path.dirname(outBasename)
    fname = os.path.basename(outBasename)

    # Creating the folder if it does not exist
    if not os.path.isdir(outDir):
        path = Path(outDir)
        try:
            path.mkdir()
        except OSError:
            print("Failed to make nested directory")
        else:
            print("Nested directory made")

    # Reading the txt file
    bundIds = np.loadtxt(textFilename, skiprows=1, delimiter=',', dtype='u4', ndmin=2)
    bundIds = np.sort(bundIds, axis=1)

    # Reading the HDF5 file
    hf = h5py.File(hfName, 'r')
    tempVar = hf.get('header/affine')
    affine = np.array(tempVar)
    tempVar = hf.get('header/dim')
    dim = np.array(tempVar)
    tempVar = hf.get('header/gmregions')
    stnames = np.array(tempVar)
    tempVar = hf.get('header/nsubjects')
    nsubj = np.array(tempVar)
    tempVar   = hf.get('header/gmcolors')
    gmcolors  = np.array(tempVar)

    # Reading the consistency matrix. It can be used for thresholding
    tempVar  = hf.get('matrices/consistency')
    consistM = np.array(tempVar) / nsubj

    if boolcoll:
        collMask  = np.zeros((dim[0], dim[1], dim[2]), dtype='float')  # Collapsing all the bundles
        collMaskc = np.zeros((dim[0], dim[1], dim[2], 3), dtype='float')


    # If the csv file contains only one column then all the bundles involving the selected regions are extracted
    if np.shape(bundIds)[1] == 1:
        nstr     = len(consistM)
        sbundIds = np.zeros([1, 2], dtype='u4')
        for i in np.arange(0, len(bundIds)):
            a1 = np.ones([nstr, 1], dtype='u4') * bundIds[i]
            a2 = np.linspace(1, nstr, nstr, dtype='u4')
            a2 = a2[:, None]
            out_mat = np.concatenate((a1, a2), axis=1)
            sbundIds = np.concatenate((sbundIds, out_mat), axis=0)
        sbundIds = np.sort(sbundIds, axis=1)
        t = sbundIds[:, 0] - sbundIds[:, 1] != 0
        bundIds = sbundIds[t, :]

    # Bundles to be used
    bundloc  = np.argwhere(np.triu(consistM, 1) >= subth / 100)
    X        = bundloc[:, 0] + 1
    Y        = bundloc[:, 1] + 1


    # Reading the scalar map
    booltab = False
    if os.path.isfile(mapFilename):
        mapI = nib.load(mapFilename)
        dataM = mapI.get_fdata()
        booltab = True

    # Loop along the bundles
    # p = ProgressBar(len(bundIds))
    tabList   = [['BundleId', 'SourceROI', 'TargetROI', 'Consistency' ,'MeanValue', 'StdValue', 'MedianValue']]
    tic = time.perf_counter()
    tbund = len(bundIds)

    for i in np.arange(tbund):
        t = gmcolors[bundIds[i, :]-1, :]
        bdcolor = np.mean(t, axis=0) / 255

        # Reading the selected bundles
        tempVar = hf.get("atlas/" + str(bundIds[i, 0]) + "_" + str(bundIds[i, 1]))

        if tempVar != None:
            voxIds = np.array(tempVar)

            # Creating the empty volume
            array_data = np.zeros((dim[0], dim[1], dim[2]), dtype=float)

            # Assigning the probability
            array_data[voxIds[:, 0], voxIds[:, 1], voxIds[:, 2]] = voxIds[:, 3] / nsubj

            # Thresholding using the currency value
            if voxth > 0:
                array_data[array_data < voxth] = 0

            # Writing the nifti file
            if np.any(array_data):
                collMask            = collMask  + np.asarray(array_data >0)
                collMaskc[:,:,:,0]  = collMaskc[:,:,:,0] + array_data*bdcolor[0]
                collMaskc[:,:,:,1]  = collMaskc[:,:,:,1] + array_data*bdcolor[1]
                collMaskc[:,:,:,2]  = collMaskc[:,:,:,2] + array_data*bdcolor[2]
                if boolbund:
                    # Saving only the bundles intercepting the mask
                    bundFilename = os.path.join(outDir, fname + "-fromCSV-" + str(bundIds[i, 0]) + "_" + str(bundIds[i, 1]) + "-" + scaleId + ".nii.gz")
                    array_img = nib.Nifti1Image(array_data, affine)
                    nib.save(array_img, bundFilename)

                # Creating the table
                if booltab:
                    mVal      = np.mean(dataM[array_data >= voxth])
                    sVal      = np.std(dataM[array_data >= voxth])
                    dVal      = np.median(dataM[array_data >= voxth])
                    sourROI   = stnames[bundIds[i, 0]-1].decode('UTF-8')
                    targROI   = stnames[bundIds[i, 1]-1].decode('UTF-8')
                    cVal = consistM[bundIds[i, 0] - 1, bundIds[i, 1]  - 1] * 100
                    tabList.append([str(bundIds[i, 0]) + "_" + str(bundIds[i, 1]), sourROI, targROI, cVal, mVal, sVal, dVal])

            printProgressBar(i+1, tbund, 'Processing bundle: ' + '(' + str(i + 1) + '/' + str(tbund) + ')')

    # Saving the table
    if booltab:
        tabFilename = outBasename + "-table-fromCSV-" + scaleId + ".csv"
        with open(tabFilename, mode='w') as bundTab:
            bund = csv.writer(bundTab, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in np.arange(len(tabList)):
                bund.writerow(tabList[i])

    # Saving the collapsed image
    if boolcoll:
        collFilename = os.path.join(outDir, fname + "-fromCSV-bundcollap-" + scaleId + ".nii.gz")
        collMaskc[:, :, :, 0] = np.divide(collMaskc[:, :, :, 0], collMask, where=True)
        collMaskc[:, :, :, 1] = np.divide(collMaskc[:, :, :, 1], collMask, where=True)
        collMaskc[:, :, :, 2] = np.divide(collMaskc[:, :, :, 2], collMask, where=True)
        collMaskc[np.isnan(collMaskc)] = 0

        imgcoll               = nib.Nifti1Image(collMaskc, affine)
        nib.save(imgcoll, collFilename)

    # Timer
    toc = time.perf_counter()
    print(f"Processed in {toc - tic:0.4f} seconds")

def bundles_to_connectivity(hfName, outBasename, scaleId, mapFilename, voxth=0, subth=0, boolbund=False, boolforce=False):
    """
     Compute connectivity matrix for a specified scale where the streng of connection is mean, median and standard deviation of a certain scalar map .
     @params:
         hfName            - Required  : h5 filename (Str)
         textFilename      - Required  : Comma Separated Value file with the bundles of interest (i.e. 10_57, bundle between regions 10 and 57) (Str)
         outBasename       - Required  : Output basename (Str)
         scaleId           - Required  : ScaleId: scale1, scale2, scale3 or scale4  (Str)
         mapFilename       - Required  : Scalar map filename (Int)
         voxth             - Optional  : Voxel-wise consistency threshold (range: 0-1)(Float)
         subth             - Optional  : Subject-wise consistency threshold in precentage (range: 0-100)(Float)
         boolbund          - Optional  : Boolean variable to save the selected bundles (Bool)
         boolforce         - Optional  : Rewrite the connectivity matrices (Bool)
     """
    outDir = os.path.dirname(outBasename)
    fname = os.path.basename(outBasename)

    # Creating the folder if it does not exist
    if not os.path.isdir(outDir):
        path = Path(outDir)
        try:
            path.mkdir()
        except OSError:
            print("Failed to make nested directory")
        else:
            print("Nested directory made")

    outBasename + "-connmat-mean-" + scaleId + ".h5"


    if os.path.isfile(outBasename + "-connmat-mean-" + scaleId + ".h5") and boolforce == True:
        print(" The file " + outBasename + "-connmat-mean-" + scaleId + ".h5" + " will be overwritten.")
        os.remove(outBasename + "-connmat-mean-" + scaleId + ".h5")
        print("  ")
    if os.path.isfile(outBasename + "-connmat-std-" + scaleId + ".h5") and boolforce == True:
        print(" The file " + outBasename + "-connmat-std-" + scaleId + ".h5" + " will be overwritten.")
        os.remove(outBasename + "-connmat-std-" + scaleId + ".h5")
        print("  ")
    if os.path.isfile(outBasename + "-connmat-median-" + scaleId + ".h5") and boolforce == True:
        print(" The file " + outBasename + "-connmat-median-" + scaleId + ".h5" + " will be overwritten.")
        os.remove(outBasename + "-connmat-median-" + scaleId + ".h5")
        print("  ")

    # Reading the HDF5 file
    hf        = h5py.File(hfName, 'r')
    tempVar   = hf.get('header/affine')
    affine    = np.array(tempVar)
    tempVar   = hf.get('header/dim')
    dim       = np.array(tempVar)

    # Reading the fields to plot the network
    tempVar   = hf.get('header/gmregions')
    stnames   = np.array(tempVar)
    tempVar = hf.get('header/gmcodes')
    regCodes = np.array(tempVar)
    tempVar = hf.get('header/gmcoords')
    regCoords = np.array(tempVar)
    tempVar = hf.get('header/gmcolors')
    regColors = np.array(tempVar)

    # Reading the number of subjects used to create the atlas
    tempVar   = hf.get('header/nsubjects')
    nsubj     = np.array(tempVar)

    # Reading the consistency matrix. It can be used for thresholding
    tempVar   = hf.get('matrices/consistency')
    consistM  = np.array(tempVar)/nsubj

    # Bundles to be used
    bundloc = np.argwhere(np.triu(consistM, 1) > subth/100)
    X = bundloc[:, 0] + 1
    Y = bundloc[:, 1] + 1

    # tempVar = hf.get('header/scale')
    # scaleId = np.array(tempVar)

    # Reading the scalar map
    booltab = False
    if os.path.isfile(mapFilename):
        mapI = nib.load(mapFilename)
        dataM = mapI.get_fdata()
        booltab = True

    # Creating empty connectivity matrices
    nstr        = len(stnames)
    meanMat     = np.zeros([nstr, nstr])  # Number of fibers
    stdMat      = np.zeros([nstr, nstr])  # Fiber Length
    medMat      = np.zeros([nstr, nstr])  # Number of Points. Important to compute the mean along fibers


    tic     = time.perf_counter()

    # Reading the bundles keys
    # tempVar       = hf.get('atlas')
    # bundIds       = list(tempVar.keys())
    # tbund         = len(bundIds)

    tbund = len(X)
    # Loop along the bundles
    for i in np.arange(0, tbund - 1):
        # tempS   = bundIds[i]
        tempS   = str(X[i]) + "_" + str(Y[i])
        tempVar = hf.get("atlas/" + tempS)
        voxIds  = np.array(tempVar)
        # roiIds  = tempS.split('_')

        # Creating the empty volume
        array_data = np.zeros((dim[0], dim[1], dim[2]), dtype=float)

        # Assigning the probability
        array_data[voxIds[:, 0], voxIds[:, 1], voxIds[:, 2]] = voxIds[:, 3] / nsubj

        # # Thresholding using the currency value
        # if voxth > 0:
        #     array_data[array_data < voxth] = 0

        # Writing the nifti file
        if np.any(array_data):

            if boolbund:
                # Saving only the bundles intercepting the mask
                # bundFilename = os.path.join(outDir, fname + "-atlas-" + str(roiIds[0]) + "_" + str(roiIds[1]) + "-" + scaleId + ".nii.gz")
                bundFilename = os.path.join(outDir, fname + "-atlas-" + str(X[i]) + "_" + str(Y[i]) + "-" + scaleId + ".nii.gz")
                array_img = nib.Nifti1Image(array_data, affine)
                nib.save(array_img, bundFilename)

            # Creating the connectivity matrices
            # meanMat[int(roiIds[0]) - 1, int(roiIds[1]) - 1] = np.mean(dataM[array_data >= voxth])
            # meanMat[int(roiIds[1]) - 1, int(roiIds[0]) - 1] = meanMat[int(roiIds[0]) - 1, int(roiIds[1]) - 1]
            meanMat[int(X[i]) - 1, int(Y[i]) - 1] = np.mean(dataM[array_data >= voxth])
            meanMat[int(Y[i]) - 1, int(X[i]) - 1] = meanMat[int(X[i]) - 1, int(Y[i]) - 1]

            # stdMat[int(roiIds[0]) - 1, int(roiIds[1]) - 1] = np.std(dataM[array_data >= voxth])
            # stdMat[int(roiIds[1]) - 1, int(roiIds[0]) - 1] = stdMat[int(roiIds[0]) - 1, int(roiIds[1]) - 1]
            stdMat[int(X[i]) - 1, int(Y[i]) - 1] = np.std(dataM[array_data >= voxth])
            stdMat[int(Y[i]) - 1, int(X[i]) - 1] = stdMat[int(X[i]) - 1, int(Y[i]) - 1]

            # medMat[int(roiIds[0]) - 1, int(roiIds[1]) - 1] = np.median(dataM[array_data >= voxth])
            # medMat[int(roiIds[1]) - 1, int(roiIds[0]) - 1] = medMat[int(roiIds[0]) - 1, int(roiIds[1]) - 1]
            medMat[int(X[i]) - 1, int(Y[i]) - 1] = np.median(dataM[array_data >= voxth])
            medMat[int(Y[i]) - 1, int(X[i]) - 1] = medMat[int(X[i]) - 1, int(Y[i]) - 1]

        printProgressBar(i+1, tbund, 'Processing bundle: ' + '(' + str(i+1) + '/' + str(tbund) + ')')

    # Saving the connectivity matrices

    # Mean
    meanFilename = outBasename + "-connmat-mean-" + scaleId + ".h5"
    # save_matrix(meanMat, meanFilename, gmcoords, , stnames)
    save_matrix(meanMat, meanFilename, stnames, regCodes, regCoords, regColors)

    # Std
    stdFilename = outBasename + "-connmat-std-" + scaleId + ".h5"
    # save_matrix(stdMat, stdFilename)
    save_matrix(stdMat, stdFilename, stnames, regCodes, regCoords, regColors)

    # Median
    medFilename = outBasename + "-connmat-median-" + scaleId + ".h5"
    # save_matrix(medMat, medFilename)
    save_matrix(medMat, medFilename, stnames, regCodes, regCoords, regColors)

    # Timer
    toc = time.perf_counter()
    print(f"Processed in {toc - tic:0.4f} seconds")

def main():
    import sys
    # 0. Handle inputs
    parser = _build_args_parser()
    args = parser.parse_args()
        
    print(args)
    if args.verbose is not None:
        v = np.int(args.verbose[0])
    else:
        v = 0
        print('- Verbose set to 0\n')
    if v:
        print('\nInputs\n')
#

    # Getting the path of the current running python file
    cwd = os.getcwd()

    scaleid   = args.scale[0]
    file      = args.bfile[0]
    fmeth     = args.method[0]
    fout      = args.out[0]
    fmap      = args.map[0]
    voxth     = args.voxth[0]
    subth     = args.subth[0]
    boolall   =  args.all
    boolbund  = args.extract
    boolcoll  = args.collapse
    boolforce = args.force

    hfName = os.path.join(cwd,'probconnatlas', "wm.connatlas." + scaleid + ".h5")

    if fmeth == 'connectivity':
        bundles_to_connectivity(hfName, fout, scaleid, fmap, voxth, subth, boolbund, boolforce)

    elif fmeth == 'consfile':
        # fid = open(file, "r+")
        imask = False
        if file[-7:]== '.nii.gz' or file[-4:] == '.nii':
            imask = True

        if imask == True:
            bundles_from_mask(hfName, file, fout, scaleid, fmap, voxth, subth, boolall, boolbund, boolcoll)
        else:
            bundles_from_textfile(hfName, file, fout, scaleid, fmap, voxth, subth, boolbund, boolcoll)
    else:
        print("Please enter a correct method (connectivity or consfile). ")
        sys.exit()


if __name__ == "__main__":
    main()
