#!/bin/bash
# This script performs a spatial registration between an image in native space 
# and a template using ANTs
# It then applys the resulting transformation to a scalar map to move it to the 
# template space. 
# 
# Mandatory input arguments
#        -i     : Individual image.
#        -s     : Scalar map in individual space.
#        -o     : Output basename.
#             
# Optional arguments:
#        -t     : Template image.
#
#
# 
# Author: Yasser Aleman 
# CHUV, Radiology Dept
# 15-04-2021

tempVar=$(which 'scalmap_to_template.sh')
rootDir=$(dirname $tempVar)
templDir=$rootDir/'mni_icbm152_t1_tal_nlin_asym_09c'

# 1. Default template
tempImage=$templDir/'mni_icbm152_t1_tal_nlin_asym_09c.nii.gz'

# 0. Reading the options
while getopts i:o:s:t:tm: flag
do
    case "${flag}" in
        i) indImage=${OPTARG};;
        o) outBas=${OPTARG};;
        s) indScalMap=${OPTARG};;
        t) tempImage=${OPTARG};;
    esac
done
echo "Input image: $indImage";
echo "Output basename: $outBas";
echo "Scalar map in individual space: $indScalMap";
echo "Template image: $tempImage";

# 2. Input and output for the scalar map
outDir=$(dirname $outBas);
filePref=$(basename $outBas);
outScalMap=$filePref_temp'.nii.gz'

#######  Regstration to MNI
# 3. Creating the output folder
spatTransfDir=$outDir/'ants-transf2temp'/
mkdir -p $spatTransfDir;

# 4. Estimating the spatial transformation and creating the Jacobians
defFile=$spatTransfDir/$filePref'_desc-ind2template_'
antsRegistrationSyN.sh -d 3 -f $tempImage -m $indImage -t s -o $defFile

CreateJacobianDeterminantImage 3 $defFile'1Warp.nii.gz' $defFile'desc-Warp_jacobian.nii.gz'
CreateJacobianDeterminantImage 3 $defFile'1InverseWarp.nii.gz' $defFile'desc-InverseWarp_jacobian.nii.gz'

if [ ! -f $defFile'1Warp.nii.gz' ]; then
	echo "Stage 11: Spatial registration has failed for subject " $Id >> $logsdir/$Id'.pipelineerrors.log'
fi

# 5. Applying the spatial transformation to the scalar map
antsApplyTransforms -d 3 -e 3 -i $indScalMap -o $outScalMap -r $tempImage -t $defFile'1Warp.nii.gz' -t $defFile'0GenericAffine.mat' -n Linear

