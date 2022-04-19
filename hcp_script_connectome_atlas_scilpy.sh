# this script assumes that it is executed at the root of a folder with all subject IDs of hcp_1200
# Within each folder ID, the script requires to find the dwi.nii.gz, dwi.bval, dwi.bvec, t1.nii.gz
# hence, the user must rename and organize files from the HCP download
# 
# [root-path]
# ├── 100206
# │   ├── dwi.nii.gz
# │   ├── dwi.bval
# │   ├── dwi.bvec
# │   └── t1.nii.gz
# └── 904044
#     ├── dwi.nii.gz
#     ├── dwi.bval
#     ├── dwi.bvec
#     └── t1.nii.gz
# ...

# Filenames must be exactly as above. The script will fail otherwise.

# This script was tested with the scilpy release 1.3.0 (https://github.com/scilus/scilpy)
# singularity exec -B [root-path-where-data-and-container-is] ./scilus_1.3.0.sif ./hcp_script_connectome_atlas_scilpy.sh
# http://scil.dinf.usherbrooke.ca/containers_list/scilus_1.3.0.sif 
for i in [1-9]*;
do
    echo "Processing subject ID $i"

    cd $i/
    ####################
    # preparing T1 data
    ####################
    bet t1.nii.gz t1_mask.nii.gz -m -f 0.25
    scil_run_nlmeans.py t1.nii.gz t1_nlm.nii.gz 1 --mask t1_mask_mask.nii.gz   
    cp t1_nlm.nii.gz t1_brain.nii.gz
    
    # preparing masks
    fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -g -o t1_brain.nii.gz t1_brain.nii.gz
    mv t1_brain_seg_2.nii.gz mask_wm.nii.gz
    mv t1_brain_seg_1.nii.gz mask_gm.nii.gz
    mv t1_brain_seg_0.nii.gz mask_csf.nii.gz
    mv t1_brain_pve_2.nii.gz map_wm.nii.gz
    mv t1_brain_pve_1.nii.gz map_gm.nii.gz
    mv t1_brain_pve_0.nii.gz map_csf.nii.gz
    scil_compute_maps_for_particle_filter_tracking.py  map_wm.nii.gz map_gm.nii.gz map_csf.nii.gz
    scil_image_math.py convert mask_wm.nii.gz wm.nii.gz --data_type uint8 -f
    scil_image_math.py addition wm.nii.gz interface.nii.gz  wm_interface.nii.gz

    # a bit cleanning up
    mkdir T1_analysis
    rm -rf t1_brain_pveseg.nii.gz t1_brain_mixeltype.nii.gz t1_brain_seg.nii.gz t1_mask.nii.gz
    mv t1_mask_mask.nii.gz mask_t1.nii.gz
    mv t1_nlm* t1_brain* interface.nii.gz mask* map* T1_analysis/
    mv wm* T1_analysis/
    
    ########################
    # Diffusion processing
    ########################
    scil_extract_b0.py dwi.nii.gz dwi.bval dwi.bvec b0_mean.nii.gz --mean --b0_thr 20
    bet b0_mean.nii.gz b0_mean_mask.nii.gz -m -f 0.15
    mv b0_mean_mask_mask.nii.gz mask_b0.nii.gz

    # keep low b-values for DTI
    scil_extract_dwi_shell.py dwi.nii.gz dwi.bval dwi.bvec 0 1000 dwi_dti.nii.gz dwi_dti.bval dwi_dti.bvec --tolerance 20
    
    # dti
    mkdir DTI
    cd DTI
    scil_compute_dti_metrics.py ../dwi_dti.nii.gz ../dwi_dti.bval ../dwi_dti.bvec --mask ../mask_b0.nii.gz
    # ignore warnings! This is normal
    cd ../

    # fodf
    mkdir FODF
    cd FODF
    scil_compute_ssst_frf.py ../dwi.nii.gz ../dwi.bval ../dwi.bvec frf.txt --mask ../mask_b0.nii.gz
    scil_set_response_function.py frf.txt 15,4,4 frf_fixed.txt -f
    scil_compute_ssst_fodf.py ../dwi.nii.gz ../dwi.bval ../dwi.bvec frf_fixed.txt fodf.nii.gz --mask ../mask_b0.nii.gz --processes 8 
    # ignore warnings! This is normal
    cd ../

    # cleaning up
    mkdir ExtractedShells
    mv dwi_dti.* b0_mean* mask_b0* ExtractedShells/

    #############################################################################
    # Particle Filter Tractography
    # Girard et al 2014
    # Fiber compression and filtering by length included in the tractography now
    #############################################################################
    # scil_compute_pft.py FODF/fodf.nii.gz T1_analysis/interface.nii.gz \
    #  			T1_analysis/map_include.nii.gz T1_analysis/map_exclude.nii.gz \
    #  			--npv 60 prob_pft_fodf_npv30_int_20-200_fc02.trk --min_length 20 --max_length 200 --compress 0.2 --step 0.5

    scil_compute_pft.py FODF/fodf.nii.gz T1_analysis/wm_interface.nii.gz \
     			T1_analysis/map_include.nii.gz T1_analysis/map_exclude.nii.gz \
    			--npv 10 prob_pft_fodf_npv10_wm_int_20-200_fc02.trk --min_length 20 --max_length 200 \
			--compress 0.2 --step 0.5
    
    # final output used in the Connectome Atlas
    ln -s DTI/fa.nii.gz fa.nii.gz
    ln -s prob_pft_fodf_npv10_wm_int_20-200_fc02.trk final_tracks.trk

    cd ../
done





