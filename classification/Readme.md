# Classification code

Split into two sections

 - XGboost for interacting vs non-interacting
 - Resnet-18 for pairing vs stacking classification

## Training stage

XGboost (CPU is fine)
 - `python xgb_data.py`
 - `python xgb_train.py`

Resnet-18 (gpu based)
 - `python res_data.py`
 - `python res_train.py`

## Inference stage

both of these are currently minimal implementations, but basically small snippets, easy to cut and paste into the main ui

 - `python xgb_inference.py 1a1t.cif`
 - `python res_inference.py "output_images_projected3D_128x3_final_bonds/stack/1c2w_A37_A38.png"`
