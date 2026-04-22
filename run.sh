#!/bin/bash

EXP_NAME="train_gpt2_prosqa"

oarsub -p "network_address='lig-gpu8.imag.fr' OR network_address='lig-gpu9.imag.fr' OR network_address='lig-gpu10.imag.fr'" \
       -l /host=1/gpu=1,walltime=20:00:00 \
       -O "runs/${EXP_NAME}.txt" \
       -E "runs/${EXP_NAME}.txt" \
       "./scripts/${EXP_NAME}.sh"