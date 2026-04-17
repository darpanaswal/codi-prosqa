EXP_NAME="train_gpt2_prosqa"

oarsub -p "network_address='lig-gpu7.imag.fr'" \
       -l /host=1/gpu=2,walltime=80:00:00 \
       -O "runs/${EXP_NAME}.txt" \
       -E "runs/${EXP_NAME}.txt" \
       "./scripts/${EXP_NAME}.sh"