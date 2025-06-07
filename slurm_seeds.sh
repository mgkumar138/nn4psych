#!/bin/bash

# Define the arrays of hyperparameters
# epochs=50000
# trials=200
# maxdisp_values=(20 10 5) 
# rewardsize_values=(2 5 10)
# lr_values=(0.0 0.000001 0.00001 0.0001)
# nrnn_values=(64 128 256)
# loadmodel_values=(0)
# gamma_values=(0.95 0.99 0.9 0.8)

eval "$(cat export_pretrain_params.sh)"

# Iterate through all combinations of hyperparameters
for gamma in "${gamma_values[@]}"; do
  for maxdisp in "${maxdisp_values[@]}"; do
    for rewardsize in "${rewardsize_values[@]}"; do
      for lr in "${lr_values[@]}"; do
        for nrnn in "${nrnn_values[@]}"; do
          for loadmodel in "${loadmodel_values[@]}"; do

            job_name="gamma${gamma}_maxdisp${maxdisp}_rsz${rewardsize}_lr${lr}_nrnn${nrnn}_loadmodel${loadmodel}"

            # Write a temporary SLURM script for each combination
            cat <<EOT > temp_slurm_${job_name}.sh
#!/bin/bash
#SBATCH -J $job_name
#SBATCH -c 5
#SBATCH -t 07:00:00
#SBATCH -p seas_compute
#SBATCH --gres=gpu:0
#SBATCH --mem=5G
#SBATCH -o log_maxdisp/${job_name}_%A.%a.out
#SBATCH -e log_maxdisp/${job_name}_%A.%a.err
#SBATCH --array=0-4

eval "\$(conda shell.bash hook)"
conda activate pytorch

CMD="python -u pretrain_rnn.py --epochs $epochs --trials $trials --maxdisp $maxdisp --rewardsize $rewardsize --lr $lr --gamma $gamma --nrnn $nrnn --loadmodel $loadmodel --seed \${SLURM_ARRAY_TASK_ID}"

echo \$CMD
eval \$CMD
EOT

            # Submit the job and check for submission error
            sbatch temp_slurm_${job_name}.sh
            if [ $? -ne 0 ]; then
              echo "Error submitting job $job_name"
            else
              # Remove the temporary SLURM script
              rm temp_slurm_${job_name}.sh
            fi

          done
        done
      done
    done
  done
done