#!/bin/bash -l

# Read Settings file
source settings.sh

echo "Environment:"         $ENV
echo "Model:"               $MODEL
echo "L2 Regularizer:"      $L2
echo "Off-policy target:"   $OPT
echo "Learning rate:"       $LR
echo "NN size:"             $NN

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=zoo_DVRACER_${ENV}
#SBATCH --output=zoo_${ENV}_%j.out
#SBATCH --error=zoo_${ENV}_err_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

RUNPATH=${SCRATCH}/pettingZoo/${ENV}/\$SLURM_JOB_ID
mkdir -p \$RUNPATH

pushd ..

cat run-dvracer.py

cp run-dvracer.py \$RUNPATH
cp _jobs/settings.sh \$RUNPATH
cp -r _model/ \$RUNPATH

popd

pushd \$RUNPATH

OMP_NUM_THREADS=12 python3 run-dvracer.py --env "$ENV" --l2 $L2 --opt $OPT --lr $LR --model $MODEL --nn $NN --run 0

cd results
resdir=\$(ls -d _result_dvracer_*)
python3 -m korali.rlview --dir \$resdir --output dvracer.png

popd

date
EOF

sbatch run.sbatch
