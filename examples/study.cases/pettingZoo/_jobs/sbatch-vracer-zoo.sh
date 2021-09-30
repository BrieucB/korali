#!/bin/bash -l

source ../settings.sh

echo $ENV
echo $MODEL
echo $DIS
echo $L2
echo $OPT
echo $LR

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=zoo_VRACER_${ENV}
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

cat run-vracer.py

cp run-vracer.py \$RUNPATH
cp _jobs/settings.sh \$RUNPATH
cp -r _model/ \$RUNPATH

popd

pushd \$RUNPATH

OMP_NUM_THREADS=12 python3 run-vracer.py --env "$ENV" --dis "$DIS" --l2 $L2 --opt $OPT --lr $LR --model $model

cd results
resdir=\$(ls -d _result_vracer_*)
python3 -m korali.rlview --dir \$resdir --output vracer.png

popd

date
EOF

sbatch run.sbatch
