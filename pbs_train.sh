#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee

source activate b06201018
cd $PBS_O_WORKDIR
module load cuda/cuda-10.0/x86_64

for img in field house mount picasso tree vase water
do

mkdir logs/${img}-2-patch
cp logs/200000.tar logs/${img}-2-patch/200000.tar
python run_nerf_patch.py \
--config configs/fern.txt \
--expname ${img}-2-patch \
--style_path imgs/${img}-2.jpg \
--patch_size 80 \
--patch_num 2 \
--N_iters 7001 \
--lrate 1e-5 \
--w_content 1 \
--w_style 1e3 \
--w_tv 1e-4 \
--no_batching \
--i_weights 1000 \
--no_resume
rm logs/${img}-2-patch/200000.tar

python run_nerf_patch.py \
--config configs/fern.txt \
--expname ${img}-2-patch \
--style_path imgs/${img}-2.jpg \
--patch_size 80 \
--patch_num 2 \
--N_iters 7001 \
--lrate 1e-5 \
--w_content 1 \
--w_style 1e3 \
--w_tv 1e-4 \
--no_batching \
--i_weights 1000 \
--no_resume \
--render_only

mkdir logs/${img}-2-overlap
cp logs/200000.tar logs/${img}-2-overlap/200000.tar
python run_nerf_overlap.py \
--config configs/fern.txt \
--expname ${img}-2-overlap \
--style_path imgs/${img}-2.jpg \
--patch_size 80 \
--patch_num 2 \
--N_iters 7001 \
--lrate 1e-5 \
--w_content 1 \
--w_style 1e3 \
--w_tv 1e-4 \
--no_batching \
--i_weights 1000 \
--no_resume
rm logs/${img}-2-overlap/200000.tar

python run_nerf_overlap.py \
--config configs/fern.txt \
--expname ${img}-overlap \
--style_path imgs/${img}-2.jpg \
--patch_size 80 \
--patch_num 2 \
--N_iters 7001 \
--lrate 1e-5 \
--w_content 1 \
--w_style 1e3 \
--w_tv 1e-4 \
--no_batching \
--i_weights 1000 \
--no_resume \
--render_only
done

source deactivate