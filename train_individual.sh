mkdir logs/$1-2-$2
cp logs/200000.tar logs/$1-2-$2/200000.tar
python run_nerf_$2.py \
--config configs/fern.txt \
--expname $1-2-$2 \
--style_path imgs/$1-2.jpg \
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
rm logs/$1-2-patch/200000.tar

python run_nerf_$2.py \
--config configs/fern.txt \
--expname $1-2-$2 \
--style_path imgs/$1-2.jpg \
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