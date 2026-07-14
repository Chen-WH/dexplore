python -u dexplore/eval_distill.py \
    --task Dexplore_Distill --distill \
    --cfg_env dexplore/data/cfg/inspire_distill.yaml \
    --cfg_train dexplore/data/cfg/train/rlg/inspire_distill.yaml \
    --checkpoint checkpoint/inspire_distill/nn/latest.pth \
    --num_envs 4 --output eval_results_distill.json
