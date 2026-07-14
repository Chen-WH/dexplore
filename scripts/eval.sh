python -u dexplore/evaluate.py \
    --task Dexplore_Inspire \
    --cfg_env dexplore/data/cfg/inspire.yaml \
    --cfg_train dexplore/data/cfg/train/rlg/inspire.yaml \
    --checkpoint checkpoint/inspire.pth \
    --headless --num_envs 64 --output eval_results_inspire.json
