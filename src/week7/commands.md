python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 2 --lora_alpha 16 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/budget0p2_lora_only_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 2 --lora_alpha 16 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/budget0p2_lora_only_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 2 --lora_alpha 16 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7/budget0p2_lora_only_s2.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/budget0p2_norm_only_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/budget0p2_norm_only_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7/budget0p2_norm_only_s2.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layerwise --policy deep_lora --lora_r 2 --lora_alpha 16 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/budget0p2_hybrid_layerwise_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layerwise --policy deep_lora --lora_r 2 --lora_alpha 16 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/budget0p2_hybrid_layerwise_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layerwise --policy deep_lora --lora_r 2 --lora_alpha 16 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7/budget0p2_hybrid_layerwise_s2.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 4 --lora_alpha 32 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/budget0p5_lora_only_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 4 --lora_alpha 32 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/budget0p5_lora_only_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 4 --lora_alpha 32 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7/budget0p5_lora_only_s2.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/budget0p5_norm_only_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/budget0p5_norm_only_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7/budget0p5_norm_only_s2.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layerwise --policy deep_lora --lora_r 3 --lora_alpha 24 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/budget0p5_hybrid_layerwise_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layerwise --policy deep_lora --lora_r 3 --lora_alpha 24 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/budget0p5_hybrid_layerwise_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layerwise --policy deep_lora --lora_r 3 --lora_alpha 24 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7/budget0p5_hybrid_layerwise_s2.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/budget1p0_lora_only_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/budget1p0_lora_only_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7/budget1p0_lora_only_s2.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/budget1p0_norm_only_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/budget1p0_norm_only_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7/budget1p0_norm_only_s2.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layerwise --policy deep_lora --lora_r 6 --lora_alpha 24 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/budget1p0_hybrid_layerwise_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layerwise --policy deep_lora --lora_r 6 --lora_alpha 24 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/budget1p0_hybrid_layerwise_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layerwise --policy deep_lora --lora_r 6 --lora_alpha 24 --lora_dropout 0.0 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7/budget1p0_hybrid_layerwise_s2.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layers --policy even_lora --lora_r 3 --lora_alpha 24 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/layersweep_hybrid_evenlora_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layers --policy even_lora --lora_r 3 --lora_alpha 24 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/layersweep_hybrid_evenlora_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layers --policy odd_lora --lora_r 3 --lora_alpha 24 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/layersweep_hybrid_oddlora_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layers --policy odd_lora --lora_r 3 --lora_alpha 24 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/layersweep_hybrid_oddlora_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layerwise --policy deep_lora --lora_r 3 --lora_alpha 24 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7/layersweep_hybrid_deeplora_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layerwise --policy deep_lora --lora_r 3 --lora_alpha 24 --epochs 5 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7/layersweep_hybrid_deeplora_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 4 --lora_alpha 32 --lora_dropout 0.0 --epochs 20 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7_long/budget0p5_lora_only_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 4 --lora_alpha 32 --lora_dropout 0.0 --epochs 20 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7_long/budget0p5_lora_only_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method lora --lora_r 4 --lora_alpha 32 --lora_dropout 0.0 --epochs 20 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7_long/budget0p5_lora_only_s2.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 20 --lr 1e-3 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7_long/budget0p5_norm_only_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 20 --lr 1e-3 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7_long/budget0p5_norm_only_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method norm --epochs 20 --lr 1e-3 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7_long/budget0p5_norm_only_s2.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layers --policy even_lora --lora_r 3 --lora_alpha 24 --lora_dropout 0.0 --epochs 20 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 0 --log_csv reports/week7_long/budget0p5_hybrid_layers_s0.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layers --policy even_lora --lora_r 3 --lora_alpha 24 --lora_dropout 0.0 --epochs 20 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 1 --log_csv reports/week7_long/budget0p5_hybrid_layers_s1.csv

python src/week6/train.py --dataset cifar100 --model vit_small_patch16_224 --method hybrid_layers --policy even_lora --lora_r 3 --lora_alpha 24 --lora_dropout 0.0 --epochs 20 --lr 1e-3 --lr_lora 5e-4 --lr_norm 1e-3 --weight_decay 0.05 --seed 2 --log_csv reports/week7_long/budget0p5_hybrid_layers_s2.csv

