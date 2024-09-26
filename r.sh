# python3 main_split.py --trial 5 --cycle 10 --epochs 200  --dataset 'CIFAR10' --n-class 10 --n-query 500 --method 'Random'  --n-initial 500
# python3 main_split.py --trial 5 --cycle 10 --epochs 200  --dataset 'CIFAR10' --n-class 10 --n-query 500 --method 'Coreset'  --n-initial 500
# python3 main_split.py --trial 5 --cycle 10 --epochs 200  --dataset 'CIFAR10' --n-class 10 --n-query 500 --method 'Uncertainty' --uncertainty 'Entropy'  --n-initial 500
# python3 main_split.py --trial 5 --cycle 10 --epochs 200  --dataset 'CIFAR10' --n-class 10 --n-query 500 --method 'BADGE' --n-initial 500
python3 main_split.py --trial 5 --cycle 10 --epochs 200  --dataset 'CIFAR10' --n-class 10 --n-query 500 --method 'AlphaMixSampling'  --n-initial 500
python3 main_split.py --trial 5 --cycle 10 --epochs 200  --dataset 'CIFAR10' --n-class 10 --n-query 500 --method 'LFOSA'  --n-initial 500
python3 main_split.py --trial 5 --cycle 10 --epochs 200  --dataset 'CIFAR10' --n-class 10 --n-query 500 --method 'TIDAL'  --n-initial 500
