# Simple-Vision-Transformer-for-CIFAR10
Explanation of components and usage:
## 1: ```seed_everything```:
```os.environ['PYTHONHASHEED']```: Seeds all hashable operations (like on ```dict```) <br>
```torch.manual_seed```: Seeds CPU operations, initializing weights <br>
```torch.cuda.manual_seed_all```: Seeds GPU operations <br>
```torch.backends.cudnn.deterministic = True``` + ```torch.backends.cudnn.benchmark = False```: Seeds algorithms used in CUDA Deep Neural Networks instead of running benchmarks choosing optimal algorithms each step. <br>
## 2: Usage
Command line ```python vit_cifar_10.py --epochs ... --batch-size ... --lr ... --weight-decay ... --seed... --patch-size... --embed-dim... --depth ... --heads ... --mlp-ratio ... --device ... --save-dir ... --resume ...```
