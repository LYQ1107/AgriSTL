# Run Scripts

This file collects the custom run scripts added in this project.
Unless otherwise stated, commands in the main sections use Windows `cmd` style.
In Windows `cmd`, multi-line commands use `^`.
In Linux/macOS shells, use `\` or write the whole command on one line.

## 1. `namin` Image Prediction

### Training
```bat
python tools\train.py -d namin --lr 1e-3 -c configs\namin\simvp\SimVP_gSTA.py --ex_name namin_simvp_gsta
python tools\train.py -d namin --lr 1e-3 -c configs\namin\TimesNet.py --ex_name namin_timesnet
python tools\train.py -d namin --lr 1e-3 -c configs\namin\TimeMixer.py --ex_name namin_timemixer --gpus 1
python tools\train.py -d namin --lr 1e-3 -c configs\namin\TimeSformer.py --ex_name namin_timesformer --gpus 1
python tools\train.py -d namin --lr 1e-3 -c configs\namin\PredRNNv2.py --ex_name namin_predrnnv2 --gpus 1
python tools\train.py -d namin --lr 1e-3 -c configs\namin\TAU.py --ex_name namin_tau --gpus 1
python -m tools.train -d namin -m dmvfn -c configs\namin\DMVFN.py --ex_name namin_dmvfn --gpus 0
```

### Visualization on Linux/macOS
```bash
python tools/visualizations/vis_video.py \
    -d namin \
    -w work_dirs/namin_simvp_gsta \
    --index 0 \
    --save_dirs vis/fig_namin_vis_simvp_gsta

python tools/visualizations/vis_video.py \
    -d namin \
    -w work_dirs/namin_timesnet \
    --index 0 \
    --save_dirs vis/fig_namin_vis

python tools/visualizations/vis_video.py \
    -d namin \
    -w work_dirs/namin_timemixer \
    --index 0 \
    --save_dirs vis/fig_namin_vis_timemixer

python tools/visualizations/vis_video.py \
    -d namin \
    -w work_dirs/namin_predrnnv2 \
    --index 0 \
    --save_dirs vis/fig_namin_vis_predrnnv2

python tools/visualizations/vis_video.py \
    -d namin \
    -w work_dirs/namin_tau \
    --index 0 \
    --save_dirs vis/fig_namin_vis_tau

python tools/visualizations/vis_video.py \
    -d namin \
    -w work_dirs/namin_timesformer \
    --index 0 \
    --save_dirs vis/fig_namin_vis_timesformer

```

## 2. `yield` Estimation

### Data Preparation
```bat
python tools\prepare_data\prepare_yield_data.py --src-dir data\all_samples_30 --dst-dir data\yield_30
python tools\prepare_data\prepare_yield_data.py --src-dir data\all_samples_50 --dst-dir data\yield_50
```

### Training
```bat
python tools\train.py -d yield -m yield3dcnn -c configs\yield\Yield3DCNN.py --data_root .\data\yield_30 --ex_name yield30_baseline --num_workers 0 -b 1 -vb 1
python tools\train.py -d yield -m yield3dcnn -c configs\yield\Yield3DCNN.py --data_root .\data\yield_50 --ex_name yield50_baseline --num_workers 0 -b 1 -vb 1
```

### Training Without Final Test
```bat
python tools\train.py -d yield -m yield3dcnn -c configs\yield\Yield3DCNN.py --data_root .\data\yield_30 --ex_name yield30_baseline --num_workers 0 -b 1 -vb 1 --skip_test_after_train
```

### Background Training on Linux Server
```bash
nohup python tools/train.py -d yield -m yield3dcnn -c configs/yield/Yield3DCNN.py --data_root ./data/yield_30 --ex_name yield30_baseline --num_workers 0 -b 1 -vb 1 > yield30_baseline.log 2>&1 &
```

## 3. Useful Log Commands on Linux Server

```bash
tail -f yield30_baseline.log
tail -n 100 yield30_baseline.log
grep -n "Traceback" yield30_baseline.log
```
