adapted from [Supervised Fitting of Geometric Primitives to 3D Point Clouds](https://github.com/lingxiaoli94/SPFN)

# Environment setup
Install environment.yml with conda. May need to add tensorrt to `LD_LIBRARY_PATH by appending to end of file:
```
variables:
  LD_LIBRARY_PATH: /path/to/conda-env/lib/python3.x/site-packages/tensorrt/
```

# Usage
Run prediction from `experiments/`
```
python ../spfn/eval.py /path/to/eval_config.yml
```

To finish parameter estimation and visualize:
```
python utils/recover_prim_params.py --files /path/to/glob/output.h5 --output /path/to/savedir [--metadata /path/to/metadata.h5 [--render]]
```
