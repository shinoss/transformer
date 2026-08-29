### small gpt training

setup:
```
uv sync
```

run locally:
```
python train.py
```

run on modal:
```
modal run train.py
```

log results to wandb:
```
--wandb
```

resume from checkpoint stored in Modal volume:
```
--resume-from-ckpt run_0829_1854_step_1000
```

you will see a log like this:
```
INFO:train:Resuming training from checkpoint run_0829_1911_step_2000  from step=2001
```
