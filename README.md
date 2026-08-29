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

after ~3k steps on TinyStories, it gets reasonable outputs:
```
[Step 3000] Model Output:
Once upon a time there was a little boy named Ben. She loved to play with her friends. One day, she saw a big, red ball. She was very excited to see her friend, a little girl named Lily. She was very excited to see the tree.
Tim's mom said, "I will help you find my toy."
Tim and Sue played with the toy car. They played with the park, and they all played together every day.
```
