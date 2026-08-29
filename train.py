import argparse
from datetime import datetime
import logging
import modal
import os
from pprint import pprint
import torch
import tiktoken
import wandb

from transformer.model import TransformerLM
from transformer.optimizer import AdamW
from transformer.utils import cross_entropy
from transformer.data import MemoryMappedDataLoader, save_checkpoint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("smallgpt")
image = (
    modal.Image.debian_slim()
    .uv_pip_install(["torch", "numpy", "tiktoken", "huggingface_hub", "wandb", "einops"])
    .add_local_python_source("transformer")
)

PROJECT = "small_gpt"
EOS_TOKEN = "<|endoftext|>"

def create_args(arglist):
    parser = argparse.ArgumentParser("simple training script")

    # model (17M params, excl. embeddings)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--rope_theta", type=int, default=10_000)

    # optim related
    parser.add_argument("--lr", type=float, default=7e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--b1", type=float, default=0.95)
    parser.add_argument("--b2", type=float, default=0.99)
    parser.add_argument("--wd", type=float, default=1e-7)

    # data, default to TinyStories V2
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--train-path", type=str, default="yhshin1020/tinystories:tinystoriesv2_train.bin")

    # train
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--load-ckpt-path", type=str, default=None)

    # generation
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args(args=arglist)
    return args

def create_config(args):
    return {
        "model.d_model": args.d_model,
        "model.d_ff": args.d_ff,
        "model.num_heads": args.num_heads,
        "model.num_layers": args.num_layers,
        "model.rope_theta": args.rope_theta,
        "optim.lr": args.lr,
        "optim.betas": (args.b1, args.b2),
        "optim.weight_decay": args.wd,
        "optim.eps": args.eps,
        "data.vocab_size": args.vocab_size,
        "data.max_context_length": args.max_context,
        "data.train_path": args.train_path,
        "trainer.batch_size": args.batch_size,
        "trainer.log_every": args.log_every,
        "trainer.save_every": args.save_every,
        "trainer.wandb": args.wandb,
        "trainer.max_iter": args.max_iter,
        "gen.temp": args.temp,
        "gen.top_p": args.top_p,
    }

def create_model(args):
    model = TransformerLM(
        args.vocab_size, 
        args.max_context, 
        args.num_layers, 
        args.d_model, 
        args.d_ff, 
        args.num_heads, 
        args.rope_theta
    )
    return model


def sample_text(
    model: torch.nn.Module, 
    sample_prompt, 
    max_context_len: int, 
    tokenizer, 
    eos_token_id: int, 
    temp: float,
    top_p: float,
):
    generated = model.generate(
        sample_prompt, 
        max_context_len, 
        temp=temp,
        top_p=top_p,
        eos_token_id=eos_token_id
    )
    decoded = tokenizer.decode_batch(generated.cpu().tolist())
    print(decoded[0])


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

@app.function(
    gpu="A100-40GB", 
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=3600, # 1h timeout
)
def train(*arglist):
    args = create_args(arglist)

    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])

    config = create_config(args) 
    pprint(config)

    date = datetime.now().strftime("%m%d")
    time = datetime.now().strftime("%H%M")
    run_name = f"run_{date}_{time}"

    device = get_device()
    logger.info(f"Running on {device}")

    model = create_model(args)
    model.to(device)
    logger.info(f"Model size: {model.get_size_gb():.2f}GB")

    betas = (args.b1, args.b2)
    optim = AdamW(model.parameters(), args.lr, betas, args.eps, args.wd)

    if args.wandb:
        run = wandb.init(project=PROJECT, name=run_name, config=config)

    model.train()
    logger.info("Starting training...")

    tokenizer = tiktoken.get_encoding("gpt2")
    eos_token_id = tokenizer.encode(EOS_TOKEN, allowed_special={EOS_TOKEN})[0]
    sample_prompt = torch.tensor(
        [tokenizer.encode("Once upon a time there was a little boy named Ben")],
        device=device,
    )

    print("Model output before training: ")
    sample_text(model, sample_prompt, args.max_context, tokenizer, eos_token_id, args.temp, args.top_p)

    dl = MemoryMappedDataLoader(
        ds_path=args.train_path,
        batch_size=args.batch_size,
        context_len=args.max_context,
        device=device,
    )

    it = 0

    while it < args.max_iter:
        # mem_allocated = torch.cuda.memory_allocated()
        # logger.info(f"Iteration {it}: memory allocated: {mem_allocated/1024**3}")
        train, target = dl.get_batch()

        if it == 0:
            decoded = tokenizer.decode(train[0].cpu().tolist())
            print("Sample training data: ")
            print(decoded)

        optim.zero_grad()
        pred = model(train)
        loss = cross_entropy(pred, target)

        if it % args.log_every == 0:
            loss_cpu = loss.cpu().item()
            logger.info(f"Train loss at iteration {it}: {loss_cpu}")
            if args.wandb: 
                run.log({"loss": loss_cpu}, step=it)

        loss.backward()
        optim.step()
        del loss, pred

        if it > 0 and it % args.save_every == 0:
            os.makedirs("outputs", exist_ok=True)
            save_path = f"outputs/{run_name}_iteration{it}"
            save_checkpoint(model, optim, it, save_path)
            print(f"Model output at iteration {it}")
            sample_text(model, sample_prompt, args.max_context, tokenizer, eos_token_id, args.temp, args.top_p)
        it += 1

    if args.wandb:
        run.finish()


@app.local_entrypoint()
def modal_local(*arglist):
    train.remote(*arglist)


if __name__ == "__main__":
    train.local()
