from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import argparse
import torch
import timeit
import numpy as np 

def get_args():
    parser = argparse.ArgumentParser()
    # --- hyperparameters --- #
    parser.add_argument('--vocab_size',type=int,default=10000,help='vocab size')
    parser.add_argument('--batch_size',type=int,default=4,help='bs')
    parser.add_argument('--context_length', type=int, default=256, help='Context length')
    parser.add_argument('--d_model', type=int, default=768, help='model dim')
    parser.add_argument('--d_ff', type=int, default=3072, help='ff dim')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--rope_theta', type=float, default=10000.0,help='Base frequency for RoPE. Increase for longer context.')

    # --- optimizer --- #
    parser.add_argument('--beta1', type=float, default=0.9,help='beta1')
    parser.add_argument('--beta2', type=float, default=0.95,help="beta2")
    parser.add_argument('--weight_decay', type=float, default=1e-2,help="weight_decay")
    parser.add_argument('--eps', type=float, default=1e-8,help="eps")
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    # --- control --- #
    parser.add_argument('--warm_up',type=int,default=5,help='warm up steps')
    parser.add_argument('--iters',type=int,default=10,help='measure steps')

    return parser.parse_args()

def generate_random_batch(batch_size,context_length,vocab_size,device):
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size,context_length),
        device=device,
        dtype=torch.long
    )
    labels = input_ids.clone()
    return input_ids,labels

def run_step(model,optimizer,x,y):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = cross_entropy(logits,y)
    loss.backward()
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def benchmark():
    print("[*] 正在解析参数并初始化设备...")
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 正在创建模型 (d_model={args.d_model}, layers={args.n_layer})...")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.n_layer,
        num_heads=args.n_head,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1,args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    # --- generate data --- #
    print("[*] 正在准备随机测试数据...")
    x,y= generate_random_batch(
            args.batch_size,
            args.context_length,
            args.vocab_size,
            device
        )
    
    # --- warm up --- #
    print(f"[>] 开始预热 (Warm-up steps: {args.warm_up})...")
    for _ in range(args.warm_up):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy(logits,y)
        loss.backward()
        optimizer.step()

    # --- Measurement --- #
    # print(f"[>] 开始正式测量 (Iters: {args.iters})...")
    # total_time = timeit.timeit(lambda:run_step(model,optimizer,x,y),number=args.iters)
    # avg_latency = total_time / args.iters
    # throughput = (args.batch_size * args.context_length) / avg_latency
    
    # print(f"总迭代次数: {args.iters}")
    # print(f"总耗时: {total_time:.4f} s")
    # print(f"平均单步耗时: {avg_latency:.4f} s")
    # print(f"吞吐量: {throughput:.2f} tokens/sec")

    forward_times = []
    backward_times = []
    for i in range(args.iters):
        print(f"\r正在测量第 {i+1}/{args.iters} 次迭代...", end="", flush=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # --- forward ---
        optimizer.zero_grad(set_to_none=True)
        t0 = timeit.default_timer()
        logits = model(x)
        loss = cross_entropy(logits,y)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = timeit.default_timer()
        forward_times.append(t1-t0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # --- backward ---
        t2 = timeit.default_timer()
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = timeit.default_timer()
        backward_times.append(t3-t2)

    f_avg = np.mean(forward_times)
    f_std = np.std(forward_times)

    b_avg = np.mean(backward_times)
    b_std = np.std(backward_times)

    print(f"\n{'='*30}")
    print(f"Forward Pass:  {f_avg:.6f}s ± {f_std:.6f}s")
    print(f"Backward Pass: {b_avg:.6f}s ± {b_std:.6f}s")



if __name__ == "__main__":
    benchmark()
    
