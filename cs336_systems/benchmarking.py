from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch.cuda.nvtx as nvtx
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
    parser.add_argument('--profile_memory', action='store_true', help='是否生成显存快照')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'], help='Precision mode')

    return parser.parse_args()

def profile_step(model, optimizer, x, y, args, ptdtype):
    # 根据精度生成文件名，例如：memory_snapshot_fp16.pickle
    snapshot_file = f"memory_snapshot_{args.precision}.pickle"
    print(f"\n[!] 正在记录 {args.precision} 模式下的显存历史...")

    # 1. 开启记录 (包含 Python 堆栈)
    torch.cuda.memory._record_memory_history(enabled='all', max_entries=100000)

    # 2. 执行受控的单步更新
    # 使用与正式训练完全一致的 autocast 环境
    use_autocast = args.precision != 'fp32'
    use_scaler = args.precision == 'fp16'
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    optimizer.zero_grad(set_to_none=True)
    
    with torch.autocast(device_type='cuda', dtype=ptdtype, enabled=use_autocast):
        logits = model(x)
        loss = cross_entropy(logits, y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # 3. 导出快照
    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot(snapshot_file)
    
    # 4. 关闭记录
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"[OK] {args.precision} 显存快照已导出至 {snapshot_file}")

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
    
    ptdtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.precision]
    
    # 只有 fp16 需要 GradScaler，bf16 和 fp32 使用不执行任何操作的 dummy scaler 即可
    # 或者用逻辑判断，这里我们直接初始化，后面按需使用
    use_scaler = args.precision == 'fp16'
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    if args.profile_memory:
        profile_step(model,optimizer,x,y,args,ptdtype)
        return

    # --- warm up --- #
    print(f"训练精度：{ptdtype}")
    print(f"[>] 开始预热 (Warm-up steps: {args.warm_up})...")
    for _ in range(args.warm_up):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=ptdtype):
            logits = model(x)
            loss = cross_entropy(logits,y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    optimize_times = []
    for i in range(args.iters):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # --- Forward ---
        with torch.autocast(device_type='cuda', dtype=ptdtype, enabled=(args.precision != 'fp32')):
            t0 = timeit.default_timer()
            logits = model(x)
            loss = cross_entropy(logits, y)
            torch.cuda.synchronize()
            t1 = timeit.default_timer()
        forward_times.append(t1 - t0)

        # --- Backward ---
        t2 = timeit.default_timer()
        # 如果是 FP16，需要 scale loss
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t3 = timeit.default_timer()
        backward_times.append(t3 - t2)

        # --- Optimize ---
        t4 = timeit.default_timer()
        # scaler.step 内部会自动判断是否需要 unscale
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        t5 = timeit.default_timer()
        optimize_times.append(t5 - t4)

    f_avg = np.mean(forward_times)
    f_std = np.std(forward_times)

    b_avg = np.mean(backward_times)
    b_std = np.std(backward_times)

    o_avg = np.mean(optimize_times)
    o_std = np.std(optimize_times)

    print(f"\n{'='*30}")
    print(f"Forward Pass:  {f_avg:.6f}s ± {f_std:.6f}s")
    print(f"Backward Pass: {b_avg:.6f}s ± {b_std:.6f}s")
    print(f"Optimize:  {o_avg:.6f}s ± {o_std:.6f}s")
    


if __name__ == "__main__":
    benchmark()
