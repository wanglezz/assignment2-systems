import subprocess
import torch

configs = [
    {"name":"small","d_model":768,"d_ff":3072,"n_layer":12,"n_head": 12},
    {"name":"medium","d_model":1024,"d_ff":4096,"n_layer":24,"n_head": 16},
    {"name":"large","d_model":1280,"d_ff":5120,"n_layer":36,"n_head": 20},
    {"name":"xl","d_model":1600,"d_ff":6400,"n_layer":48,"n_head": 25},
    {"name":"2.7B","d_model":2560,"d_ff":10240,"n_layer":32,"n_head": 32}
]

def run_all():
    for config in configs:
        cmd = [
            "uv","run","python","benchmarking.py",
            "--d_model", str(config["d_model"]),
            "--d_ff",str(config["d_ff"]),
            "--n_layer", str(config["n_layer"]),
            "--n_head",str(config["n_head"])
        ]
        print(f"--- 正在开始测试: {config['name']} ---")
        try:
            subprocess.run(cmd,check=True)
            print(f"Successfully completed: {config['name']}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while testing {config['name']}:")
            print(f"Exit code: {e.returncode}")
            print(f"跳过该配置，继续执行下一个...")
            torch.cuda.empty_cache()

    print("所有测试任务已尝试完毕。")

if __name__ == "__main__":
    run_all()