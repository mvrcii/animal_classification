import argparse
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description='Run a Python script on SLURM with GPU options.')
    parser.add_argument('--gpu', type=int, default=0, choices=[0, 1],
                        help="Choose GPU: 0=rtx2080ti (default), 1=rtx3090")
    parser.add_argument('--batch_size', type=int, help="Batch size for the model training.")
    parser.add_argument('--train', action='store_true', help="Use train.py instead of train_cv.py (default)")
    parser.add_argument('--job_name', type=str, default='train_cv', help="Job name for SLURM.")
    parser.add_argument('--attach', action='store_true', help="Attach to log output.")
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--CV_fold_path', type=str, default="data/cross_folds")
    args = parser.parse_args()

    # Training arguments
    script_name = "train.py" if args.train else "train_cv.py"
    data_dir = args.data_dir
    cv_fold_path = args.CV_fold_path

    cmd_str = f"python3 {script_name} --data_dir {data_dir}"
    if not args.train:
        cmd_str += f" --CV_fold_path {cv_fold_path}"
    if args.batch_size:
        cmd_str += f" --batch_size {args.batch_size}"

    # Slurm arguments
    gpu_types = {0: "rtx2080ti", 1: "rtx3090"}
    gpu_str = gpu_types[args.gpu]

    slurm_cmd = f'sbatch -p ls6 -J "{args.job_name}" --gres=gpu:{gpu_str}:1 --wrap="{cmd_str}" -o "logs/slurm-%j.out"'

    result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            print(f"Slurm job ID: {job_id}")

            if args.attach:
                print("Attaching to log file... (waiting 10s)")
                time.sleep(10)
                tail_cmd = f"tail -f logs/slurm-{job_id}.out"
                subprocess.run(tail_cmd, shell=True)
        else:
            print("Failed to submit job to Slurm or parse job ID.")


if __name__ == "__main__":
    main()
