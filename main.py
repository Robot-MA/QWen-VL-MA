import subprocess

def run_script(script_path, *args):
    cmd = [script_path] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing {script_path} with arguments {args}:\n{result.stderr}")
    else:
        print(f"Output of {script_path} with arguments {args}:\n{result.stdout}")


if __name__ == "__main__":
    run_script("/home/duanj1/m2t2/manipulate_anything/RLBench/dataset_generator.py","eval.checkpoint=/home/duanj1/m2t2/manipulate_anything/RLBench/pick_and_place.pth","eval.mask_thresh=0.01", "eval.retract=0.2", "rlbench.demo_path=/home/duanj1/m2t2/manipulate_anything/RLBench/val", "rlbench.task_name=pick_one_ycb_place")
    run_script("/home/duanj1/CameraCalibration/LLMs/Qwen-VL/locate.py")
    
