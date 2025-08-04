import modal
import subprocess
import sys

# The name for the persistent volume where your project is stored.
VOLUME_NAME = "val_vid_tracker"

# The path to your main script inside the volume.
MAIN_SCRIPT_PATH = "/val_tracker.py"

# --- Modal App Definition ---
app = modal.App("tracker-launcher")

# Define the remote storage volume.
project_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Define the software environment for the container.
# Using `force_build=True` can be helpful if you change the dependencies.
image = (
    modal.Image.debian_slim(python_version="3.13.5")
    .pip_install(
        "torch",
        "torchvision",
        "ultralytics",
        "opencv-python-headless",
        "scipy",
        "tqdm",
        "numpy",
        "pillow",
        "matplotlib",
        "dataclasses",
        "pathlib",
        "pygame"
    )
    .apt_install(
        "git",
        "libgl1-mesa-glx",
        "libglib2.0-0"
        )
)

@app.function(
    # Mount the project volume to the `/project` directory inside the container.
    # All your code, models, and input files will be available at this path.
    volumes={"/project": project_volume},
    # Use the custom image with all dependencies installed.
    image=image,
    # Request a GPU for model inference. Adjust if you need a different type.
    # I recommend using the L40S as it is faster.
    gpu="L40S",
    # Set a longer timeout for video processing.
    timeout=3600, # 1 hour
    # Allow the container to be idle for a few minutes without shutting down.
    min_containers=1
)
def run_valorant_tracker(*args):
    """
    This function runs in a Modal container, executes the main tracking script,
    and streams its output back to your local terminal.
    """
    script_path_in_container = f"/project{MAIN_SCRIPT_PATH}"
    
    # Build command with arguments  
    command = [sys.executable, script_path_in_container] + list(args)
    
    print(f"Attempting to run: {' '.join(command)}")
    print(f"Working directory: /project/")

    # We use subprocess to run your main script as if you were in a terminal.
    # The `cwd` (current working directory) is set to the project root,
    # so all relative paths in your script (like for models) work correctly.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd="/project/", # Set the working directory
        text=True,
        bufsize=1, # Line-buffered
    )

    # Stream the output from the script in real-time.
    print("--- Script Output ---")
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.wait()
    print("--- End of Script Output ---")
    if process.returncode == 0:
        print("\nRemote script finished successfully.")
        print("Results will be available in track_output/ after sync")
    else:
        print(f"\nRemote script failed with exit code: {process.returncode}")

@app.local_entrypoint()
def main():
    """
    This function runs on your local machine to trigger the remote execution.
    """
    import sys
    
    # Get command line arguments (skip the script name)
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if args:
        print(f"Launching remote Valorant video tracker with args: {' '.join(args)}")
    else:
        print("Launching remote Valorant video tracker with refactored architecture...")
    
    run_valorant_tracker.remote(*args)