import subprocess
import os

# git clone https://github.com/NVIDIA/apex
# poetry run python -m pip install -v disable-pip-version-check --no-cache-dir --no-build-isolation --global-option=--cpp_ext --global-option=--cuda_ext /app/apex/


def install_apex():
    apex_repo_url = "https://github.com/NVIDIA/apex"

    try:
        subprocess.run(["git", "clone", apex_repo_url], check=True)
    except:
        pass

    subprocess.run(
        [
            "pip",
            "install",
            "-v",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-build-isolation",
            "--global-option=--cpp_ext",
            "--global-option=--cuda_ext",
            "./apex/",
        ],
        check=True,
    )

    # remove the cloned repo # this is a folder!
    os.system("rm -rf ./apex/")


if __name__ == "__main__":
    install_apex()
