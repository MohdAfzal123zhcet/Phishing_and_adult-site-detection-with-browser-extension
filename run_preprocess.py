import os, sys, runpy
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)
# run the preprocess script
runpy.run_path(os.path.join(repo_root, "scripts", "preprocess_chunked.py"), run_name="__main__")
