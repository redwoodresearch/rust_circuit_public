import os

WORKING_RRFS_DIR = "/working/rrfs"
HOME_RRFS_DIR = f'{os.environ.get("HOME", "/home/ubuntu")}/rrfs'
RRFS_DIR = os.environ.get(
    "RR_RRFS_DIR",
    HOME_RRFS_DIR if os.path.exists(HOME_RRFS_DIR) else WORKING_RRFS_DIR,
)
