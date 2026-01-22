from __future__ import annotations

from pathlib import Path
import urllib.request


URL_SNAP = "https://users.flatironinstitute.org/~camels/Sims/Astrid_DM/CV/CV_0/snapshot_090.hdf5"
URL_GROUPS = "https://users.flatironinstitute.org/~camels/Sims/Astrid_DM/CV/CV_0/groups_090.hdf5"

FILENAME_SNAP = "CAMELS_snapshot.hdf5"
FILENAME_GROUPS = "CAMELS_groups.hdf5"

def ensure_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"Found: {destination}")
        return

    print(f"Missing: {destination} — downloading...")
    urllib.request.urlretrieve(url, destination)
    print("Done.")


def main() -> None:
    # Directory *relative to this module file*
    module_dir = Path(__file__).resolve().parent

    # Optional: put downloads into a subfolder next to the script
    data_dir = module_dir / "data"

    snap_path = data_dir / FILENAME_SNAP
    groups_path = data_dir / FILENAME_GROUPS

    ensure_download(URL_SNAP, snap_path)
    ensure_download(URL_GROUPS, groups_path)


if __name__ == "__main__":
    main()