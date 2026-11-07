"""One-command downloader for either or both datasets in this project."""

from __future__ import annotations

import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all", "zenodo20", "covid19_20"), default="all")
    args = parser.parse_args()
    if __package__:
        from . import download_covid19_20, download_zenodo20, prepare_zenodo20
    else:
        import download_covid19_20
        import download_zenodo20
        import prepare_zenodo20
    if args.dataset in {"all", "zenodo20"}:
        download_zenodo20.main()
        prepare_zenodo20.main()
    if args.dataset in {"all", "covid19_20"}:
        download_covid19_20.main()


if __name__ == "__main__":
    main()
