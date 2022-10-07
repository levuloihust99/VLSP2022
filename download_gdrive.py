import argparse
import gdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    gdown.download_folder(url=args.url, quiet=False)


if __name__ == "__main__":
    main()
