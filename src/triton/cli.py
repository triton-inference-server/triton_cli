import argparse


def parse_args():
    parser = argparse.ArgumentParser(prog='triton', description='CLI to interact with Triton Inference Server')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

if __name__ == "__main__":
    main()
