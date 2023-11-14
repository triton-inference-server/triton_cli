#!/usr/bin/env python3
import parser
from logger import logger


def main():
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"{e}")


if __name__ == "__main__":
    main()
