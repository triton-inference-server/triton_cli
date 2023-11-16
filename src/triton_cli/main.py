#!/usr/bin/env python3
import parser

import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("triton")


def main():
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"{e}")
        raise e


if __name__ == "__main__":
    main()
