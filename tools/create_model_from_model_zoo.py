# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test IR roundtrip with ONNX model zoo.

Usage:
    python model_zoo_test.py --jobs 8
"""

from __future__ import annotations

import argparse
import multiprocessing.pool
import os
import pathlib
import tempfile

import create_model
import onnx_ir as ir
import tqdm
from onnx import hub


def create_one_model(model_info: hub.ModelInfo):
    model_name = model_info.model
    model_path = model_info.model_path
    with tempfile.TemporaryDirectory() as temp_dir:
        # For parallel testing, this must be in a separate process because hub.set_dir
        # is not thread-safe.
        hub.set_dir(temp_dir)
        model_proto = hub.load(model_name)
    print(f"\n----Creating from: {model_name} @ {model_path}----")
    out_path = pathlib.Path(__file__).parent / "models" / "model_zoo" / f"{model_name}.onnx"
    create_model.create_model(ir.from_proto(model_proto), os.fspath(out_path))


def main():
    parser = argparse.ArgumentParser(
        description="Create test models from ONNX model zoo."
    )
    parser.add_argument(
        "-k",
        type=str,
        default=None,
        help="Keyword to filter the models. Default is None.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run. Default is 1.",
    )
    args = parser.parse_args()

    model_list = hub.list_models()
    if args.k:
        # Filter the models by name
        name = args.k.lower()
        model_list = [model for model in model_list if name in model.model.lower()]

    # run checker on each model
    # Use multi-processing to speed up the testing process
    with multiprocessing.pool.Pool(args.jobs) as pool:
        list(
            tqdm.tqdm(
                pool.imap_unordered(create_one_model, model_list),
                "Creating...",
                total=len(model_list),
            )
        )


if __name__ == "__main__":
    main()
