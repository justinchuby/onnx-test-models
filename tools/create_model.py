import onnx_ir as ir
import onnx_ir.passes.common as common_passes
import onnxscript.version_converter


class InitializerToInputPass(ir.passes.InPlacePass):
    def __init__(self, size_limit: int = 64):
        """
        Initialize the pass to convert initializers to inputs.

        Args:
            size_limit (int): The maximum size of an initializer to be converted.
        """
        super().__init__()
        self.size_limit = size_limit

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        """
        Convert initializers in the model to inputs.

        Args:
            model (ir.Model): The ONNX model to process.
        """
        modified = False
        for initializer in tuple(model.graph.initializers.values()):
            if initializer.const_value.size <= self.size_limit:
                continue
            modified = True
            initializer.const_value = None
            model.graph.initializers.pop(initializer.name)
            if not initializer.is_graph_input():
                model.graph.inputs.append(initializer)
        return ir.passes.PassResult(model, modified)


def process_model(model: ir.Model, target_version: int) -> ir.Model:
    passes = ir.passes.Sequential(
        onnxscript.version_converter.ConvertVersionPass(target_version, fallback=True),
        common_passes.LiftConstantsToInitializersPass(),
        common_passes.LiftSubgraphInitializersToMainGraphPass(),
        InitializerToInputPass(),
    )
    model = passes(model).model
    model = ir.external_data.load_to_model(model)
    return model


def create_model(in_path: str, out_path: str) -> None:
    """
    Create a processed ONNX model from the input path and save it to the output path.

    Args:
        in_path (str): Path to the input ONNX model.
        out_path (str): Path to save the processed ONNX model.
    """
    model = ir.load(in_path)
    model = process_model(model)
    ir.save(model, out_path)
    print(f"Processed model saved to {out_path}")


def main():
    """
    Main function to handle command line arguments and process the model.

    Args:
        args: Command line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Process an ONNX model.")
    parser.add_argument("in_path", type=str, help="Path to the input ONNX model.")
    parser.add_argument(
        "out_path", type=str, help="Path to save the processed ONNX model."
    )
    parsed_args = parser.parse_args()

    create_model(parsed_args.in_path, parsed_args.out_path)


if __name__ == "__main__":
    main()
