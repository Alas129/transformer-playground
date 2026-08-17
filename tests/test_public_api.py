"""
The package's public surface.

src/__init__.py had grown to cover modern.py and moe.py but never lora.py or
train.py, so `from src import LoRALinear` failed while `from src import RMSNorm`
worked -- an inconsistency a reader hits before any docs explain it.
"""

import importlib

import pytest

import src

MODULES = ["embeddings", "attention", "transformer", "gpt", "modern", "moe",
           "lora", "train"]


class TestAllIsHonest:
    def test_every_name_in_all_is_importable(self):
        missing = [name for name in src.__all__ if not hasattr(src, name)]

        assert missing == [], f"__all__ names nothing exists for: {missing}"

    def test_no_duplicates(self):
        assert len(src.__all__) == len(set(src.__all__))

    def test_star_import_works(self):
        namespace = {}
        exec("from src import *", namespace)

        for name in src.__all__:
            assert name in namespace


class TestEveryModuleIsRepresented:
    @pytest.mark.parametrize("module_name", MODULES)
    def test_module_imports(self, module_name):
        importlib.import_module(f"src.{module_name}")

    @pytest.mark.parametrize("module_name", MODULES)
    def test_module_contributes_to_the_package_api(self, module_name):
        """
        Each module should surface at least one name through the package, so the
        package is a real index of what src/ offers rather than a partial one.
        """
        module = importlib.import_module(f"src.{module_name}")
        public = {
            name for name in vars(module)
            if not name.startswith("_")
            and getattr(vars(module)[name], "__module__", "") == module.__name__
        }

        assert public & set(src.__all__), (
            f"src.{module_name} exports nothing through the package"
        )


class TestKeyEntryPoints:
    @pytest.mark.parametrize("name", [
        "LoRALinear", "apply_lora", "merge_lora", "MultiAdapterLoRALinear",
        "CharTokenizer", "TextDataset", "train_gpt", "generate_text",
        "split_text", "configure_optimizer", "lr_lambda_for", "load_gpt",
        "LearnablePositionalEncoding", "DecoderBlock",
    ])
    def test_is_reachable_from_the_package(self, name):
        assert hasattr(src, name), f"src.{name} is not exported"
