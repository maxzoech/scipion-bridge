"""Instance names of protocols, values given to their constructors, and the
configuration of a run as a mapping of plain values (in-process)."""

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import pytest

import scipion_bridge as B
from scipion_bridge.backend.ray import RayPipelineRunner
from scipion_bridge.core.protocol.run_config import (
    check_required,
    default_parameters,
    parameter_values,
)
from scipion_bridge.core.streaming import FileRunStore
from scipion_bridge.single_particle import Particle


class Precision(Enum):
    FP16 = "fp16"
    FP32 = "fp32"


class Align(B.Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input()
    num_classes: B.Field[int] = B.Field(default=10)
    precision: B.Field[Precision] = B.Field(default=Precision.FP32)

    @classmethod
    def outputs(cls):
        return {"particles": B.Set[Particle]}

    def steps(self):
        return self.particles.map(self._apply)

    def _apply(self, particles):
        return {"particles": particles}


class NeedsModel(B.Protocol):
    particles: B.Input[B.Set[Particle]] = B.Input()
    model: B.Field[str] = B.Field(optional=False)

    @classmethod
    def outputs(cls):
        return {"particles": B.Set[Particle]}

    def steps(self):
        return self.particles.map(self._apply)

    def _apply(self, particles):
        return {"particles": particles}


@pytest.fixture(autouse=True)
def rewire_test_container(setup_test_container):
    """A runner wires the Ray container when it builds its pipeline; wire the
    test container again so later tests see their overrides."""
    yield
    setup_test_container.wire(packages=["scipion_bridge"])


class Config(Mapping):
    """A mapping that is no dict, as OmegaConf's DictConfig."""

    def __init__(self, **values):
        self._values = values

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


# -- Names and constructor values ------------------------------------------------
@pytest.mark.parametrize(
    "declaration",
    [
        "name: B.Field[int] = B.Field(default=1)",
        "name: B.Input[B.Set[Particle]] = B.Input()",
        "name: int",
        "protocol_id: B.Field[str] = B.Field(default='a')",
    ],
)
def test_name_and_protocol_id_cannot_be_declared(declaration):
    source = f"""
class Named(B.Protocol):
    {declaration}

    @classmethod
    def outputs(cls):
        return {{}}

    def steps(self):
        pass
"""
    with pytest.raises(TypeError, match="reserved"):
        exec(source, {"B": B, "Particle": Particle})


def test_the_name_defaults_to_the_class_name():
    assert Align().name == "Align"
    assert Align(name="align").name == "align"


def test_a_name_must_be_an_identifier():
    with pytest.raises(ValueError, match="identifier"):
        Align(name="my align")


def test_unknown_constructor_values_are_rejected():
    with pytest.raises(TypeError, match="no parameters"):
        Align(num_class=5)


def test_constructor_values_are_the_defaults_of_the_instance():
    align = Align(num_classes=50)

    assert align.num_classes.value == 50
    assert Align().num_classes.value == 10
    assert default_parameters(align) == {
        "Align": {"num_classes": 50, "precision": "FP32"},
    }


def test_a_required_parameter_given_to_the_constructor_is_not_missing():
    with pytest.raises(ValueError, match=r"NeedsModel\.model"):
        check_required(NeedsModel(), {})

    check_required(NeedsModel(model="small"), {})


def test_parameter_values_convert_enum_names_and_reject_unknown_keys():
    assert parameter_values(
        Align(),
        Config(Align=Config(precision="FP16", num_classes=3)),
    ) == {"Align": {"precision": Precision.FP16, "num_classes": 3}}
    with pytest.raises(ValueError, match="No protocols named"):
        parameter_values(Align(), {"Other": {}})
    with pytest.raises(ValueError, match="Unknown parameters"):
        parameter_values(Align(), {"Align": {"classes": 3}})


# -- The configuration of a run ------------------------------------------------
def test_the_default_config_lists_inputs_parameters_and_options():
    runner = RayPipelineRunner(Align(name="align", num_classes=50) | NeedsModel())

    config = runner.default_config()

    assert config["inputs"] == {"particles": None}
    assert config["parameters"] == {
        "align": {"num_classes": 50, "precision": "FP32"},
        "NeedsModel": {"model": None},
    }
    assert config["num_loaders"] == 4 and config["unique_elements"] == "off"
    assert config["run_dir"] is None


def test_launch_converts_a_mapping_of_plain_values(tmp_path):
    runner = RayPipelineRunner(Align(), sink=lambda outputs: None)
    (tmp_path / "a.star").touch()
    (tmp_path / "b.star").touch()

    with patch.object(runner, "run") as run:
        runner.launch(
            inputs=Config(particles=[str(tmp_path / "*.star")]),
            parameters=Config(Align=Config(precision="FP16")),
            run_dir=str(tmp_path / "run"),
            unique_elements="warn",
            num_loaders=2,
        )

    run.assert_called_once_with(
        {"particles": [tmp_path / "a.star", tmp_path / "b.star"]},
    )
    assert runner.parameters == {"Align": {"precision": Precision.FP16}}
    assert isinstance(runner.run_store, FileRunStore)
    assert runner.unique_elements is B.UniqueElements.WARN
    assert runner.num_loaders == 2


def test_launch_rejects_unknown_options_and_inputs():
    runner = RayPipelineRunner(Align())

    with pytest.raises(TypeError, match="Unknown options"):
        runner.launch(num_loader=2)
    with pytest.raises(ValueError, match=r"Unknown inputs \['micrographs'\]"):
        runner.launch(inputs={"micrographs": "m.star"})


def test_launch_explains_without_running(tmp_path, capsys):
    runner = RayPipelineRunner(Align())

    with patch.object(runner, "run") as run:
        runner.launch(explain=str(tmp_path / "report.html"))

    run.assert_not_called()
    assert "Align" in (tmp_path / "report.html").read_text()


def test_the_command_line_is_one_frontend_of_the_config(tmp_path):
    runner = RayPipelineRunner(Align() | NeedsModel())
    parser = runner.argument_parser()
    parser.add_argument("--launch-dashboard", action="store_true")

    args = parser.parse_args(
        [
            *["--particles", "a.star"],
            *["--Align.num-classes", "5", "--Align.precision", "FP16"],
            *["--NeedsModel.model", "small"],
            *["--run-dir", str(tmp_path), "--no-checkpoint-outputs"],
            "--launch-dashboard",
        ],
    )

    assert args.launch_dashboard
    assert runner.config_of(args) == {
        "inputs": {"particles": ["a.star"]},
        "parameters": {
            "Align": {"num_classes": 5, "precision": "FP16"},
            "NeedsModel": {"model": "small"},
        },
        "run_dir": str(tmp_path),
        "checkpoint_outputs": False,
    }


def test_the_terminal_application_launches_the_parsed_config():
    runner = RayPipelineRunner(Align())

    with patch.object(runner, "launch") as launch:
        runner.launch_as_terminal_application(
            ["--particles", "a.star", "--Align.num-classes", "5"],
        )

    launch.assert_called_once_with(
        inputs={"particles": ["a.star"]},
        parameters={"Align": {"num_classes": 5}},
    )


def test_inputs_given_as_one_path_are_read_on_the_driver():
    runner = RayPipelineRunner(Align())

    with patch.object(runner, "run") as run:
        runner.launch(inputs={"particles": "data/particles.star"})

    run.assert_called_once_with({"particles": Path("data/particles.star")})
