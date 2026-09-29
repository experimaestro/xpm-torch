from typing import (
    BinaryIO,
    Callable,
    Iterator,
    List,
    TextIO,
    TypeVar,
    Union,
    Iterable,
    Tuple,
    Type,
)
import inspect
import logging
import os
import re
import torch
from pathlib import Path
from subprocess import run
import tempfile
from experimaestro import SubmitHook, Job, Launcher
from threading import Thread

T = TypeVar("T")


class StreamGenerator(Thread):
    """Create a FIFO pipe (*nix only) that is fed by the provider generator"""

    def __init__(self, generator: Callable[[Union[TextIO, BinaryIO]], None], mode="wb"):
        super().__init__()
        tmpdir = tempfile.mkdtemp()
        self.mode = mode
        self.filepath = Path(os.path.join(tmpdir, "fifo.json"))
        os.mkfifo(self.filepath)
        self.generator = generator
        self.error = False

    def run(self):
        try:
            with self.filepath.open(self.mode) as out:
                try:
                    self.generator(out)
                except Exception:
                    # Just write something so the file is closed
                    if isinstance(out, TextIO):
                        out.write("")
                    else:
                        out.write(b"0")
                    raise
        except Exception:
            self.error = True
            raise

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.join()
        self.filepath.unlink()
        self.filepath.parent.rmdir()
        if self.error:
            raise AssertionError("Error with the generator")


class Handler:
    """Returns a result that depends on the type of the argument

    Example:
    ```
    handler = Handler()

    @handler()
    def trectopics(topics: TrecTopics):
        return ("-topicreader", "Trec", "-topics", topics.path)

    @handler()
    def tsvtopics(topics: ir_csv.Topics):
        return ("-topicreader", "TsvInt", "-topics", topics.path)

    command.extend(handler[topics])

    ```
    """

    def __init__(self):
        self.handlers: List[Tuple[Type, Callable]] = []
        self.defaulthandler = None

    def default(self):
        assert self.defaulthandler is None

        def annotate(method):
            self.defaulthandler = method
            return method

        return annotate

    def __call__(self):
        def annotate(method):
            spec = inspect.getfullargspec(method)
            assert len(spec.args) == 1 and spec.varargs is None

            self.handlers.append((spec.annotations[spec.args[0]], method))

        return annotate

    def __getitem__(self, key):
        try:
            handler = next(
                handler
                for cls, handler in self.handlers
                if issubclass(key.__class__, cls)
            )
        except StopIteration:
            if self.default is None:
                raise RuntimeError(
                    f"No handler for {key.__class__} and no default handler"
                )
            handler = self.defaulthandler

        return handler(key)



def to_device(obj, device: Union[torch.device, str]):
    """Move tensors to device recursively. Handles nested structures.

    Args:
        obj: A tensor, dict, list, tuple, or nested combination
        device: Target device (e.g., 'cpu', torch.device('cuda'))

    Returns:
        Object with all tensors moved to the specified device
    """
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Handle NamedTuples (which have _fields)
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):
            return type(obj)(*(to_device(v, device) for v in obj))
        return type(obj)(to_device(v, device) for v in obj)
    else:
        # For other types, return as-is
        return obj


def foreach(iterator: Iterable[T], fn: Callable[[T], None]):
    for t in map(fn, iterator):
        pass


def batchiter(batchsize: int, iter: Iterator[T], keeppartial=True) -> Iterator[List[T]]:
    """Group items together to form of list of size `batchsize`"""
    samples = []
    for sample in iter:
        samples.append(sample)
        if len(samples) % batchsize == 0:
            yield samples
            samples = []

    # Yield last samples if keeppartial is true
    if keeppartial and len(samples) > 0:
        yield samples


def find_java_home(min_version: int = 6) -> str:
    """Find JAVA HOME"""
    paths = []

    # (1) Use environment variable
    if java_home := os.environ.get("FORCE_JAVA_HOME", None):
        return java_home

    if java_home := os.environ.get("JAVA_HOME", None):
        paths.append(Path(java_home) / "bin" / "java")

    # Try java
    paths.append("java")

    # (2) Use java -XshowSettings:properties
    for p in paths:
        try:
            p = run(
                [p, "-XshowSettings:properties", "-version"],
                check=True,
                capture_output=True,
            )

            if m := re.search(
                rb".*\n\s+java.version = (\d+)\.[\d\.]+(-\w+)?\n.*",
                p.stderr,
                re.MULTILINE,
            ):
                version = int(m[1].decode())
                if min_version <= version:
                    if m := re.search(
                        rb".*\n\s+java.home = (.*)\n.*", p.stderr, re.MULTILINE
                    ):
                        return m[1].decode()
                else:
                    logging.info(
                        "Java search (version >= %d): skipping %s", min_version, p
                    )

        except Exception:
            # silently ignore
            pass

    raise FileNotFoundError(f"Java (version >= {min_version}) not found")


class needs_java(SubmitHook):
    """Experimaestro hook that ensures that JAVA_HOME is set"""

    def __init__(self, version: int):
        self.version = version

    def spec(self):
        return self.version

    def process(self, job: Job, launcher: Launcher):
        job.environ["JAVA_HOME"] = find_java_home(self.version)


class Initializable:
    """Base class for all initializable (but just once)"""

    def initialize(self, *args, **kwargs):
        """Main initialization

        Calls :py:meth:`__initialize__` once (using :py:meth:`__initialize__`)
        """
        if not self._initialized:
            self._initialized = True
            self.__initialize__(*args, **kwargs)
        self._initialized = True

    def __init__(self):
        self._initialized = False

    def __initialize__(self, *args, **kwargs):
        """Initialize the object

        Parameters depend on the actual class
        """
        pass


def count_safetensors_params(file_path: Union[str, Path]) -> int:
    """Fast header-only parameter count for a .safetensors file.

    Reads only the unsigned 8-byte header size and JSON metadata block
    without loading weight arrays into memory.

    Args:
        file_path: Path to the .safetensors file.

    Returns:
        Total number of parameters across all tensors.
    """
    import json
    import math
    import struct

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Safetensors file not found at '{path}'")

    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))

    return sum(
        math.prod(info["shape"])
        for k, info in header.items()
        if k != "__metadata__" and "shape" in info
    )


def format_num_params(num_params: int) -> str:
    """Format integer parameter count into readable string (e.g. 22714113 -> '22.71M')."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def build_model_card(
    save_path: Union[str, Path],
    template_path: Union[str, Path],
    model_name: str,
    base_model: str,
    language: str = "en",
    license: str = "apache-2.0",
    pipeline_tag: str = "text-classification",
    **template_kwargs,
):
    """Build and write a Hugging Face ModelCard to README.md in save_path.

    Automatically calculates parameter counts from model.safetensors if present
    and passes total_parameters into the template context.
    """
    from huggingface_hub import ModelCard, ModelCardData

    save_path = Path(save_path)
    template_path = Path(template_path)

    if not template_path.exists():
        logging.warning(f"Model card template not found at '{template_path}'")
        return None

    st_path = save_path / "model.safetensors"
    if st_path.exists():
        try:
            num_params = count_safetensors_params(st_path)
            total_params_str = f"{num_params:,} ({format_num_params(num_params)})"
        except Exception as e:
            logging.warning(f"Could not compute parameter count from safetensors: {e}")
            total_params_str = "N/A"
    else:
        total_params_str = "N/A"

    card_data = ModelCardData(
        language=language,
        license=license,
        base_model=str(base_model),
        model_name=str(model_name),
        pipeline_tag=pipeline_tag,
    )

    card = ModelCard.from_template(
        card_data,
        template_path=str(template_path),
        base=str(base_model),
        model_id=str(model_name),
        total_parameters=total_params_str,
        **template_kwargs,
    )

    card.save(save_path / "README.md")
    logging.info(f"Model card written to {save_path / 'README.md'}")
    return card

