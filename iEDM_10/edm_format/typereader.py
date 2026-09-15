"""
Allows registration of type-readers by name, and retrieval of those readers.

Each reader function takes a single argument; a BaseReader object

"""

from __future__ import annotations

from inspect import isclass
from typing import Any, Callable, NamedTuple, TypeVar

from .basereader import BaseReader

# The registry is intentionally heterogeneous: each named EDM stream type
# reads and returns a different shape (int, float, vector, a class instance,
# ...). `Any` here is the registry's actual contract, not a shortcut.
ReaderFn = Callable[[BaseReader], Any]

_typeReaders: dict[str, ReaderFn] = {}

FnT = TypeVar("FnT")


class Property(NamedTuple):
    name: str
    value: Any


class AnimatedProperty(NamedTuple):
    name: str
    argument: int
    keys: list["Keyframe"]


class ArgumentProperty(NamedTuple):
    name: str
    argument: int


class Keyframe(NamedTuple):
    frame: float
    value: Any


def get_type_reader(typeName: str) -> ReaderFn:
    try:
        return _typeReaders[typeName]
    except KeyError:
        raise KeyError(
            "No reader defined for stream type '{}'".format(typeName)
        ) from None


def generate_property_reader(generic_type: str) -> ReaderFn:
    def _read_property(data: BaseReader) -> Property:
        name = data.read_string()
        value = get_type_reader(generic_type)(data)
        return Property(name, value)

    return _read_property


def generate_keyframe_reader(generic_type: str) -> ReaderFn:
    def _read_keyframe(stream: BaseReader) -> Keyframe:
        frame = stream.read_double()
        value = get_type_reader(generic_type)(stream)
        return Keyframe(frame=frame, value=value)

    return _read_keyframe


def generate_animated_property_reader(keyframe_type: str) -> ReaderFn:
    def _read_animatedproperty(stream: BaseReader) -> AnimatedProperty:
        name = stream.read_string()
        argument = stream.read_uint()
        count = stream.read_count("animated property key count")
        reader = get_type_reader(keyframe_type)
        keys = [reader(stream) for _ in range(count)]
        return AnimatedProperty(name=name, argument=argument, keys=keys)

    return _read_animatedproperty


def reads_type(withName: str) -> Callable[[FnT], FnT]:
    """Simple registration function to read named type objects"""

    def wrapper(fn: FnT) -> FnT:
        if not isclass(fn):
            _typeReaders[withName] = fn  # type: ignore[assignment]
        elif hasattr(fn, "read"):
            _typeReaders[withName] = fn.read
        else:
            raise RuntimeError("Unrecognised type reader {}".format(fn))
        fn.forTypeName = withName  # type: ignore[attr-defined]
        return fn

    return wrapper


def allow_properties(w: FnT) -> FnT:
    """Decorator to generate type-readers for type-as-property values"""
    name = "model::Property<{}>".format(w.forTypeName)  # type: ignore[attr-defined]
    _typeReaders[name] = generate_property_reader(w.forTypeName)  # type: ignore[attr-defined]
    return w


def animatable(keyname: str) -> Callable[[FnT], FnT]:
    def _wrapper(fn: FnT) -> FnT:
        keyframe_type = "model::Key<{}>".format(keyname)
        prop_type = "model::AnimatedProperty<{}>".format(
            fn.forTypeName  # type: ignore[attr-defined]
        )
        _typeReaders[keyframe_type] = generate_keyframe_reader(
            fn.forTypeName  # type: ignore[attr-defined]
        )
        _typeReaders[prop_type] = generate_animated_property_reader(keyframe_type)
        return fn

    return _wrapper


@allow_properties
@reads_type("unsigned int")
def _read_uint(data: BaseReader) -> int:
    return data.read_uint()


@animatable(keyname="key::FLOAT")
@allow_properties
@reads_type("float")
def read_prop_float(data: BaseReader) -> float:
    return data.read_float()


@animatable(keyname="key::VEC2F")
@allow_properties
@reads_type("osg::Vec2f")
def readVec2f(data: BaseReader) -> Any:
    return data.read_vec2f()


@animatable(keyname="key::VEC3F")
@allow_properties
@reads_type("osg::Vec3f")
def readVec3f(data: BaseReader) -> Any:
    return data.read_vec3f()


@animatable(keyname="key::VEC4F")
@allow_properties
@reads_type("osg::Vec4f")
def readVec4f(data: BaseReader) -> Any:
    return data.read_vec4f()


@allow_properties
@reads_type("osg::Vec3d")
def readVec3d(data: BaseReader) -> Any:
    return data.read_vec3d()


@reads_type("osg::Matrixf")
def readMatrixf(stream: BaseReader) -> Any:
    return stream.read_matrixf()


@reads_type("osg::Matrixd")
def readMatrixd(stream: BaseReader) -> Any:
    return stream.read_matrixd()


@reads_type("osg::Quat")
def readQuaternion(stream: BaseReader) -> Any:
    return stream.read_quaternion()


@allow_properties
@reads_type("const char*")
def readConstChar(stream: BaseReader) -> str:
    return stream.read_string()


@reads_type("model::ArgumentProperty")
def read_argproperty(stream: BaseReader) -> ArgumentProperty:
    name = stream.read_string()
    arg = stream.read_uint()
    return ArgumentProperty(name, arg)
