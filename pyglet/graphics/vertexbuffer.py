"""OpenGL Buffer Objects.

:py:class:`~BufferObject` and a :py:class:`~BackedBufferObject` are provided.
The first is a lightweight abstraction over an OpenGL buffer, as created
with ``glGenBuffers``. The backed buffer object is similar, but provides a
full mirror of the data in CPU memory. This allows for delayed uploading of
changes to GPU memory, which can improve performance is some cases.
"""
from __future__ import annotations

import abc
import ctypes
import sys
from typing import TYPE_CHECKING, Sequence, Type

from _ctypes import Array, _Pointer, _SimpleCData

import pyglet
from pyglet.gl.gl import (
    GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_MAP_READ_BIT,
    GL_MAP_WRITE_BIT,
    GL_MAP_COHERENT_BIT,
    GL_MAP_PERSISTENT_BIT,
    GL_SYNC_FLUSH_COMMANDS_BIT,
    GL_SYNC_GPU_COMMANDS_COMPLETE,
    GL_TIMEOUT_EXPIRED,
    GL_WRITE_ONLY,
    GLubyte,
    GLuint,
    glBindBuffer,
    glBufferData,
    glBufferStorage,
    glBufferSubData,
    glClientWaitSync,
    glDeleteBuffers,
    glDeleteSync,
    glFenceSync,
    glGenBuffers,
    glMapBuffer,
    glMapBufferRange,
    glUnmapBuffer,
)

if TYPE_CHECKING:
    from pyglet.gl import Context
    from pyglet.graphics.shader import Attribute

CTypesDataType = Type[_SimpleCData]
CTypesPointer = _Pointer

class AbstractBuffer:
    """Abstract buffer of byte data.

    Attributes:
        size:
            Size of buffer, in bytes
        ptr:
            Memory offset of the buffer, as used by the ``glVertexPointer`` family of functions
    """

    ptr: int = 0
    size: int = 0

    @abc.abstractmethod
    def bind(self, target: int = GL_ARRAY_BUFFER) -> None:
        """Bind this buffer to an OpenGL target."""

    @abc.abstractmethod
    def unbind(self) -> None:
        """Reset the buffer's OpenGL target."""

    @abc.abstractmethod
    def set_data(self, data: Sequence[int] | CTypesPointer) -> None:
        """Set the entire contents of the buffer.

        Args:
            data:
                The byte array to set.

        """

    @abc.abstractmethod
    def set_data_region(self, data: Sequence[int] | CTypesPointer, start: int, length: int) -> None:
        """Set part of the buffer contents.

        Args:
            data:
                The byte array of data to set
            start:
                Offset to start replacing data
            length:
                Length of region to replace

        """

    @abc.abstractmethod
    def map(self) -> CTypesPointer[ctypes.c_ubyte]:
        """Map the entire buffer into system memory.

        The mapped region must be subsequently unmapped with `unmap` before
        performing any other operations on the buffer.

        Returns:
            Pointer to the mapped block in memory
        """

    @abc.abstractmethod
    def unmap(self) -> None:
        """Unmap a previously mapped memory block."""

    def resize(self, size: int) -> None:
        """Resize the buffer to a new size.

        Args:
            size:
                New size of the buffer, in bytes

        """

    @abc.abstractmethod
    def delete(self) -> None:
        """Delete this buffer, reducing system resource usage."""


class BufferObject(AbstractBuffer):
    """Lightweight representation of an OpenGL Buffer Object.

    The data in the buffer is not replicated in any system memory (unless it
    is done so by the video driver).  While this can reduce system memory usage,
    performing multiple small updates to the buffer can be relatively slow.
    The target of the buffer is ``GL_ARRAY_BUFFER`` internally to avoid
    accidentally overriding other states when altering the buffer contents.
    The intended target can be set when binding the buffer.
    """

    id: int
    size: int
    usage: int
    _context: Context | None

    def __init__(self, size: int, usage: int = GL_DYNAMIC_DRAW) -> None:
        """Initialize the BufferObject with the given size and draw usage.

        Buffer data is cleared on creation.
        """
        self.size = size
        self.usage = usage
        self._context = pyglet.gl.current_context

        buffer_id = GLuint()
        glGenBuffers(1, buffer_id)
        self.id = buffer_id.value

        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        data = (GLubyte * self.size)()
        glBufferData(GL_ARRAY_BUFFER, self.size, data, self.usage)

    def invalidate(self) -> None:
        glBufferData(GL_ARRAY_BUFFER, self.size, None, self.usage)

    def bind(self, target: int = GL_ARRAY_BUFFER) -> None:
        glBindBuffer(target, self.id)

    def unbind(self) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def bind_to_index_buffer(self) -> None:
        """Binds this buffer as an index buffer on the active vertex array."""
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.id)

    def set_data(self, data: Sequence[int] | CTypesPointer) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glBufferData(GL_ARRAY_BUFFER, self.size, data, self.usage)

    def set_data_region(self, data: Sequence[int] | CTypesPointer, start: int, length: int) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glBufferSubData(GL_ARRAY_BUFFER, start, length, data)

    def map(self) -> CTypesPointer[ctypes.c_byte]:
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        return ctypes.cast(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY),
                           ctypes.POINTER(ctypes.c_byte * self.size)).contents

    def map_range(self, start: int, size: int, ptr_type: type[CTypesPointer]) -> CTypesPointer:
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        return ctypes.cast(glMapBufferRange(GL_ARRAY_BUFFER, start, size, GL_MAP_WRITE_BIT), ptr_type).contents

    def unmap(self) -> None:
        glUnmapBuffer(GL_ARRAY_BUFFER)

    def delete(self) -> None:
        glDeleteBuffers(1, GLuint(self.id))
        self.id = None

    def __del__(self) -> None:
        if self.id is not None:
            try:
                self._context.delete_buffer(self.id)
                self.id = None
            except (AttributeError, ImportError):
                pass  # Interpreter is shutting down

    def resize(self, size: int) -> None:
        # Map, create a copy, then reinitialize.
        temp = (ctypes.c_byte * size)()

        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        data = glMapBufferRange(GL_ARRAY_BUFFER, 0, self.size, GL_MAP_READ_BIT)
        ctypes.memmove(temp, data, min(size, self.size))
        glUnmapBuffer(GL_ARRAY_BUFFER)

        self.size = size
        glBufferData(GL_ARRAY_BUFFER, self.size, temp, self.usage)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, size={self.size})"


class BackedBufferObject(BufferObject):
    """A buffer with system-memory backed store.

    Updates to the data via ``set_data`` and ``set_data_region`` will be held
    in system memory until ``commit`` is called.  The advantage is that fewer
    OpenGL calls are needed, which can increase performance at the expense of
    system memory.

    Dirty state is tracked as a min/max byte span PLUS a bounded set of the
    exact regions written since the last commit. With many objects sharing
    one large buffer (e.g. thousands of sprites in a batch), two small writes
    at opposite ends of the buffer previously forced an upload of nearly the
    entire buffer every frame. When only a few regions were written, commit
    now uploads just those regions; busy buffers (more than ``_max_spans``
    distinct regions) gracefully fall back to the single-span upload, so the
    write path stays as cheap as before under heavy load.
    """
    data: CTypesDataType
    data_ptr: int
    _dirty_min: int
    _dirty_max: int
    _dirty: bool
    stride: int
    count: int
    ctype: CTypesDataType

    # Track up to this many distinct written regions per commit cycle.
    # Beyond this, fall back to a single min..max span upload.
    _max_spans: int = 64
    # Merge recorded regions separated by less than this many bytes, so
    # commit issues a handful of medium uploads instead of many tiny ones.
    _merge_gap: int = 2048
    # More merged runs than this -> single span upload (protects against
    # pathological scatter producing too many GL calls).
    _max_uploads: int = 32

    def __init__(self, size: int, c_type: CTypesDataType, stride: int, count: int,  # noqa: D107
                 usage: int = GL_DYNAMIC_DRAW) -> None:
        super().__init__(size, usage)

        self.c_type = c_type
        self._ctypes_size = ctypes.sizeof(c_type)
        number = size // self._ctypes_size
        self.data = (c_type * number)()
        self.data_ptr = ctypes.addressof(self.data)

        self._dirty_min = sys.maxsize
        self._dirty_max = 0
        self._dirty = False
        self._dirty_spans = set()  # None means overflowed: use min..max span
        self._regions = {}         # (start, count) -> ctypes view

        self.stride = stride
        self.count = count

    def commit(self) -> None:
        """Commits all saved changes to the underlying buffer before drawing.

        Allows submitting multiple changes at once, rather than having to call
        glBufferSubData for every change. When few distinct regions were
        written, only those regions are uploaded.
        """
        if not self._dirty:
            return

        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        dirty_min = self._dirty_min
        size = self._dirty_max - dirty_min
        if size > 0:
            spans = self._dirty_spans
            runs = None
            if spans is not None and len(spans) > 1:
                # merge the recorded regions into contiguous upload runs
                merge_gap = self._merge_gap
                runs = []
                run_start = run_end = None
                for span_start, span_end in sorted(spans):
                    if run_end is None:
                        run_start, run_end = span_start, span_end
                    elif span_start - run_end <= merge_gap:
                        if span_end > run_end:
                            run_end = span_end
                    else:
                        runs.append((run_start, run_end))
                        run_start, run_end = span_start, span_end
                runs.append((run_start, run_end))
                if len(runs) > self._max_uploads:
                    runs = None
                else:
                    # if the runs cover most of the buffer anyway, prefer the
                    # single envelope upload (and its glBufferData orphan path)
                    # over multiple glBufferSubData calls into a buffer the GPU
                    # may still be reading from
                    total_bytes = 0
                    for run_start, run_end in runs:
                        total_bytes += run_end - run_start
                    if total_bytes * 2 >= self.size:
                        runs = None

            if runs is not None and len(runs) > 1:
                data_ptr = self.data_ptr
                for run_start, run_end in runs:
                    glBufferSubData(GL_ARRAY_BUFFER, run_start, run_end - run_start, data_ptr + run_start)
            elif size == self.size:
                glBufferData(GL_ARRAY_BUFFER, self.size, self.data, self.usage)
            else:
                glBufferSubData(GL_ARRAY_BUFFER, dirty_min, size, self.data_ptr + dirty_min)

            self._dirty_min = sys.maxsize
            self._dirty_max = 0
            self._dirty = False
            self._dirty_spans = set()

    def get_region(self, start: int, count: int) -> Array[CTypesDataType]:
        # per-instance cache (not lru_cache): a class-level cache keyed on
        # (self, start, count) pins every buffer ever cached, keeping dead
        # domains' buffers (and their persistent mappings) alive forever
        try:
            return self._regions[(start, count)]
        except KeyError:
            byte_start = self.stride * start  # byte offset
            array_count = self.count * count  # number of values
            ptr_type = ctypes.POINTER(self.c_type * array_count)
            region = ctypes.cast(self.data_ptr + byte_start, ptr_type).contents
            self._regions[(start, count)] = region
            return region

    def get_region_for_write(self, start: int, count: int) -> Array[CTypesDataType]:
        """Get a region view intended for the caller to write into.

        Equivalent to ``get_region`` followed by ``invalidate_region``; this
        sits behind every VertexList attribute access, so the dirty marking
        is inlined to save a call.
        """
        region = self.get_region(start, count)
        if count > 0:
            # replicated from self.invalidate_region
            byte_start = self.stride * start
            byte_end = byte_start + self.stride * count
            if byte_start < self._dirty_min:
                self._dirty_min = byte_start
            if byte_end > self._dirty_max:
                self._dirty_max = byte_end
            self._dirty = True
            spans = self._dirty_spans
            if spans is not None:
                if len(spans) < self._max_spans:
                    spans.add((byte_start, byte_end))
                else:
                    self._dirty_spans = None  # overflowed: min..max span upload
        return region

    def set_region(self, start: int, count: int, data: Sequence[float]) -> None:
        if count <= 0:
            return
        array_start = self.count * start
        array_end = self.count * count + array_start

        self.data[array_start:array_end] = data

        # replicated from self.invalidate_region
        byte_start = self.stride * start
        byte_end = byte_start + self.stride * count
        # As of Python 3.11, this is faster than min/max:
        if byte_start < self._dirty_min:
            self._dirty_min = byte_start
        if byte_end > self._dirty_max:
            self._dirty_max = byte_end
        self._dirty = True
        spans = self._dirty_spans
        if spans is not None:
            if len(spans) < self._max_spans:
                spans.add((byte_start, byte_end))
            else:
                self._dirty_spans = None  # overflowed: min..max span upload

    def resize(self, size: int) -> None:
        # size is the allocator size * attribute.stride
        number = size // ctypes.sizeof(self.c_type)
        data = (self.c_type * number)()
        ctypes.memmove(data, self.data, min(size, self.size))
        self.data = data
        self.data_ptr = ctypes.addressof(data)
        self.size = size

        # Set the dirty range to be the entire buffer.
        self._dirty_min = 0
        self._dirty_max = self.size
        self._dirty = True
        self._dirty_spans = None

        self._regions.clear()

    def invalidate(self) -> None:
        super().invalidate()
        # buffer storage was orphaned: everything must be re-uploaded
        self._dirty_min = 0
        self._dirty_max = self.size
        self._dirty = True
        self._dirty_spans = None

    def invalidate_region(self, start: int, count: int) -> None:
        if count <= 0:
            return
        byte_start = self.stride * start
        byte_end = byte_start + self.stride * count
        # As of Python 3.11, this is faster than min/max:
        if byte_start < self._dirty_min:
            self._dirty_min = byte_start
        if byte_end > self._dirty_max:
            self._dirty_max = byte_end
        self._dirty = True
        spans = self._dirty_spans
        if spans is not None:
            if len(spans) < self._max_spans:
                spans.add((byte_start, byte_end))
            else:
                self._dirty_spans = None  # overflowed: min..max span upload


class AttributeBufferObject(BackedBufferObject):
    """A backed buffer used for Shader Program attributes."""

    def __init__(self, size: int, attribute: Attribute) -> None:  # noqa: D107
        # size is the allocator size * attribute.stride (buffer size)
        super().__init__(size, attribute.c_type, attribute.stride, attribute.count)


class IndexedBufferObject(BackedBufferObject):
    """A backed buffer used for indices."""

    def __init__(self, size: int, c_type: CTypesDataType, stride: int, count: int,  # noqa: D107
                 usage: int = GL_DYNAMIC_DRAW) -> None:
        super().__init__(size, c_type, stride, count, usage)


class DrawFence:
    """Guards persistently mapped buffers against draws still in flight.

    A vertex domain using persistently mapped buffers arms this after issuing
    its draw commands. The first CPU write to any of the domain's buffers
    afterwards waits, guaranteeing the GPU has finished reading the mapped
    memory before it is modified.

    Arming is just a flag: the actual GL sync object is created lazily inside
    :py:meth:`wait`. Because GL commands execute in submission order, a fence
    inserted at wait time still signals only after every previously issued
    draw has completed, so the guarantee is identical, while domains that are
    drawn but never written between draws (static scenery, idle text) create
    no sync objects at all. This also means an armed fence holds no GL object
    that could leak when its domain is discarded.
    """

    __slots__ = ('armed',)

    # wait in 16.7ms slices, give up (and proceed) after ~150ms
    _WAIT_SLICE_NS = 16_666_666
    _MAX_WAIT_SLICES = 9

    def __init__(self) -> None:
        self.armed = False

    def arm(self) -> None:
        """Mark that draw commands reading the mapped buffers were issued."""
        self.armed = True

    def wait(self) -> None:
        """Block until all previously issued GL commands completed."""
        if not self.armed:
            return
        self.armed = False
        # inserted after the draws in the command stream, so waiting on it
        # waits for them; typically already signaled by the time we get here
        sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        try:
            # first wait flushes the command stream so the fence is
            # guaranteed to eventually signal
            result = glClientWaitSync(sync, GL_SYNC_FLUSH_COMMANDS_BIT, 0)
            slices = self._MAX_WAIT_SLICES
            while result == GL_TIMEOUT_EXPIRED and slices > 0:
                result = glClientWaitSync(sync, 0, self._WAIT_SLICE_NS)
                slices -= 1
            # GL_ALREADY_SIGNALED / GL_CONDITION_SATISFIED: done.
            # GL_WAIT_FAILED or timeout cap: proceed rather than hang; worst
            # case is a one-frame visual artifact on a wedged driver.
        finally:
            glDeleteSync(sync)


class PersistentBufferObject(AbstractBuffer):
    """A persistently mapped OpenGL buffer.

    Requires an OpenGL 4.4+ context, or ``GL_ARB_buffer_storage``.
    The buffer is mapped once at creation and stays mapped for its whole
    lifetime: reads and writes go directly to GPU-visible memory, so no
    commit/upload step is needed before drawing. The coherent mapping makes
    CPU writes automatically visible to subsequent GL commands.

    Read/write hazards against draws already issued but not yet executed
    are handled cooperatively with the owning vertex domain through the
    :py:class:`DrawFence` assigned to :py:attr:`fence`.

    .. warning:: Attribute regions view driver-owned mapped memory. Do not
        retain a region object (e.g. ``region = vlist.position``) across a
        buffer resize (any vertex list creation can grow the buffer) or
        deletion: the old mapping is unmapped and the retained view becomes
        invalid. Re-access the attribute property each time instead, as all
        of pyglet's own modules do.
    """

    #: Assigned by the owning vertex domain: a DrawFence shared by all the
    #: domain's buffers. May be None (no synchronization).
    fence: DrawFence | None = None

    def __init__(self, size, attribute, vao):
        # NOTE: Persistent buffers cannot be resized in place. On resize, a
        #       new buffer is created and the data copied over, so a
        #       reference to the attribute (and VAO) is required to re-point
        #       the attribute at the new buffer.

        self.size = size
        self.attribute = attribute
        self.stride = attribute.stride
        self.count = attribute.count
        self.c_type = attribute.c_type
        self.vao = vao

        self._context = pyglet.gl.current_context
        self._regions = {}  # (start, count) -> ctypes view

        # GL_MAP_READ_BIT keeps reads through the mapping defined behavior:
        # resize, VertexList.migrate and user code all read attribute regions.
        self.flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT

        buffer_id = GLuint()
        glGenBuffers(1, buffer_id)
        self.id = buffer_id.value
        glBindBuffer(GL_ARRAY_BUFFER, self.id)

        initial = (GLubyte * size)()
        glBufferStorage(GL_ARRAY_BUFFER, size, initial, self.flags)
        self._map_buffer()

    def _map_buffer(self) -> None:
        self.data = self._map_bound_buffer(self.size)
        self.data_ptr = ctypes.addressof(self.data)

    def _map_bound_buffer(self, size: int):
        # maps the buffer currently bound to GL_ARRAY_BUFFER; raises
        # ValueError (NULL pointer) if mapping failed
        number = size // ctypes.sizeof(self.c_type)
        ptr_type = ctypes.POINTER(self.c_type * number)
        return ctypes.cast(glMapBufferRange(GL_ARRAY_BUFFER, 0, size, self.flags), ptr_type).contents

    def set_data(self, data: Sequence[int] | CTypesPointer) -> None:
        raise NotImplementedError("Not yet implemented")

    def set_data_region(self, data: Sequence[int] | CTypesPointer, start: int, length: int) -> None:
        raise NotImplementedError("Not yet implemented")

    def bind(self, target=GL_ARRAY_BUFFER):
        glBindBuffer(target, self.id)

    def unbind(self):
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def map(self) -> CTypesPointer[ctypes.c_ubyte]:
        raise NotImplementedError("PersistentBufferObjects are always mapped.")

    def map_range(self, start, size, ptr_type, flags=GL_MAP_WRITE_BIT):
        raise NotImplementedError("PersistentBufferObjects are always mapped.")

    def unmap(self) -> None:
        raise NotImplementedError("PersistentBufferObjects cannot be unmapped.")

    def commit(self) -> None:
        """No-op: writes through the coherent mapping need no upload step."""

    def delete(self) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glUnmapBuffer(GL_ARRAY_BUFFER)
        glDeleteBuffers(1, GLuint(self.id))
        self.id = None

    def __del__(self) -> None:
        if self.id is not None:
            try:
                self._context.delete_buffer(self.id)
                self.id = None
            except (AttributeError, ImportError):
                pass  # Interpreter is shutting down

    def get_region(self, start, count):
        # per-instance cache: see BackedBufferObject.get_region
        try:
            return self._regions[(start, count)]
        except KeyError:
            byte_start = self.stride * start  # byte offset
            array_count = self.count * count  # number of values
            ptr_type = ctypes.POINTER(self.c_type * array_count)
            region = ctypes.cast(self.data_ptr + byte_start, ptr_type).contents
            self._regions[(start, count)] = region
            return region

    def get_region_for_write(self, start, count):
        """Get a region view intended for the caller to write into.

        Writes land directly in GPU-visible memory, so no dirty tracking is
        needed; instead, wait for any draw still reading this memory before
        handing out a writable view.
        """
        fence = self.fence
        if fence is not None and fence.armed:
            fence.wait()
        return self.get_region(start, count)

    def set_region(self, start, count, data):
        # wait for any draw still reading this memory before writing over it
        fence = self.fence
        if fence is not None and fence.armed:
            fence.wait()

        array_start = self.count * start
        array_end = self.count * count + array_start
        self.data[array_start:array_end] = data

    def resize(self, size):
        # The GPU may still be executing draws that read the old buffer;
        # deleting is safe (GL defers destruction), but wait on the fence so
        # the copy below observes settled memory.
        fence = self.fence
        if fence is not None and fence.armed:
            fence.wait()

        # Create a temporary system-memory copy of the current data
        temp = (GLubyte * size)()
        ctypes.memmove(temp, self.data, min(size, self.size))

        # Create and map the NEW buffer first: if allocation or mapping
        # fails, the old buffer/mapping stays fully intact and the raised
        # error leaves this object in a consistent state.
        buffer_id = GLuint()
        glGenBuffers(1, buffer_id)
        new_id = buffer_id.value
        try:
            glBindBuffer(GL_ARRAY_BUFFER, new_id)
            glBufferStorage(GL_ARRAY_BUFFER, size, temp, self.flags)
            new_data = self._map_bound_buffer(size)
        except Exception:
            glDeleteBuffers(1, GLuint(new_id))
            glBindBuffer(GL_ARRAY_BUFFER, self.id)
            raise

        # Success: retire the old mapping and buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glUnmapBuffer(GL_ARRAY_BUFFER)
        glDeleteBuffers(1, GLuint(self.id))

        self.id = new_id
        self.size = size
        self.data = new_data
        self.data_ptr = ctypes.addressof(new_data)

        # Re-point the vertex attribute at the new buffer
        self.vao.bind()
        glBindBuffer(GL_ARRAY_BUFFER, new_id)
        self.attribute.enable()
        self.attribute.set_pointer(0)
        if self.attribute.instance:
            self.attribute.set_divisor()
        self.vao.unbind()

        self._regions.clear()

    def sub_data(self):
        # Not necessary with persistent mapping
        pass

    def invalidate(self):
        # Not necessary with persistent mapping
        pass

    def invalidate_region(self, start, count):
        # Not necessary with persistent mapping
        pass
