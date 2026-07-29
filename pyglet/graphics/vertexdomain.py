"""Manage related vertex attributes within a single vertex domain.

A vertex "domain" consists of a set of attribute descriptions that together
describe the layout of one or more vertex buffers which are used together to
specify the vertices in a primitive.  Additionally, the domain manages the
buffers used to store the data and will resize them as necessary to accommodate
new vertices.

Domains can optionally be indexed, in which case they also manage a buffer
containing vertex indices.  This buffer is grown separately and has no size
relation to the attribute buffers.

Applications can create vertices (and optionally, indices) within a domain
with the :py:meth:`VertexDomain.create` method.  This returns a
:py:class:`VertexList` representing the list of vertices created.  The vertex
attribute data within the group can be modified, and the changes will be made
to the underlying buffers automatically.

The entire domain can be efficiently drawn in one step with the
:py:meth:`VertexDomain.draw` method, assuming all the vertices comprise
primitives of the same OpenGL primitive mode.
"""
from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any, NoReturn, Sequence, Type

from _ctypes import Array, _Pointer, _SimpleCData

from pyglet.gl.gl import (
    GL_BYTE,
    GL_DOUBLE,
    GL_FLOAT,
    GL_INT,
    GL_SHORT,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_INT,
    GL_UNSIGNED_SHORT,
    GLint,
    GLintptr,
    GLsizei,
    GLvoid,
    glDrawArrays,
    glDrawArraysInstanced,
    glDrawElements,
    glDrawElementsInstanced,
    glMultiDrawArrays,
    glMultiDrawElements,
)
import pyglet
from pyglet.graphics import allocation, shader, vertexarray
from pyglet.graphics.vertexbuffer import (
    AttributeBufferObject,
    DrawFence,
    IndexedBufferObject,
    PersistentBufferObject,
)

CTypesDataType = Type[_SimpleCData]
CTypesPointer = _Pointer

if TYPE_CHECKING:
    from pyglet.graphics.allocation import Allocator
    from pyglet.graphics.shader import Attribute
    from pyglet.graphics.vertexarray import VertexArray


def _nearest_pow2(v: int) -> int:
    # From http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    # Credit: Sean Anderson
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1


_c_types = {
    GL_BYTE: ctypes.c_byte,
    GL_UNSIGNED_BYTE: ctypes.c_ubyte,
    GL_SHORT: ctypes.c_short,
    GL_UNSIGNED_SHORT: ctypes.c_ushort,
    GL_INT: ctypes.c_int,
    GL_UNSIGNED_INT: ctypes.c_uint,
    GL_FLOAT: ctypes.c_float,
    GL_DOUBLE: ctypes.c_double,
}

_gl_types = {
    'b': GL_BYTE,
    'B': GL_UNSIGNED_BYTE,
    'h': GL_SHORT,
    'H': GL_UNSIGNED_SHORT,
    'i': GL_INT,
    'I': GL_UNSIGNED_INT,
    'f': GL_FLOAT,
    'd': GL_DOUBLE,
}


def _persistent_buffers_supported() -> bool:
    """Whether persistently mapped vertex buffers can be used on this context."""
    if not getattr(pyglet.options, 'persistent_vertex_buffers', True):
        return False
    from pyglet.gl import gl_info
    if not gl_info.have_context():
        return False
    return gl_info.have_version(4, 4) or gl_info.have_extension('GL_ARB_buffer_storage')


def _make_persistent_attribute_property(name: str, buffer: PersistentBufferObject,  # noqa: ARG001
                                        fence: DrawFence) -> property:
    # Persistent mapping: writes land directly in GPU-visible memory, so no
    # dirty tracking is needed. The fence guards the first write after a draw
    # against the GPU still reading the mapped memory.
    get_region = buffer.get_region
    set_region = buffer.set_region

    def _attribute_getter(self: VertexList) -> Array[float | int]:
        if fence.armed:
            fence.wait()
        return get_region(self.start, self.count)

    def _attribute_setter(self: VertexList, data: Any) -> None:
        set_region(self.start, self.count, data)

    return property(_attribute_getter, _attribute_setter)


def _make_attribute_property(name: str, buffer: AttributeBufferObject) -> property:  # noqa: ARG001
    # NOTE: the buffer is captured directly instead of looked up through
    #       self.domain on every access: this is the hottest path in pyglet
    #       for sprite/shape/text heavy applications (every attribute write
    #       goes through the getter). VertexList.migrate reassigns the
    #       instance __class__ so migrated lists bind to the new domain's
    #       buffers. The buffer object itself is stable for the domain's
    #       lifetime (resize mutates it in place).
    get_region = buffer.get_region
    set_region = buffer.set_region
    stride = buffer.stride
    max_spans = buffer._max_spans  # noqa: SLF001

    def _attribute_getter(self: VertexList) -> Array[float | int]:
        start = self.start
        count = self.count
        region = get_region(start, count)
        # inline of buffer.invalidate_region (saves a call per write)
        if count > 0:
            byte_start = stride * start
            byte_end = byte_start + stride * count
            if byte_start < buffer._dirty_min:  # noqa: SLF001
                buffer._dirty_min = byte_start  # noqa: SLF001
            if byte_end > buffer._dirty_max:  # noqa: SLF001
                buffer._dirty_max = byte_end  # noqa: SLF001
            buffer._dirty = True  # noqa: SLF001
            spans = buffer._dirty_spans  # noqa: SLF001
            if spans is not None:
                if len(spans) < max_spans:
                    spans.add((byte_start, byte_end))
                else:
                    buffer._dirty_spans = None  # noqa: SLF001
        return region

    def _attribute_setter(self: VertexList, data: Any) -> None:
        set_region(self.start, self.count, data)

    return property(_attribute_getter, _attribute_setter)


class VertexList:
    """A list of vertices within a :py:class:`VertexDomain`.

    Use :py:meth:`VertexDomain.create` to construct this list.
    """
    count: int
    start: int
    domain: VertexDomain | InstancedVertexDomain
    indexed: bool = False
    instanced: bool = False
    initial_attribs: dict

    def __init__(self, domain: VertexDomain, start: int, count: int) -> None:  # noqa: D107
        self.domain = domain
        self.start = start
        self.count = count
        self.initial_attribs = domain.attribute_meta

    def draw(self, mode: int) -> None:
        """Draw this vertex list in the given OpenGL mode.

        Args:
            mode:
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.
        """
        self.domain.draw_subset(mode, self)

    def resize(self, count: int, index_count: int | None = None) -> None:  # noqa: ARG002
        """Resize this group.

        Args:
            count:
                New number of vertices in the list.
            index_count:
                Ignored for non indexed VertexDomains

        """
        new_start = self.domain.safe_realloc(self.start, self.count, count)
        if new_start != self.start:
            # Copy contents to new location
            for buffer in self.domain.attrib_name_buffers.values():
                old_data = buffer.get_region(self.start, self.count)
                buffer.set_region(new_start, self.count, old_data)
        self.start = new_start
        self.count = count

    def delete(self) -> None:
        """Delete this group."""
        self.domain.allocator.dealloc(self.start, self.count)

    def set_instance_source(self, domain: InstancedVertexDomain, instance_attributes: Sequence[str]) -> None:
        assert self.instanced is False, "Vertex list is already an instance."
        assert list(domain.attribute_names.keys()) == list(self.domain.attribute_names.keys()), \
            'Domain attributes must match.'

        new_start = domain.safe_alloc(self.count)
        for key, current_buffer in self.domain.attrib_name_buffers.items():
            new_buffer = domain.attrib_name_buffers[key]
            old_data = current_buffer.get_region(self.start, self.count)
            if key in instance_attributes:
                attrib = domain.attribute_names[key]
                count = 1
                old_data = old_data[:attrib.count]
            else:
                count = self.count
            new_buffer.set_region(new_start, count, old_data)

        self.domain.allocator.dealloc(self.start, self.count)
        self.domain = domain
        # rebind attribute properties to the new domain's buffers
        self.__class__ = domain._vertexlist_class  # noqa: SLF001
        self.start = new_start
        self.instanced = True

    def migrate(self, domain: VertexDomain | InstancedVertexDomain) -> None:
        """Move this group from its current domain and add to the specified one.

        Attributes on domains must match.
        (In practice, used to change parent state of some vertices).

        Args:
            domain:
                Domain to migrate this vertex list to.

        """
        assert list(domain.attribute_names.keys()) == list(self.domain.attribute_names.keys()), \
            'Domain attributes must match.'

        new_start = domain.safe_alloc(self.count)
        for name, old_buffer in self.domain.attrib_name_buffers.items():
            new_buffer = domain.attrib_name_buffers[name]
            old_data = old_buffer.get_region(self.start, self.count)
            new_buffer.set_region(new_start, self.count, old_data)

        self.domain.allocator.dealloc(self.start, self.count)
        self.domain = domain
        # rebind attribute properties to the new domain's buffers
        self.__class__ = domain._vertexlist_class  # noqa: SLF001
        self.start = new_start

    def set_attribute_data(self, name: str, data: Any) -> None:
        buffer = self.domain.attrib_name_buffers[name]
        count = self.count
        try:
            # set_region marks dirty state (backed buffers) or waits on the
            # draw fence (persistently mapped buffers) as appropriate
            buffer.set_region(self.start, count, data)
        except ValueError:
            expected = buffer.count * count
            msg = f"Invalid data size for '{name}'. Expected {expected}, got {len(data)}."
            raise ValueError(msg) from None

    def add_instance(self, **kwargs: Any) -> VertexInstance:
        assert self.instanced
        self.domain._instances += 1  # noqa: SLF001

        instance_id = self.domain._instances  # noqa: SLF001

        start = self.domain.safe_alloc_instance(3)

        for buffer, attribute in self.domain.buffer_attributes:
            if attribute.instance:
                assert attribute.name in kwargs, (f"{attribute.name} is defined as an instance attribute, "
                                                  f"keyword argument not found.")
                buffer.set_region(instance_id - 1, 1, kwargs[attribute.name])

        return self.domain._vertexinstance_class(self, instance_id, start)  # noqa: SLF001

    def delete_instance(self, instance: VertexInstance) -> None:
        assert self.instanced
        if instance.id != self.domain._instances:  # noqa: SLF001
            msg = "Only the last instance added can be removed."
            raise Exception(msg)

        self.domain._instances -= 1  # noqa: SLF001

        self.domain.instance_allocator.dealloc(instance.start, 3)


class IndexedVertexList(VertexList):
    """A list of vertices within an :py:class:`IndexedVertexDomain` that are indexed.

    Use :py:meth:`IndexedVertexDomain.create` to construct this list.
    """
    domain: IndexedVertexDomain | InstancedIndexedVertexDomain
    indexed: bool = True

    index_count: int
    index_start: int

    def __init__(self, domain: IndexedVertexDomain, start: int, count: int, index_start: int,  # noqa: D107
                 index_count: int) -> None:
        super().__init__(domain, start, count)
        self.index_start = index_start
        self.index_count = index_count

    def delete(self) -> None:
        """Delete this group."""
        super().delete()
        self.domain.index_allocator.dealloc(self.index_start, self.index_count)

    def migrate(self, domain: IndexedVertexDomain | InstancedIndexedVertexDomain) -> None:
        """Move this group from its current indexed domain and add to the specified one.

        Attributes on domains must match.  (In practice, used
        to change parent state of some vertices).

        Args:
            domain:
                Indexed domain to migrate this vertex list to.
        """
        old_start = self.start
        old_domain = self.domain
        super().migrate(domain)

        # Note: this code renumber the indices of the *original* domain
        # because the vertices are in a new position in the new domain
        if old_start != self.start:
            diff = self.start - old_start
            old_indices = old_domain.index_buffer.get_region(self.index_start, self.index_count)
            old_domain.index_buffer.set_region(self.index_start, self.index_count, [i + diff for i in old_indices])

        # copy indices to new domain
        old_array = old_domain.index_buffer.get_region(self.index_start, self.index_count)
        # must delloc before calling safe_index_alloc or else problems when same
        # batch is migrated to because index_start changes after dealloc
        old_domain.index_allocator.dealloc(self.index_start, self.index_count)

        new_start = self.domain.safe_index_alloc(self.index_count)
        self.domain.index_buffer.set_region(new_start, self.index_count, old_array)

        self.index_start = new_start

    def set_instance_source(self, domain: IndexedVertexDomain | InstancedIndexedVertexDomain,
                            instance_attributes: Sequence[str]) -> None:
        assert self.instanced is False, "IndexedVertexList is already an instance."
        old_start = self.start
        old_domain = self.domain
        super().set_instance_source(domain, instance_attributes)

        assert list(domain.attribute_names.keys()) == list(self.domain.attribute_names.keys()), \
            'Domain attributes must match.'

        # Note: this code renumber the indices of the *original* domain
        # because the vertices are in a new position in the new domain
        if old_start != self.start:
            diff = self.start - old_start
            old_indices = old_domain.index_buffer.get_region(self.index_start, self.index_count)
            old_domain.index_buffer.set_region(self.index_start, self.index_count, [i + diff for i in old_indices])

        # copy indices to new domain
        old_array = old_domain.index_buffer.get_region(self.index_start, self.index_count)
        # must delloc before calling safe_index_alloc or else problems when same
        # batch is migrated to because index_start changes after dealloc
        old_domain.index_allocator.dealloc(self.index_start, self.index_count)

        new_start = self.domain.safe_index_alloc(self.index_count)
        self.domain.index_buffer.set_region(new_start, self.index_count, old_array)

        self.index_start = new_start

    @property
    def indices(self) -> list[int]:
        """Array of index data."""
        start = self.start
        return [i - start for i in self.domain.index_buffer.get_region(self.index_start, self.index_count)]

    @indices.setter
    def indices(self, data: Sequence[int]) -> None:
        start = self.start
        # The vertex data is offset in the buffer, so offset the index values to match. Ex:
        # vertex_buffer: [_, _, _, _, 1, 2, 3, 4]
        self.domain.index_buffer.set_region(self.index_start, self.index_count, tuple(i + start for i in data))


class VertexInstance:
    id: int
    start: int
    _vertex_list: VertexList | IndexedVertexList

    def __init__(self, vertex_list: VertexList | IndexedVertexList, instance_id: int, start: int) -> None:
        self.id = instance_id
        self.start = start
        self._vertex_list = vertex_list

    @property
    def domain(self) -> InstancedVertexDomain | InstancedIndexedVertexDomain:
        return self._vertex_list.domain

    def delete(self) -> None:
        self._vertex_list.delete_instance(self)
        self._vertex_list = None


class VertexDomain:
    """Management of a set of vertex lists.

    Construction of a vertex domain is usually done with the
    :py:func:`create_domain` function.
    """

    attribute_meta: dict[str, dict[str, Any]]
    allocator: Allocator
    buffer_attributes: list[tuple[AttributeBufferObject, Attribute]]
    vao: VertexArray
    attribute_names: dict[str, Attribute]
    attrib_name_buffers: dict[str, AttributeBufferObject]

    _property_dict: dict[str, property]
    _vertexlist_class: type

    _initial_count: int = 16
    _vertex_class: type[VertexList] = VertexList

    # draw-array cache, rebuilt only when the allocator map changes
    _draw_cache_version: int = -1
    _draw_primcount: int = 0
    _draw_starts_gl = None
    _draw_sizes_gl = None

    # persistent-mapping support: instanced subclasses opt out
    _allow_persistent: bool = True
    _fence: DrawFence | None = None

    def __init__(self, attribute_meta: dict[str, dict[str, Any]]) -> None:  # noqa: D107
        self.attribute_meta = attribute_meta
        self.allocator = allocation.Allocator(self._initial_count)
        self.vao = vertexarray.VertexArray()

        self.attribute_names = {}  # name: attribute
        self.buffer_attributes = []  # list of (buffer, attribute)
        self.attrib_name_buffers = {}  # dict of AttributeName: AttributeBufferObject (for VertexLists)

        self._property_dict = {}  # name: property(_getter, _setter)

        # Persistently mapped buffers eliminate the commit/upload step, but
        # require GL 4.4 (or ARB_buffer_storage) and are not wired up for
        # instanced domains. Falls back to backed buffers when unavailable.
        use_persistent = (self._allow_persistent
                          and not any(meta['instance'] for meta in attribute_meta.values())
                          and _persistent_buffers_supported())
        if use_persistent:
            self._fence = DrawFence()

        for name, meta in attribute_meta.items():
            assert meta['format'][0] in _gl_types, f"'{meta['format']}' is not a valid attribute format for '{name}'."
            location = meta['location']
            count = meta['count']
            gl_type = _gl_types[meta['format'][0]]
            normalize = 'n' in meta['format']
            instanced = meta['instance']

            self.attribute_names[name] = attribute = shader.Attribute(name, location, count, gl_type, normalize,
                                                                      instanced)

            # Create buffer:
            buffer = None
            if use_persistent:
                try:
                    buffer = PersistentBufferObject(attribute.stride * self.allocator.capacity, attribute, self.vao)
                    buffer.fence = self._fence
                    self._property_dict[attribute.name] = _make_persistent_attribute_property(
                        name, buffer, self._fence)
                except Exception:  # noqa: BLE001
                    # capability probe passed but creation failed (driver
                    # quirk): fall back to backed buffers from here on
                    use_persistent = False
                    buffer = None

            if buffer is None:
                buffer = AttributeBufferObject(attribute.stride * self.allocator.capacity, attribute)
                self._property_dict[attribute.name] = _make_attribute_property(name, buffer)

            self.attrib_name_buffers[name] = buffer
            self.buffer_attributes.append((buffer, attribute))

        # drop the fence if the persistent fallback left no persistent buffers
        if self._fence is not None and not any(
                isinstance(b, PersistentBufferObject) for b, _ in self.buffer_attributes):
            self._fence = None

        # Make a custom VertexList class w/ properties for each attribute in the ShaderProgram:
        self._vertexlist_class = type(self._vertex_class.__name__, (self._vertex_class,), self._property_dict)

        self.vao.bind()
        for buffer, attribute in self.buffer_attributes:
            buffer.bind()
            attribute.enable()
            attribute.set_pointer(buffer.ptr)
            if attribute.instance:
                attribute.set_divisor()
        self.vao.unbind()

    def safe_alloc(self, count: int) -> int:
        """Allocate vertices, resizing the buffers if necessary."""
        try:
            return self.allocator.alloc(count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            for buffer, _ in self.buffer_attributes:
                buffer.resize(capacity * buffer.stride)
            self.allocator.set_capacity(capacity)
            return self.allocator.alloc(count)

    def safe_realloc(self, start: int, count: int, new_count: int) -> int:
        """Reallocate vertices, resizing the buffers if necessary."""
        try:
            return self.allocator.realloc(start, count, new_count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            for buffer, _ in self.buffer_attributes:
                buffer.resize(capacity * buffer.stride)
            self.allocator.set_capacity(capacity)
            return self.allocator.realloc(start, count, new_count)

    def create(self, count: int, index_count: int | None = None) -> VertexList:  # noqa: ARG002
        """Create a :py:class:`VertexList` in this domain.

        Args:
            count:
                Number of vertices to create.
            index_count:
                Ignored for non indexed VertexDomains
        """
        start = self.safe_alloc(count)
        return self._vertexlist_class(self, start, count)

    def draw(self, mode: int) -> None:
        """Draw all vertices in the domain.

        All vertices in the domain are drawn at once. This is the
        most efficient way to render primitives.

        Args:
            mode:
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.

        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.commit()

        # rebuild the draw arrays only when allocations changed; with a
        # stable scene this is pure cache-hit every frame
        allocator = self.allocator
        if self._draw_cache_version != allocator.version:
            starts, sizes = allocator.get_allocated_regions()
            primcount = len(starts)
            self._draw_primcount = primcount
            if primcount > 1:
                self._draw_starts_gl = (GLint * primcount)(*starts)
                self._draw_sizes_gl = (GLsizei * primcount)(*sizes)
            elif primcount == 1:
                self._draw_starts_gl = starts[0]
                self._draw_sizes_gl = sizes[0]
            self._draw_cache_version = allocator.version

        primcount = self._draw_primcount
        if primcount == 0:
            pass
        elif primcount == 1:
            # Common case
            glDrawArrays(mode, self._draw_starts_gl, self._draw_sizes_gl)
        else:
            glMultiDrawArrays(mode, self._draw_starts_gl, self._draw_sizes_gl, primcount)

        fence = self._fence
        if fence is not None:
            fence.arm()

    def draw_subset(self, mode: int, vertex_list: VertexList) -> None:
        """Draw a specific VertexList in the domain.

        The `vertex_list` parameter specifies a :py:class:`VertexList`
        to draw. Only primitives in that list will be drawn.

        Args:
            mode:
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.
            vertex_list:
                Vertex list to draw.

        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.commit()

        glDrawArrays(mode, vertex_list.start, vertex_list.count)

        fence = self._fence
        if fence is not None:
            fence.arm()

    @property
    def is_empty(self) -> bool:
        return not self.allocator.starts

    def delete(self) -> None:
        """Release this domain's GL resources (buffers and VAO).

        Called by the Batch when an emptied domain is discarded. Without
        this, teardown depends on the cyclic garbage collector (every domain
        creates a dynamic VertexList class, which is a reference cycle), so
        applications with tuned or frozen gc accumulate dead domains' GL
        buffers -- including persistent mappings -- indefinitely.
        """
        for buffer, _ in self.buffer_attributes:
            try:
                buffer.delete()
            except Exception:  # noqa: BLE001
                pass
        self.buffer_attributes.clear()
        self.attrib_name_buffers.clear()
        try:
            self.vao.delete()
        except Exception:  # noqa: BLE001
            pass

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}@{id(self):x} {self.allocator}>'


def _make_instance_attribute_property(name: str) -> property:
    def _attribute_getter(self: VertexInstance) -> Array[CTypesDataType]:
        buffer = self.domain.attrib_name_buffers[name]
        region = buffer.get_region(self.id - 1, 1)
        buffer.invalidate_region(self.id - 1, 1)
        return region

    def _attribute_setter(self: VertexInstance, data: Any) -> None:
        buffer = self.domain.attrib_name_buffers[name]
        buffer.set_region(self.id - 1, 1, data)

    return property(_attribute_getter, _attribute_setter)


def _make_restricted_instance_attribute_property(name: str) -> property:
    def _attribute_getter(self: VertexInstance) -> Array[CTypesDataType]:
        buffer = self.domain.attrib_name_buffers[name]
        return buffer.get_region(self.id - 1, 1)

    def _attribute_setter(_self: VertexInstance, _data: Any) -> NoReturn:
        msg = f"Attribute '{name}' is not an instanced attribute."
        raise Exception(msg)

    return property(_attribute_getter, _attribute_setter)


class InstancedVertexDomain(VertexDomain):  # noqa: D101
    _allow_persistent = False  # persistent mapping is not wired for instancing
    instance_allocator: Allocator
    _instances: int
    _instance_properties: dict[str, property]
    _vertexinstance_class: type

    def __init__(self, attribute_meta: dict[str, dict[str, Any]]) -> None:
        super().__init__(attribute_meta)
        self._instances = 1
        self.instance_allocator = allocation.Allocator(self._initial_count)

        self._instance_properties = {}
        for name, attribute in self.attribute_names.items():
            if attribute.instance:
                self._instance_properties[name] = _make_instance_attribute_property(name)
            else:
                self._instance_properties[name] = _make_restricted_instance_attribute_property(name)

        self._vertexinstance_class = type('VertexInstance', (VertexInstance,), self._instance_properties)

    def safe_alloc_instance(self, count: int) -> int:
        try:
            return self.instance_allocator.alloc(count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            for buffer, attribute in self.buffer_attributes:
                if attribute.instance:
                    buffer.resize(capacity * buffer.stride)
            self.instance_allocator.set_capacity(capacity)
            return self.instance_allocator.alloc(count)

    def safe_alloc(self, count: int) -> int:
        """Allocate vertices, resizing the buffers if necessary."""
        try:
            return self.allocator.alloc(count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            for buffer, _ in self.buffer_attributes:
                buffer.resize(capacity * buffer.stride)
            self.allocator.set_capacity(capacity)
            return self.allocator.alloc(count)

    def safe_realloc(self, start: int, count: int, new_count: int) -> int:
        """Reallocate vertices, resizing the buffers if necessary."""
        try:
            return self.allocator.realloc(start, count, new_count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            for buffer, _ in self.buffer_attributes:
                buffer.resize(capacity * buffer.stride)
            self.allocator.set_capacity(capacity)
            return self.allocator.realloc(start, count, new_count)

    def draw(self, mode: int) -> None:
        """Draw all vertices in the domain.

        All vertices in the domain are drawn at once. This is the
        most efficient way to render primitives.

        Args:
            mode:
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.

        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.commit()

        starts, sizes = self.allocator.get_allocated_regions()
        glDrawArraysInstanced(mode, starts[0], sizes[0], self._instances)

    def draw_subset(self, mode: int, vertex_list: VertexList) -> None:
        """Draw a specific VertexList in the domain.

        The `vertex_list` parameter specifies a :py:class:`VertexList`
        to draw. Only primitives in that list will be drawn.

        Args:
            mode:
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.
            vertex_list:
                Vertex list to draw.
        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.commit()

        glDrawArraysInstanced(mode, vertex_list.start, vertex_list.count, self._instances)

    @property
    def is_empty(self) -> bool:
        return not self.allocator.starts


class IndexedVertexDomain(VertexDomain):
    """Management of a set of indexed vertex lists.

    Construction of an indexed vertex domain is usually done with the
    :py:func:`create_domain` function.
    """
    index_allocator: Allocator
    index_gl_type: int
    index_c_type: CTypesDataType
    index_element_size: int
    index_buffer: IndexedBufferObject
    _initial_index_count = 16
    _vertex_class = IndexedVertexList

    def __init__(self, attribute_meta: dict[str, dict[str, Any]],  # noqa: D107
                 index_gl_type: int = GL_UNSIGNED_INT) -> None:
        super().__init__(attribute_meta)

        self.index_allocator = allocation.Allocator(self._initial_index_count)

        self.index_gl_type = index_gl_type
        self.index_c_type = shader._c_types[index_gl_type]  # noqa: SLF001
        self.index_element_size = ctypes.sizeof(self.index_c_type)
        self.index_buffer = IndexedBufferObject(self.index_allocator.capacity * self.index_element_size,
                                                shader._c_types[index_gl_type],
                                                self.index_element_size,
                                                1)

        self.vao.bind()
        self.index_buffer.bind_to_index_buffer()
        self.vao.unbind()

        # Make a custom VertexList class w/ properties for each attribute in the ShaderProgram:
        self._vertexlist_class = type(self._vertex_class.__name__, (self._vertex_class,),
                                      self._property_dict)

    def safe_index_alloc(self, count: int) -> int:
        """Allocate indices, resizing the buffers if necessary."""
        try:
            return self.index_allocator.alloc(count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            self.index_buffer.resize(capacity * self.index_element_size)
            self.index_allocator.set_capacity(capacity)
            return self.index_allocator.alloc(count)

    def safe_index_realloc(self, start: int, count: int, new_count: int) -> int:
        """Reallocate indices, resizing the buffers if necessary."""
        try:
            return self.index_allocator.realloc(start, count, new_count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            self.index_buffer.resize(capacity * self.index_element_size)
            self.index_allocator.set_capacity(capacity)
            return self.index_allocator.realloc(start, count, new_count)

    def delete(self) -> None:
        """Release GL resources including the index buffer."""
        super().delete()
        try:
            self.index_buffer.delete()
        except Exception:  # noqa: BLE001
            pass

    def create(self, count: int, index_count: int) -> IndexedVertexList:
        """Create an :py:class:`IndexedVertexList` in this domain.

        Args:
            count:
                Number of vertices to create
            index_count:
                Number of indices to create

        """
        start = self.safe_alloc(count)
        index_start = self.safe_index_alloc(index_count)
        return self._vertexlist_class(self, start, count, index_start, index_count)

    def draw(self, mode: int) -> None:
        """Draw all vertices in the domain.

        All vertices in the domain are drawn at once. This is the
        most efficient way to render primitives.

        Args:
            mode:
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.

        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.commit()

        self.index_buffer.commit()

        # rebuild the draw arrays only when index allocations changed; the
        # pointer/size ctypes arrays for glMultiDrawElements are expensive
        # to construct per frame on fragmented domains
        allocator = self.index_allocator
        if self._draw_cache_version != allocator.version:
            starts, sizes = allocator.get_allocated_regions()
            primcount = len(starts)
            self._draw_primcount = primcount
            if primcount > 1:
                starts = [s * self.index_element_size + self.index_buffer.ptr for s in starts]
                self._draw_starts_gl = (ctypes.POINTER(GLvoid) * primcount)(*(GLintptr * primcount)(*starts))
                self._draw_sizes_gl = (GLsizei * primcount)(*sizes)
            elif primcount == 1:
                self._draw_starts_gl = self.index_buffer.ptr + starts[0] * self.index_element_size
                self._draw_sizes_gl = sizes[0]
            self._draw_cache_version = allocator.version

        primcount = self._draw_primcount
        if primcount == 0:
            pass
        elif primcount == 1:
            # Common case
            glDrawElements(mode, self._draw_sizes_gl, self.index_gl_type, self._draw_starts_gl)
        else:
            glMultiDrawElements(mode, self._draw_sizes_gl, self.index_gl_type, self._draw_starts_gl, primcount)

        fence = self._fence
        if fence is not None:
            fence.arm()

    def draw_subset(self, mode: int, vertex_list: IndexedVertexList) -> None:
        """Draw a specific IndexedVertexList in the domain.

        The `vertex_list` parameter specifies a :py:class:`IndexedVertexList`
        to draw. Only primitives in that list will be drawn.

        Args:
            mode:
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.
            vertex_list:
                Vertex list to draw.
        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.commit()

        self.index_buffer.commit()

        glDrawElements(mode, vertex_list.index_count, self.index_gl_type,
                       self.index_buffer.ptr +
                       vertex_list.index_start * self.index_element_size)

        fence = self._fence
        if fence is not None:
            fence.arm()


class InstancedIndexedVertexDomain(IndexedVertexDomain, InstancedVertexDomain):
    """Management of a set of indexed vertex lists.

    Construction of an indexed vertex domain is usually done with the
    :py:func:`create_domain` function.
    """
    _initial_index_count: int = 16

    def __init__(self, attribute_meta: dict[str, dict[str, Any]],  # noqa: D107
                 index_gl_type: int = GL_UNSIGNED_INT) -> None:
        super().__init__(attribute_meta, index_gl_type)

    def safe_index_alloc(self, count: int) -> int:
        """Allocate indices, resizing the buffers if necessary.

        Returns:
            The starting index of the allocated region.
        """
        try:
            return self.index_allocator.alloc(count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            self.index_buffer.resize(capacity * self.index_element_size)
            self.index_allocator.set_capacity(capacity)
            return self.index_allocator.alloc(count)

    def safe_index_realloc(self, start: int, count: int, new_count: int) -> int:
        """Reallocate indices, resizing the buffers if necessary."""
        try:
            return self.index_allocator.realloc(start, count, new_count)
        except allocation.AllocatorMemoryException as e:
            capacity = _nearest_pow2(e.requested_capacity)
            self.index_buffer.resize(capacity * self.index_element_size)
            self.index_allocator.set_capacity(capacity)
            return self.index_allocator.realloc(start, count, new_count)

    def create(self, count: int, index_count: int) -> IndexedVertexList:
        """Create an :py:class:`IndexedVertexList` in this domain.

        Args:
            count:
                Number of vertices to create
            index_count:
                Number of indices to create

        """
        start = self.safe_alloc(count)
        index_start = self.safe_index_alloc(index_count)
        return self._vertexlist_class(self, start, count, index_start, index_count)

    def draw(self, mode: int) -> None:
        """Draw all vertices in the domain.

        All vertices in the domain are drawn at once. This is the
        most efficient way to render primitives.

        Args:
            mode:
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.

        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.commit()

        starts, sizes = self.index_allocator.get_allocated_regions()
        glDrawElementsInstanced(mode, sizes[0], self.index_gl_type,
                                self.index_buffer.ptr + starts[0] * self.index_element_size, self._instances)

    def draw_subset(self, mode: int, vertex_list: IndexedVertexList) -> None:
        """Draw a specific IndexedVertexList in the domain.

        The ``vertex_list`` parameter specifies a :py:class:`IndexedVertexList`
        to draw. Only primitives in that list will be drawn.

        Args:
            mode:
                OpenGL drawing mode, e.g. ``GL_POINTS``, ``GL_LINES``, etc.
            vertex_list:
                Vertex list to draw.

        """
        self.vao.bind()
        for buffer, _ in self.buffer_attributes:
            buffer.commit()

        glDrawElementsInstanced(mode, vertex_list.index_count, self.index_gl_type,
                                self.index_buffer.ptr +
                                vertex_list.index_start * self.index_element_size, self._instances)
