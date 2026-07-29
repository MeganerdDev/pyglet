"""Instanced sprite renderer over a texture array (fork extension).

One shared unit quad is drawn N times with ``glDrawArraysInstanced``; every
sprite is a single instance slot carrying position, scale, rotation, color
and its texture-array frame (uv rect + layer + size + anchor). Compared to
``pyglet.sprite.Sprite`` this stores ~4x less attribute data per sprite,
creates/deletes sprites without vertex-list allocation, and always renders
in exactly one draw call per renderer regardless of how many distinct
images the sprites display (they must share one TextureArray).

Slots are managed with swap-remove: deleting a sprite moves the last
instance into the freed slot, so create/delete are O(1) and the instance
range stays dense for drawing.

Buffers use the same backed/persistent machinery as vertex domains: with
GL 4.4 (or ARB_buffer_storage) and pyglet.options.persistent_vertex_buffers
enabled, attribute writes land directly in mapped GPU memory guarded by a
DrawFence; otherwise dirty regions upload on draw.

Depth semantics match the game's DepthSprite: depth test enabled, fragments
with near-zero alpha discarded, so overlap resolves by the per-sprite z.

This module is intentionally fork-only (not part of upstream pyglet).
"""
from __future__ import annotations

import ctypes
from typing import Any

import pyglet
from pyglet.gl import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_DEPTH_TEST,
    GL_FLOAT,
    GL_LEQUAL,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_TRIANGLE_STRIP,
    GL_UNSIGNED_BYTE,
    glActiveTexture,
    GL_TEXTURE0,
    glBindTexture,
    glBlendFunc,
    glDepthFunc,
    glDisable,
    glDrawArraysInstanced,
    glEnable,
)
from pyglet.graphics import shader as shader_mod
from pyglet.graphics import vertexarray
from pyglet.graphics.vertexbuffer import AttributeBufferObject, DrawFence, PersistentBufferObject
from pyglet.graphics.vertexdomain import _persistent_buffers_supported

vertex_source = """#version 150 core
in vec2 corner;

in vec3 translate;
in vec2 scale;
in float rotation;
in vec4 colors;
in vec4 uv_rect;
in float layer;
in vec2 size;
in vec2 anchor;

out vec4 vertex_colors;
out vec3 texture_coords;

uniform WindowBlock
{
    mat4 projection;
    mat4 view;
} window;

void main()
{
    vec2 local = (corner * size - anchor) * scale;
    float c = cos(-radians(rotation));
    float s = sin(-radians(rotation));
    vec2 rotated = vec2(local.x * c - local.y * s, local.x * s + local.y * c);
    gl_Position = window.projection * window.view * vec4(rotated + translate.xy, translate.z, 1.0);

    vertex_colors = colors;
    texture_coords = vec3(mix(uv_rect.xy, uv_rect.zw, corner), layer);
}
"""

fragment_source = """#version 150 core
in vec4 vertex_colors;
in vec3 texture_coords;
out vec4 final_colors;

uniform sampler2DArray sprite_texture;

void main()
{
    final_colors = texture(sprite_texture, texture_coords) * vertex_colors;
    if (final_colors.a < 0.01) {
        discard;
    }
}
"""

# per-instance attributes: name -> (components, gl format char, normalize)
_INSTANCE_ATTRIBUTES = (
    ('translate', 3, 'f', False),
    ('scale', 2, 'f', False),
    ('rotation', 1, 'f', False),
    ('colors', 4, 'B', True),
    ('uv_rect', 4, 'f', False),
    ('layer', 1, 'f', False),
    ('size', 2, 'f', False),
    ('anchor', 2, 'f', False),
)


def _region_frame(region: Any) -> tuple:
    """Extract (u0, v0, u1, v1, layer, w, h, ax, ay) from a texture region.

    Works for TextureArrayRegion (layer in the r component of tex_coords)
    and plain Texture/TextureRegion (layer 0).
    """
    tc = region.tex_coords  # 12 floats: bl, br, tr, tl with (u, v, r)
    return (tc[0], tc[1], tc[6], tc[7], tc[2],
            float(region.width), float(region.height),
            float(region.anchor_x), float(region.anchor_y))


class InstancedSpriteRenderer:
    """Draws all its sprites with one glDrawArraysInstanced call."""

    def __init__(self, texture, group=None, capacity: int = 1024,
                 blend_src: int = GL_SRC_ALPHA, blend_dest: int = GL_ONE_MINUS_SRC_ALPHA,
                 program=None) -> None:
        """Create a renderer for sprites sharing ``texture``.

        Args:
            texture: a TextureArray (or any texture bindable at its target)
                shared by every sprite of this renderer.
            group: optional pyglet Group whose set_state/unset_state wrap the
                draw (e.g. a camera group). Not batched: call draw() yourself.
            capacity: initial instance capacity; grows automatically.
        """
        self.texture = texture
        self.group = group
        self.blend_src = blend_src
        self.blend_dest = blend_dest
        self.program = program or shader_mod.ShaderProgram(
            shader_mod.Shader(vertex_source, 'vertex'),
            shader_mod.Shader(fragment_source, 'fragment'),
        )

        self._capacity = max(16, capacity)
        self._sprites: list[InstancedSprite] = []

        self._use_persistent = _persistent_buffers_supported()
        self._fence = DrawFence() if self._use_persistent else None

        self.vao = vertexarray.VertexArray()
        self.vao.bind()

        # static unit-quad corners as a triangle strip: (0,0) (1,0) (0,1) (1,1)
        corner_location = self.program.attributes['corner']['location']
        corner_attr = shader_mod.Attribute('corner', corner_location, 2, GL_FLOAT, False, False)
        self._corner_buffer = AttributeBufferObject(corner_attr.stride * 4, corner_attr)
        self._corner_buffer.set_region(0, 4, (0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        self._corner_buffer.bind()
        corner_attr.enable()
        corner_attr.set_pointer(0)
        self._corner_buffer.commit()

        # per-instance attribute buffers (divisor 1)
        gl_types = {'f': GL_FLOAT, 'B': GL_UNSIGNED_BYTE}
        self.attributes = {}
        self.buffers = {}
        for name, count, fmt, normalize in _INSTANCE_ATTRIBUTES:
            location = self.program.attributes[name]['location']
            attribute = shader_mod.Attribute(name, location, count, gl_types[fmt], normalize, True)
            buffer = self._create_buffer(attribute)
            buffer.bind()
            attribute.enable()
            attribute.set_pointer(0)
            attribute.set_divisor()
            self.attributes[name] = attribute
            self.buffers[name] = buffer

        self.vao.unbind()

    def _create_buffer(self, attribute):
        size = attribute.stride * self._capacity
        if self._use_persistent:
            try:
                buffer = PersistentBufferObject(size, attribute, self.vao)
                buffer.fence = self._fence
                return buffer
            except Exception:  # noqa: BLE001
                self._use_persistent = False
                self._fence = None
        return AttributeBufferObject(size, attribute)

    # -- slot management ----------------------------------------------------

    def _grow(self) -> None:
        new_capacity = self._capacity * 2
        self.vao.bind()
        for name, buffer in self.buffers.items():
            buffer.resize(self.attributes[name].stride * new_capacity)
            if not isinstance(buffer, PersistentBufferObject):
                # persistent resize re-points its attribute itself; backed
                # buffers keep their GL id so pointers stay valid
                pass
        self.vao.unbind()
        self._capacity = new_capacity

    def create(self, region, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> InstancedSprite:
        """Create a sprite showing ``region`` (must belong to self.texture)."""
        if len(self._sprites) >= self._capacity:
            self._grow()
        slot = len(self._sprites)
        sprite = InstancedSprite(self, slot, region, x, y, z)
        self._sprites.append(sprite)
        return sprite

    def _free(self, sprite: InstancedSprite) -> None:
        slot = sprite._slot
        last = self._sprites[-1]
        if last is not sprite:
            # swap-remove: move the last instance's data into the freed slot
            last_slot = last._slot
            for buffer in self.buffers.values():
                data = tuple(buffer.get_region(last_slot, 1))
                buffer.set_region(slot, 1, data)
            last._slot = slot
            self._sprites[slot] = last
        self._sprites.pop()
        sprite._slot = -1

    def __len__(self) -> int:
        return len(self._sprites)

    # -- drawing ------------------------------------------------------------

    def draw(self) -> None:
        """Draw all live sprites in one instanced call."""
        count = len(self._sprites)
        if count == 0:
            return

        if self.group is not None:
            self.group.set_state_recursive()

        self.program.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.texture.target, self.texture.id)

        glEnable(GL_BLEND)
        glBlendFunc(self.blend_src, self.blend_dest)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        self.vao.bind()
        self._corner_buffer.commit()
        for buffer in self.buffers.values():
            buffer.commit()

        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, count)

        fence = self._fence
        if fence is not None:
            fence.arm()

        self.vao.unbind()
        glDisable(GL_DEPTH_TEST)
        self.program.stop()

        if self.group is not None:
            self.group.unset_state_recursive()

    def delete(self) -> None:
        for sprite in list(self._sprites):
            sprite._slot = -1
        self._sprites.clear()
        for buffer in self.buffers.values():
            buffer.delete()
        self._corner_buffer.delete()


class InstancedSprite:
    """A single instance slot of an InstancedSpriteRenderer.

    Exposes a Sprite-like surface: position/x/y/z, scale, scale_x/scale_y,
    rotation, opacity, color, visible, image, delete().
    """

    __slots__ = ('_renderer', '_slot', '_x', '_y', '_z', '_scale', '_scale_x', '_scale_y',
                 '_rotation', '_rgba', '_visible', '_region')

    def __init__(self, renderer: InstancedSpriteRenderer, slot: int, region,
                 x: float, y: float, z: float) -> None:
        self._renderer = renderer
        self._slot = slot
        self._x, self._y, self._z = x, y, z
        self._scale = 1.0
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._rotation = 0.0
        self._rgba = (255, 255, 255, 255)
        self._visible = True
        self._region = region

        buffers = renderer.buffers
        buffers['translate'].set_region(slot, 1, (x, y, z))
        buffers['scale'].set_region(slot, 1, (1.0, 1.0))
        buffers['rotation'].set_region(slot, 1, (0.0,))
        buffers['colors'].set_region(slot, 1, self._rgba)
        self._write_frame(region)

    def _write_frame(self, region) -> None:
        u0, v0, u1, v1, layer, w, h, ax, ay = _region_frame(region)
        slot = self._slot
        buffers = self._renderer.buffers
        buffers['uv_rect'].set_region(slot, 1, (u0, v0, u1, v1))
        buffers['layer'].set_region(slot, 1, (layer,))
        if self._visible:
            buffers['size'].set_region(slot, 1, (w, h))
        buffers['anchor'].set_region(slot, 1, (ax, ay))

    def _write_scale(self) -> None:
        self._renderer.buffers['scale'].set_region(
            self._slot, 1, (self._scale * self._scale_x, self._scale * self._scale_y))

    # -- geometry -----------------------------------------------------------

    @property
    def position(self) -> tuple:
        return (self._x, self._y, self._z)

    @position.setter
    def position(self, value: tuple) -> None:
        self._x, self._y, self._z = value
        self._renderer.buffers['translate'].set_region(self._slot, 1, value)

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        self._x = value
        self._renderer.buffers['translate'].set_region(self._slot, 1, (value, self._y, self._z))

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        self._y = value
        self._renderer.buffers['translate'].set_region(self._slot, 1, (self._x, value, self._z))

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, value: float) -> None:
        self._z = value
        self._renderer.buffers['translate'].set_region(self._slot, 1, (self._x, self._y, value))

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self._write_scale()

    @property
    def scale_x(self) -> float:
        return self._scale_x

    @scale_x.setter
    def scale_x(self, value: float) -> None:
        self._scale_x = value
        self._write_scale()

    @property
    def scale_y(self) -> float:
        return self._scale_y

    @scale_y.setter
    def scale_y(self, value: float) -> None:
        self._scale_y = value
        self._write_scale()

    @property
    def rotation(self) -> float:
        return self._rotation

    @rotation.setter
    def rotation(self, value: float) -> None:
        self._rotation = value
        self._renderer.buffers['rotation'].set_region(self._slot, 1, (value,))

    # -- appearance ---------------------------------------------------------

    @property
    def opacity(self) -> int:
        return self._rgba[3]

    @opacity.setter
    def opacity(self, value: int) -> None:
        r, g, b, _ = self._rgba
        self._rgba = (r, g, b, int(value))
        self._renderer.buffers['colors'].set_region(self._slot, 1, self._rgba)

    @property
    def color(self) -> tuple:
        return self._rgba

    @color.setter
    def color(self, rgba: tuple) -> None:
        r, g, b, *a = rgba
        self._rgba = (r, g, b, a[0] if a else 255)
        self._renderer.buffers['colors'].set_region(self._slot, 1, self._rgba)

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        if value == self._visible:
            return
        self._visible = value
        # zero size collapses the quad, mirroring Sprite's zeroed vertices
        if value:
            self._renderer.buffers['size'].set_region(
                self._slot, 1, (float(self._region.width), float(self._region.height)))
        else:
            self._renderer.buffers['size'].set_region(self._slot, 1, (0.0, 0.0))

    @property
    def width(self) -> float:
        return self._region.width * abs(self._scale * self._scale_x)

    @property
    def height(self) -> float:
        return self._region.height * abs(self._scale * self._scale_y)

    @property
    def image(self):
        return self._region

    @image.setter
    def image(self, region) -> None:
        self._region = region
        self._write_frame(region)

    def delete(self) -> None:
        if self._slot >= 0:
            self._renderer._free(self)  # noqa: SLF001
