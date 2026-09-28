# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare batched camera operations with independent pinhole calculations."""

import numpy as np
import pytest

from source.utils.geometry import geometric_utils_numpy as geometry


def camera_data(batch_shape, dtype, point_count=5):
    count = int(np.prod(batch_shape)) if batch_shape else 1
    ids = np.arange(count).reshape(batch_shape)
    intrinsics = np.broadcast_to(np.eye(3), batch_shape + (3, 3)).copy()
    intrinsics[..., 0, 0] = 400 + ids * 10
    intrinsics[..., 1, 1] = 450 + ids * 5
    intrinsics[..., 0, 1] = 3 + ids
    intrinsics[..., 0, 2] = 150 - ids
    intrinsics[..., 1, 2] = 120 + ids
    transforms = np.broadcast_to(np.eye(4), batch_shape + (4, 4)).copy()
    angle = 0.1 + ids * 0.05
    transforms[..., 0, 0] = transforms[..., 1, 1] = np.cos(angle)
    transforms[..., 0, 1] = -np.sin(angle)
    transforms[..., 1, 0] = np.sin(angle)
    transforms[..., :3, 3] = np.stack(
        [0.2 + ids * 0.03, -0.1 - ids * 0.02, 0.3 + ids * 0.01], axis=-1
    )
    points = (
        np.arange(count * point_count * 3).reshape(batch_shape + (point_count, 3))
        / 50.0
    )
    points[..., 2] += 2.0
    return tuple(x.astype(dtype) for x in (points, intrinsics, transforms))


def project_reference(points, intrinsics, transforms):
    result = np.empty(points.shape[:-1] + (2,), dtype=points.dtype)
    for index in np.ndindex(points.shape[:-2]):
        for number, point in enumerate(points[index]):
            transformed = transforms[index][:3, :3] @ point + transforms[index][:3, 3]
            x, y, z = transformed
            k = intrinsics[index]
            result[index + (number,)] = [
                (k[0, 0] * x + k[0, 1] * y) / z + k[0, 2],
                k[1, 1] * y / z + k[1, 2],
            ]
    return result


def backproject_reference(pixels, depths, intrinsics, transforms):
    result = np.empty(pixels.shape[:-1] + (3,), dtype=pixels.dtype)
    for index in np.ndindex(pixels.shape[:-2]):
        k = intrinsics[index]
        for number, (u, v) in enumerate(pixels[index]):
            y = (v - k[1, 2]) / k[1, 1]
            x = (u - k[0, 2] - k[0, 1] * y) / k[0, 0]
            point = np.array([x, y, 1], dtype=pixels.dtype) * depths[index + (number,)]
            result[index + (number,)] = (
                transforms[index][:3, :3] @ point + transforms[index][:3, 3]
            )
    return result


@pytest.mark.parametrize("batch_shape", [(), (1,), (2,), (3,), (4,), (2, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_projection_matches_independent_reference(batch_shape, dtype):
    points, k, transform = camera_data(batch_shape, dtype)
    before = [x.copy() for x in (points, k, transform)]
    actual = geometry.project(points, transform, k)
    expected = project_reference(points, k, transform)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=3e-5)
    assert actual.shape == points.shape[:-1] + (2,)
    assert actual.dtype == dtype
    for actual_input, original in zip((points, k, transform), before):
        np.testing.assert_array_equal(actual_input, original)


@pytest.mark.parametrize("batch_shape", [(), (1,), (2,), (3,), (4,), (2, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_backprojection_matches_independent_reference(batch_shape, dtype):
    points, k, transform = camera_data(batch_shape, dtype)
    pixels = (points[..., :2] * 30 + 100).astype(dtype)
    depth = points[..., 2]
    before = [x.copy() for x in (pixels, depth, k, transform)]
    actual = geometry.backproject_to_3d(pixels, depth, k, transform)
    expected = backproject_reference(pixels, depth, k, transform)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert actual.shape == points.shape
    assert actual.dtype == dtype
    for actual_input, original in zip((pixels, depth, k, transform), before):
        np.testing.assert_array_equal(actual_input, original)


@pytest.mark.parametrize("shared", ["intrinsics", "transform", "both"])
def test_camera_broadcasting_and_roundtrip(shared):
    points, k, transform = camera_data((2, 3), np.float64)
    if shared in ("intrinsics", "both"):
        k = k[0, 0]
    if shared in ("transform", "both"):
        transform = transform[0, 0]
    pixels = points[..., :2] * 20 + 100
    depth = points[..., 2]
    reconstructed = geometry.backproject_to_3d(pixels, depth, k, transform)
    recovered = geometry.project(reconstructed, np.linalg.inv(transform), k)
    np.testing.assert_allclose(recovered, pixels, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_empty_point_axis(batch_shape):
    points, k, transform = camera_data(batch_shape, np.float64, point_count=0)
    assert geometry.project(points, transform, k).shape == batch_shape + (0, 2)
    assert geometry.backproject_to_3d(
        points[..., :2], points[..., 2], k, transform
    ).shape == batch_shape + (0, 3)


def test_noncontiguous_batched_inputs():
    points, k, transform = camera_data((4,), np.float64)
    points, k, transform = points[::-2, ::-1], k[::-2], transform[::-2]
    actual = geometry.project(points, transform, k)
    np.testing.assert_allclose(
        actual, project_reference(points, k, transform), rtol=1e-12, atol=1e-12
    )


def test_unbatched_formula_is_exactly_preserved():
    points, k, transform = camera_data((), np.float64)
    transformed = geometry.from_homogeneous(
        geometry.to_homogeneous(points) @ transform.T
    )
    np.testing.assert_array_equal(
        geometry.project(points, transform, k),
        geometry.from_homogeneous(transformed @ k.T),
    )
    pixels, depth = points[..., :2], points[..., 2]
    old_backprojection = geometry.from_homogeneous(
        geometry.to_homogeneous(
            (geometry.to_homogeneous(pixels) @ np.linalg.inv(k).T) * depth[..., None]
        )
        @ transform.T
    )
    np.testing.assert_array_equal(
        geometry.backproject_to_3d(pixels, depth, k, transform), old_backprojection
    )
