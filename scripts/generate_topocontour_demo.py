#!/usr/bin/env python3
"""Generate an easy-to-read TopoContour-Mamba routing schematic.

The script intentionally depends only on NumPy and Pillow.  It implements the
same fixed-budget contour sampling rule described in the paper:

1. build a random, smoothed scalar heatmap;
2. extract closed level-set contours with marching squares;
3. give each contour N=max(N_min, round(perimeter/delta_0)) final points;
4. place 2N uniform pilots and read one Sobel field;
5. use the mean signed normal response to choose one low-value side;
6. smooth |response| with (1,2,3,2,1)/9, clip density to [1,3],
   and redistribute exactly N points by weighted cumulative arc length;
7. use the strongest smoothed pilot as the common start/return point p0.

The output is a single paper-ready heatmap panel.  It highlights all p0 points,
draws every collision-to-contour normal read, marks both loop directions, and
uses only a compact English symbol legend beside the image.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VIRIDIS_STOPS = np.array(
    [
        [0.00, 68, 1, 84],
        [0.25, 59, 82, 139],
        [0.50, 33, 145, 140],
        [0.75, 94, 201, 98],
        [1.00, 253, 231, 37],
    ],
    dtype=float,
)

LEVEL_COLORS = [
    (105, 224, 255),
    (92, 196, 255),
    (106, 149, 255),
    (164, 112, 255),
    (237, 91, 175),
    (255, 116, 92),
    (255, 190, 72),
]


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def smooth2d(array: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel1d(sigma)
    radius = len(kernel) // 2
    padded_x = np.pad(array, ((0, 0), (radius, radius)), mode="reflect")
    tmp = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded_x)
    padded_y = np.pad(tmp, ((radius, radius), (0, 0)), mode="reflect")
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded_y)


def random_heatmap(size: int, seed: int) -> np.ndarray:
    """Create a reproducible multi-peak field with occasional basins."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx / (size - 1)) * 2.0 - 1.0
    y = (yy / (size - 1)) * 2.0 - 1.0
    field = np.zeros((size, size), dtype=float)

    for _ in range(7):
        cx, cy = rng.uniform(-0.68, 0.68, size=2)
        sx, sy = rng.uniform(0.16, 0.42, size=2)
        angle = rng.uniform(0.0, math.pi)
        ca, sa = math.cos(angle), math.sin(angle)
        xr = ca * (x - cx) + sa * (y - cy)
        yr = -sa * (x - cx) + ca * (y - cy)
        amp = rng.uniform(0.65, 1.35)
        field += amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))

    # Negative blobs create bowl-like regions without using a special case.
    for _ in range(2):
        cx, cy = rng.uniform(-0.45, 0.45, size=2)
        sigma = rng.uniform(0.12, 0.23)
        amp = rng.uniform(0.28, 0.55)
        field -= amp * np.exp(-0.5 * (((x - cx) / sigma) ** 2 + ((y - cy) / sigma) ** 2))

    field += 0.10 * smooth2d(rng.normal(size=(size, size)), 3.2)
    field -= 0.40 * (x * x + y * y)
    field = smooth2d(field, 1.35)

    # A smooth low-valued border closes the contours without creating an
    # artificial one-pixel Sobel wall that would attract every p0.
    lo, hi = np.percentile(field, [1.0, 99.0])
    field = np.clip((field - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    edge_distance = np.minimum.reduce([xx, yy, size - 1 - xx, size - 1 - yy]) / (size - 1)
    ramp = np.clip(edge_distance / 0.11, 0.0, 1.0)
    envelope = ramp * ramp * (3.0 - 2.0 * ramp)
    field = smooth2d(field * envelope, 0.9)
    return np.clip(field / max(float(field.max()), 1e-12), 0.0, 1.0)


def sobel_field(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(field, 1, mode="edge")
    gx = (
        -p[:-2, :-2] + p[:-2, 2:]
        - 2.0 * p[1:-1, :-2] + 2.0 * p[1:-1, 2:]
        - p[2:, :-2] + p[2:, 2:]
    ) / 8.0
    gy = (
        -p[:-2, :-2] - 2.0 * p[:-2, 1:-1] - p[:-2, 2:]
        + p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]
    ) / 8.0
    return gx, gy


def bilinear(array: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = array.shape
    x = np.clip(points[:, 0], 0.0, w - 1.000001)
    y = np.clip(points[:, 1], 0.0, h - 1.000001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    return (
        array[y0, x0] * (1 - fx) * (1 - fy)
        + array[y0, x1] * fx * (1 - fy)
        + array[y1, x0] * (1 - fx) * fy
        + array[y1, x1] * fx * fy
    )


def _edge_point(values: list[float], x: int, y: int, edge: int, level: float) -> tuple[float, float]:
    v0, v1, v2, v3 = values  # top-left, top-right, bottom-right, bottom-left
    if edge == 0:
        t = (level - v0) / max(abs(v1 - v0), 1e-12) * (1 if v1 >= v0 else -1)
        t = np.clip(t, 0.0, 1.0)
        return x + float(t), float(y)
    if edge == 1:
        t = (level - v1) / max(abs(v2 - v1), 1e-12) * (1 if v2 >= v1 else -1)
        t = np.clip(t, 0.0, 1.0)
        return float(x + 1), y + float(t)
    if edge == 2:
        t = (level - v2) / max(abs(v3 - v2), 1e-12) * (1 if v3 >= v2 else -1)
        t = np.clip(t, 0.0, 1.0)
        return x + 1.0 - float(t), float(y + 1)
    t = (level - v3) / max(abs(v0 - v3), 1e-12) * (1 if v0 >= v3 else -1)
    t = np.clip(t, 0.0, 1.0)
    return float(x), y + 1.0 - float(t)


def marching_squares(field: np.ndarray, level: float) -> list[np.ndarray]:
    """Return closed contours as arrays of (x, y) route-grid coordinates."""
    h, w = field.shape
    table = {
        1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
        6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)], 9: [(0, 2)],
        11: [(1, 2)], 12: [(1, 3)], 13: [(0, 1)], 14: [(3, 0)],
    }
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for y in range(h - 1):
        for x in range(w - 1):
            vals = [field[y, x], field[y, x + 1], field[y + 1, x + 1], field[y + 1, x]]
            case = sum((1 << i) for i, value in enumerate(vals) if value >= level)
            if case in (0, 15):
                continue
            if case == 5:
                pairs = [(0, 1), (2, 3)] if np.mean(vals) >= level else [(3, 0), (1, 2)]
            elif case == 10:
                pairs = [(0, 3), (1, 2)] if np.mean(vals) >= level else [(0, 1), (2, 3)]
            else:
                pairs = table.get(case, [])
            for e0, e1 in pairs:
                segments.append((_edge_point(vals, x, y, e0, level), _edge_point(vals, x, y, e1, level)))

    def key(point: tuple[float, float]) -> tuple[float, float]:
        return round(point[0], 6), round(point[1], 6)

    adjacency: dict[tuple[float, float], list[tuple[float, float]]] = {}
    coords: dict[tuple[float, float], tuple[float, float]] = {}
    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for a, b in segments:
        ka, kb = key(a), key(b)
        if ka == kb:
            continue
        coords[ka], coords[kb] = a, b
        adjacency.setdefault(ka, []).append(kb)
        adjacency.setdefault(kb, []).append(ka)
        edges.append((ka, kb))

    visited: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    loops: list[np.ndarray] = []
    for first_a, first_b in edges:
        edge_id = tuple(sorted((first_a, first_b)))
        if edge_id in visited:
            continue
        start, current, previous = first_a, first_a, None
        chain = [coords[start]]
        closed = False
        while True:
            candidates = []
            for neighbor in adjacency.get(current, []):
                candidate_edge = tuple(sorted((current, neighbor)))
                if candidate_edge not in visited:
                    candidates.append(neighbor)
            if not candidates:
                break
            neighbor = candidates[0]
            if len(candidates) > 1 and previous is not None:
                nonback = [candidate for candidate in candidates if candidate != previous]
                if nonback:
                    neighbor = nonback[0]
            visited.add(tuple(sorted((current, neighbor))))
            previous, current = current, neighbor
            chain.append(coords[current])
            if current == start:
                closed = True
                break
            if len(chain) > len(edges) + 2:
                break
        if closed and len(chain) >= 5:
            loops.append(np.asarray(chain[:-1], dtype=float))
    return loops


def polygon_area(poly: np.ndarray) -> float:
    q = np.roll(poly, -1, axis=0)
    return 0.5 * float(np.sum(poly[:, 0] * q[:, 1] - q[:, 0] * poly[:, 1]))


def point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    """Even-odd containment test used only to mark retained branch terminals."""
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = poly[-1]
    for current in poly:
        x0, y0 = float(previous[0]), float(previous[1])
        x1, y1 = float(current[0]), float(current[1])
        if (y0 > y) != (y1 > y):
            crossing_x = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def perimeter(poly: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1).sum())


def orient_clockwise(poly: np.ndarray) -> np.ndarray:
    # In image coordinates (y grows down), positive shoelace area is clockwise.
    return poly if polygon_area(poly) > 0 else poly[::-1].copy()


def sample_closed(poly: np.ndarray, count: int) -> np.ndarray:
    next_poly = np.roll(poly, -1, axis=0)
    lengths = np.linalg.norm(next_poly - poly, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = cumulative[-1]
    targets = np.arange(count, dtype=float) * total / count
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, len(poly) - 1)
    frac = (targets - cumulative[indices]) / np.maximum(lengths[indices], 1e-12)
    return poly[indices] + frac[:, None] * (next_poly[indices] - poly[indices])


def contour_sampling(
    poly: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    delta0: float,
    n_min: int,
) -> dict[str, object]:
    poly = orient_clockwise(poly)
    p = perimeter(poly)
    n_final = max(n_min, int(round(p / delta0)))
    pilots = sample_closed(poly, 2 * n_final)
    tangent = np.roll(pilots, -1, axis=0) - np.roll(pilots, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    response = bilinear(gx, pilots) * normals[:, 0] + bilinear(gy, pilots) * normals[:, 1]
    side_sign = -1.0 if float(response.mean()) >= 0.0 else 1.0
    magnitude = np.abs(response)
    smooth = (
        np.roll(magnitude, 2)
        + 2.0 * np.roll(magnitude, 1)
        + 3.0 * magnitude
        + 2.0 * np.roll(magnitude, -1)
        + np.roll(magnitude, -2)
    ) / 9.0
    positive = smooth[smooth > 1e-12]
    if len(positive) == 0:
        weights = np.ones_like(smooth)
    else:
        weights = np.clip(smooth / (float(np.median(positive)) + 1e-12), 1.0, 3.0)

    maximum = float(smooth.max())
    candidates = np.flatnonzero(np.isclose(smooth, maximum, rtol=0.0, atol=1e-12))
    p0_index = min(candidates, key=lambda i: (pilots[i, 1], pilots[i, 0]))
    order = np.concatenate([np.arange(p0_index, len(pilots)), np.arange(0, p0_index)])
    pilots = pilots[order]
    weights = weights[order]
    smooth = smooth[order]

    next_pilots = np.roll(pilots, -1, axis=0)
    interval_length = np.linalg.norm(next_pilots - pilots, axis=1)
    mass = interval_length * (weights + np.roll(weights, -1)) * 0.5
    cumulative = np.concatenate([[0.0], np.cumsum(mass)])
    targets = np.arange(n_final, dtype=float) * cumulative[-1] / n_final
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, len(pilots) - 1)
    frac = (targets - cumulative[indices]) / np.maximum(mass[indices], 1e-12)
    final_points = pilots[indices] + frac[:, None] * (next_pilots[indices] - pilots[indices])

    return {
        "poly": poly,
        "perimeter": p,
        "pilots": pilots,
        "strength": smooth,
        "weights": weights,
        "points": final_points,
        "p0": final_points[0],
        "side_sign": side_sign,
        "n_final": n_final,
    }


def heatmap_rgb(field: np.ndarray) -> np.ndarray:
    flat = field.ravel()
    result = np.empty((flat.size, 3), dtype=float)
    for channel in range(3):
        result[:, channel] = np.interp(flat, VIRIDIS_STOPS[:, 0], VIRIDIS_STOPS[:, channel + 1])
    return np.clip(result.reshape(field.shape + (3,)), 0, 255).astype(np.uint8)


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_star(draw: ImageDraw.ImageDraw, center: tuple[float, float], radius: float, fill, outline=(70, 28, 15)) -> None:
    cx, cy = center
    vertices = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.43
        vertices.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(vertices, fill=fill, outline=outline)


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color, width=5, head=13) -> None:
    sx, sy = start
    ex, ey = end
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(ey - sy, ex - sx)
    left = (ex - head * math.cos(angle - 0.55), ey - head * math.sin(angle - 0.55))
    right = (ex - head * math.cos(angle + 0.55), ey - head * math.sin(angle + 0.55))
    draw.polygon([end, left, right], fill=color)


def point_on_closed(poly: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.roll(poly, -1, axis=0)
    lengths = np.linalg.norm(q - poly, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    target = (fraction % 1.0) * cumulative[-1]
    i = min(np.searchsorted(cumulative, target, side="right") - 1, len(poly) - 1)
    t = (target - cumulative[i]) / max(lengths[i], 1e-12)
    point = poly[i] + t * (q[i] - poly[i])
    tangent = (q[i] - poly[i]) / max(lengths[i], 1e-12)
    return point, tangent


def ray_segment_hit(origin: np.ndarray, direction: np.ndarray, a: np.ndarray, b: np.ndarray, min_t=0.8):
    edge = b - a
    denominator = direction[0] * edge[1] - direction[1] * edge[0]
    if abs(denominator) < 1e-10:
        return None
    delta = a - origin
    t = (delta[0] * edge[1] - delta[1] * edge[0]) / denominator
    u = (delta[0] * direction[1] - delta[1] * direction[0]) / denominator
    if t > min_t and -1e-8 <= u <= 1.0 + 1e-8:
        return float(t)
    return None


def first_collision(origin: np.ndarray, direction: np.ndarray, records: list[dict], size: int) -> np.ndarray:
    best = float("inf")
    for record in records:
        poly = record["poly"]
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            hit = ray_segment_hit(origin, direction, a, b)
            if hit is not None and hit < best:
                best = hit
    frame = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=float)
    for a, b in zip(frame, np.roll(frame, -1, axis=0)):
        hit = ray_segment_hit(origin, direction, a, b)
        if hit is not None and hit < best:
            best = hit
    return origin + best * direction if np.isfinite(best) else origin + 8.0 * direction


def transform(points: np.ndarray, box: tuple[float, float, float, float], bounds: tuple[float, float, float, float]) -> np.ndarray:
    left, top, right, bottom = box
    xmin, ymin, xmax, ymax = bounds
    sx = (right - left) / max(xmax - xmin, 1e-9)
    sy = (bottom - top) / max(ymax - ymin, 1e-9)
    scale = min(sx, sy)
    ox = left + (right - left - (xmax - xmin) * scale) / 2.0
    oy = top + (bottom - top - (ymax - ymin) * scale) / 2.0
    output = np.empty_like(points, dtype=float)
    output[:, 0] = ox + (points[:, 0] - xmin) * scale
    output[:, 1] = oy + (points[:, 1] - ymin) * scale
    return output


def _render_explainer(field: np.ndarray, levels: np.ndarray, records: list[dict], output: Path, seed: int) -> None:
    width, height = 2000, 1180
    image = Image.new("RGB", (width, height), (246, 247, 249))
    draw = ImageDraw.Draw(image)
    title_font = find_font(44, bold=True)
    heading_font = find_font(29, bold=True)
    body_font = find_font(23)
    small_font = find_font(18)
    tiny_font = find_font(16)

    draw.text((70, 34), "TopoContour-Mamba：从热力图生成图像自己的扫描路线", font=title_font, fill=(24, 31, 45))
    draw.text((71, 88), f"随机种子 {seed}｜低显著性先读，高显著性后读｜每个圈独立并行", font=body_font, fill=(80, 88, 102))

    main_box = (70, 150, 1120, 1120)
    draw.rounded_rectangle(main_box, radius=24, fill=(255, 255, 255), outline=(211, 216, 225), width=2)
    draw.text((95, 170), "1  等高线、变密度采样点与共同起止点", font=heading_font, fill=(25, 31, 44))
    map_left, map_top, map_size = 112, 230, 885
    heat = Image.fromarray(heatmap_rgb(field), mode="RGB").resize((map_size, map_size), Image.Resampling.BICUBIC)
    image.paste(heat, (map_left, map_top))
    draw = ImageDraw.Draw(image)
    scale = map_size / (field.shape[0] - 1)

    def map_points(points: np.ndarray) -> list[tuple[float, float]]:
        return [(map_left + float(x) * scale, map_top + float(y) * scale) for x, y in points]

    level_to_index = {float(level): i for i, level in enumerate(levels)}
    for record in records:
        color = LEVEL_COLORS[level_to_index[float(record["level"])] % len(LEVEL_COLORS)]
        line = map_points(record["poly"])
        draw.line(line + [line[0]], fill=(23, 31, 44), width=7, joint="curve")
        draw.line(line + [line[0]], fill=color, width=3, joint="curve")
        for px, py in map_points(record["points"]):
            r = 2.8
            draw.ellipse((px - r, py - r, px + r, py + r), fill=(255, 255, 255), outline=(20, 28, 40), width=1)
        p0 = map_points(np.asarray(record["p0"])[None, :])[0]
        draw_star(draw, p0, 10, (255, 221, 64))

    # Highest retained level, largest loop: a readable terminal example.
    max_level = max(float(record["level"]) for record in records)
    terminal_candidates = [record for record in records if float(record["level"]) == max_level]
    selected = max(terminal_candidates, key=lambda record: float(record["perimeter"]))
    selected_line = map_points(selected["poly"])
    draw.line(selected_line + [selected_line[0]], fill=(255, 255, 255), width=8, joint="curve")
    draw.line(selected_line + [selected_line[0]], fill=(235, 54, 124), width=4, joint="curve")
    selected_p0 = map_points(np.asarray(selected["p0"])[None, :])[0]
    draw_star(draw, selected_p0, 14, (255, 235, 82), outline=(145, 35, 55))
    draw.text((selected_p0[0] + 15, selected_p0[1] - 32), "p0", font=body_font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(35, 27, 40))

    # Low-to-high scalar direction beside the heatmap.
    bar_x0, bar_x1 = 1025, 1056
    for i in range(220):
        value = 1.0 - i / 219.0
        color = tuple(int(v) for v in heatmap_rgb(np.array([[value]]))[0, 0])
        y0 = 360 + i * 2
        draw.rectangle((bar_x0, y0, bar_x1, y0 + 2), fill=color)
    draw.text((1012, 325), "高", font=body_font, fill=(32, 38, 49))
    draw.text((1012, 808), "低", font=body_font, fill=(32, 38, 49))
    draw_arrow(draw, (1041, 790), (1041, 385), (250, 250, 250), width=4, head=12)
    draw.text((1011, 845), "树汇总\n低 → 高", font=small_font, fill=(32, 38, 49), spacing=6)

    # Legend placed below the map, not on top of the data.
    legend_y = 1133
    draw.ellipse((120, legend_y - 5, 130, legend_y + 5), fill=(255, 255, 255), outline=(28, 35, 48))
    draw.text((140, legend_y - 14), "最终采样点（同圈总数固定）", font=small_font, fill=(42, 49, 61))
    draw_star(draw, (480, legend_y), 9, (255, 221, 64))
    draw.text((497, legend_y - 14), "p0：双向扫描共同起点与带 Tag 返回点", font=small_font, fill=(42, 49, 61))

    right_box = (1150, 150, 1930, 755)
    draw.rounded_rectangle(right_box, radius=24, fill=(255, 255, 255), outline=(211, 216, 225), width=2)
    draw.text((1180, 172), "2  放大一个终止圈：局部读取顺序", font=heading_font, fill=(25, 31, 44))

    selected_poly = np.asarray(selected["poly"])
    selected_points = np.asarray(selected["points"])
    tangent = np.roll(selected_points, -1, axis=0) - np.roll(selected_points, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    idx = max(1, len(selected_points) // 7)
    p_read = selected_points[idx]
    ordinary_dir = float(selected["side_sign"]) * normals[idx]
    opposite_dir = -ordinary_dir
    q_ord = first_collision(p_read, ordinary_dir, records, field.shape[0])
    q_opp = first_collision(p_read, opposite_dir, records, field.shape[0])

    all_zoom = np.vstack([selected_poly, q_ord[None, :], q_opp[None, :]])
    xmin, ymin = all_zoom.min(axis=0)
    xmax, ymax = all_zoom.max(axis=0)
    pad = 0.12 * max(xmax - xmin, ymax - ymin)
    bounds = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    zoom_box = (1190, 245, 1890, 610)
    zpoly = transform(selected_poly, zoom_box, bounds)
    zpoints = transform(selected_points, zoom_box, bounds)
    zp0 = transform(np.asarray(selected["p0"])[None, :], zoom_box, bounds)[0]
    zread = transform(p_read[None, :], zoom_box, bounds)[0]
    zq_ord = transform(q_ord[None, :], zoom_box, bounds)[0]
    zq_opp = transform(q_opp[None, :], zoom_box, bounds)[0]

    path = [tuple(point) for point in zpoly]
    draw.line(path + [path[0]], fill=(28, 34, 45), width=6, joint="curve")
    for point in zpoints:
        radius = 4.2
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=(255, 255, 255), outline=(42, 49, 61), width=1)
    draw_star(draw, tuple(zp0), 14, (255, 220, 52), outline=(147, 35, 57))
    draw.text((zp0[0] + 14, zp0[1] - 27), "p0 / Tag", font=small_font, fill=(112, 27, 53))

    # Opposite loop directions are marked at separate arc positions to avoid overlap.
    for fraction in (0.18, 0.58):
        a, _ = point_on_closed(zpoly, fraction)
        b, _ = point_on_closed(zpoly, fraction + 0.035)
        draw_arrow(draw, tuple(a), tuple(b), (24, 126, 208), width=6, head=12)
    for fraction in (0.36, 0.78):
        a, _ = point_on_closed(zpoly, fraction)
        b, _ = point_on_closed(zpoly, fraction - 0.035)
        draw_arrow(draw, tuple(a), tuple(b), (137, 76, 210), width=6, head=12)

    draw_arrow(draw, tuple(zq_ord), tuple(zread), (238, 133, 34), width=6, head=14)
    draw_arrow(draw, tuple(zq_opp), tuple(zread), (221, 63, 74), width=6, head=14)
    for point, fill in ((zq_ord, (238, 133, 34)), (zq_opp, (221, 63, 74)), (zread, (245, 245, 245))):
        draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=fill, outline=(30, 34, 44), width=2)

    legend_x, legend_y2 = 1188, 633
    draw.line((legend_x, legend_y2, legend_x + 34, legend_y2), fill=(24, 126, 208), width=6)
    draw.text((legend_x + 44, legend_y2 - 14), "顺时针 Contour Mamba", font=small_font, fill=(43, 49, 61))
    draw.line((1510, legend_y2, 1544, legend_y2), fill=(137, 76, 210), width=6)
    draw.text((1554, legend_y2 - 14), "逆时针 Contour Mamba", font=small_font, fill=(43, 49, 61))
    draw.line((legend_x, legend_y2 + 38, legend_x + 34, legend_y2 + 38), fill=(238, 133, 34), width=6)
    draw.text((legend_x + 44, legend_y2 + 24), "普通侧：碰撞点 → 轮廓点", font=small_font, fill=(43, 49, 61))
    draw.line((1510, legend_y2 + 38, 1544, legend_y2 + 38), fill=(221, 63, 74), width=6)
    draw.text((1554, legend_y2 + 24), "终止圈补读相反侧", font=small_font, fill=(43, 49, 61))
    draw.text((1190, 705), "同一 p0 开始；沿两个方向完整绕圈；最后回到带 Tag 的 p0。", font=body_font, fill=(38, 44, 56))

    flow_box = (1150, 785, 1930, 1120)
    draw.rounded_rectangle(flow_box, radius=24, fill=(255, 255, 255), outline=(211, 216, 225), width=2)
    draw.text((1180, 808), "3  一张图读懂完整流程", font=heading_font, fill=(25, 31, 44))
    steps = [
        ("热力图 S", "只负责生成路线"),
        ("闭合等高线", "同层多个圈并行"),
        ("Sobel 三合一", "方向 · 密度 · p0"),
        ("局部独立编码", "法线 + 双向闭环"),
        ("等高线树", "低显著性 → 高显著性"),
    ]
    x_positions = [1190, 1337, 1484, 1631, 1778]
    for i, ((name, note), x0) in enumerate(zip(steps, x_positions)):
        box = (x0, 875, x0 + 125, 1010)
        fill = (239, 245, 255) if i < 4 else (255, 242, 224)
        draw.rounded_rectangle(box, radius=14, fill=fill, outline=(151, 168, 193), width=2)
        bbox = draw.textbbox((0, 0), name, font=small_font)
        draw.text((x0 + (125 - (bbox[2] - bbox[0])) / 2, 897), name, font=small_font, fill=(28, 35, 48))
        # Two short centered lines fit more reliably than paragraph wrapping.
        note_parts = note.split(" ", 1) if " " in note else [note]
        if len(note_parts) == 1 and len(note) > 8:
            note_parts = [note[: len(note) // 2], note[len(note) // 2 :]]
        for j, part in enumerate(note_parts[:2]):
            part_box = draw.textbbox((0, 0), part, font=tiny_font)
            draw.text((x0 + (125 - (part_box[2] - part_box[0])) / 2, 942 + 22 * j), part, font=tiny_font, fill=(77, 86, 101))
        if i < len(steps) - 1:
            draw_arrow(draw, (x0 + 128, 942), (x_positions[i + 1] - 4, 942), (80, 92, 111), width=4, head=10)

    draw.text((1190, 1040), "关键：自适应采样只移动固定数量的点；父圈输出不进入局部法线或闭环扫描。", font=small_font, fill=(52, 59, 72))

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=96)


def render(field: np.ndarray, levels: np.ndarray, records: list[dict], output: Path, seed: int) -> None:
    """Render one heatmap panel plus a compact English symbol legend."""
    del seed, levels  # The field already shows the random instance and scalar ordering.
    width, height = 1900, 1220
    image = Image.new("RGB", (width, height), (255, 255, 255))
    map_left, map_top, map_size = 45, 45, 1120
    heat = Image.fromarray(heatmap_rgb(field), mode="RGB").resize((map_size, map_size), Image.Resampling.BICUBIC)
    image.paste(heat, (map_left, map_top))
    draw = ImageDraw.Draw(image)
    draw.rectangle((map_left, map_top, map_left + map_size, map_top + map_size), outline=(42, 47, 57), width=2)
    scale = map_size / (field.shape[0] - 1)

    def map_one(point: np.ndarray) -> tuple[float, float]:
        return map_left + float(point[0]) * scale, map_top + float(point[1]) * scale

    def map_many(points: np.ndarray) -> list[tuple[float, float]]:
        return [map_one(point) for point in points]

    # A light red wash makes terminal interiors visible without hiding the
    # heatmap.  Non-terminal regions receive no fill overlay.
    terminal_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    terminal_draw = ImageDraw.Draw(terminal_layer)
    for record in records:
        if bool(record["terminal"]):
            terminal_draw.polygon(map_many(np.asarray(record["poly"])), fill=(255, 48, 70, 34))
    image = Image.alpha_composite(image.convert("RGBA"), terminal_layer).convert("RGB")

    # Rays are placed beneath contours.  Every final sample emits its ordinary
    # ray; terminal contours additionally emit the opposite-side ray.  Both are
    # drawn in the actual Normal-Mamba order, from first collision q back to p.
    ray_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ray_draw = ImageDraw.Draw(ray_layer)
    for record in records:
        points = np.asarray(record["points"])
        tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        ordinary_direction = float(record["side_sign"]) * normals
        is_terminal = bool(record["terminal"])
        for point, direction in zip(points, ordinary_direction):
            q_ordinary = first_collision(point, direction, records, field.shape[0])
            draw_arrow(ray_draw, map_one(q_ordinary), map_one(point), (255, 173, 31, 112), width=2, head=6)
            if is_terminal:
                q_opposite = first_collision(point, -direction, records, field.shape[0])
                draw_arrow(ray_draw, map_one(q_opposite), map_one(point), (255, 43, 67, 118), width=2, head=6)
    image = Image.alpha_composite(image.convert("RGBA"), ray_layer).convert("RGB")
    draw = ImageDraw.Draw(image)

    for record in records:
        is_terminal = bool(record["terminal"])
        color = (255, 52, 70) if is_terminal else (255, 255, 255)
        contour = map_many(np.asarray(record["poly"]))
        draw.line(contour + [contour[0]], fill=(20, 25, 34), width=6, joint="curve")
        draw.line(contour + [contour[0]], fill=color, width=4 if is_terminal else 3, joint="curve")

        points = np.asarray(record["points"])
        mapped_points = map_many(points)
        for x, y in mapped_points:
            radius = 2.7
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 255), outline=(25, 31, 42), width=1)

        # One arrow per direction is sufficient to mark the two complete loop
        # scans; the p0 star identifies their common start and tagged return.
        n = len(mapped_points)
        cw_index = max(1, n // 5)
        ccw_index = max(2, (4 * n) // 5)
        draw_arrow(
            draw,
            mapped_points[cw_index],
            mapped_points[(cw_index + 1) % n],
            (23, 132, 210),
            width=6,
            head=12,
        )
        draw_arrow(
            draw,
            mapped_points[ccw_index],
            mapped_points[(ccw_index - 1) % n],
            (142, 77, 213),
            width=6,
            head=12,
        )
        draw_star(draw, map_one(np.asarray(record["p0"])), 10.5, (255, 222, 55), outline=(129, 32, 48))

    # Compact symbol legend.  It deliberately contains no architecture recap;
    # the paper caption can carry the prose already established in the text.
    draw.line((1200, 55, 1200, 1165), fill=(218, 222, 229), width=2)
    legend_x = 1250
    heading = find_font(32, bold=True)
    label = find_font(25)
    note = find_font(20)
    draw.text((legend_x, 63), "Symbols", font=heading, fill=(24, 29, 39))

    row_y = 145
    swatch = heatmap_rgb(np.linspace(0.0, 1.0, 100, dtype=float)[None, :])
    swatch_image = Image.fromarray(swatch, mode="RGB").resize((92, 32), Image.Resampling.BILINEAR)
    image.paste(swatch_image, (legend_x, row_y + 3))
    draw = ImageDraw.Draw(image)
    draw.rectangle((legend_x, row_y + 3, legend_x + 92, row_y + 35), outline=(43, 49, 59), width=1)
    draw.text((legend_x + 118, row_y), "S(x, y)", font=label, fill=(27, 33, 44))
    draw.text((legend_x + 118, row_y + 34), "route field", font=note, fill=(91, 99, 112))

    row_y += 105
    draw.line((legend_x, row_y + 9, legend_x + 92, row_y + 9), fill=(24, 30, 40), width=6)
    draw.line((legend_x, row_y + 9, legend_x + 92, row_y + 9), fill=(255, 255, 255), width=3)
    draw.line((legend_x, row_y + 42, legend_x + 92, row_y + 42), fill=(24, 30, 40), width=7)
    draw.line((legend_x, row_y + 42, legend_x + 92, row_y + 42), fill=(255, 52, 70), width=4)
    draw.text((legend_x + 118, row_y - 9), "C_λ", font=label, fill=(27, 33, 44))
    draw.text((legend_x + 230, row_y - 5), "non-terminal", font=note, fill=(91, 99, 112))
    draw.text((legend_x + 118, row_y + 24), "C_term", font=label, fill=(27, 33, 44))
    draw.text((legend_x + 230, row_y + 28), "terminal", font=note, fill=(91, 99, 112))

    row_y += 105
    draw.ellipse((legend_x + 37, row_y + 8, legend_x + 55, row_y + 26), fill=(255, 255, 255), outline=(24, 30, 40), width=2)
    draw.text((legend_x + 118, row_y), "p_i", font=label, fill=(27, 33, 44))
    draw.text((legend_x + 118, row_y + 34), "final contour sample", font=note, fill=(91, 99, 112))

    row_y += 105
    draw_star(draw, (legend_x + 47, row_y + 18), 15, (255, 222, 55), outline=(129, 32, 48))
    draw.text((legend_x + 118, row_y), "p0", font=label, fill=(27, 33, 44))
    draw.text((legend_x + 118, row_y + 34), "start + tagged return", font=note, fill=(91, 99, 112))

    row_y += 112
    draw.text((legend_x - 2, row_y - 7), "q", font=note, fill=(91, 99, 112))
    draw_arrow(draw, (legend_x + 24, row_y + 17), (legend_x + 93, row_y + 17), (255, 156, 20), width=7, head=14)
    draw.text((legend_x + 97, row_y - 7), "p", font=note, fill=(91, 99, 112))
    draw.text((legend_x + 148, row_y), "q → p", font=label, fill=(27, 33, 44))
    draw.text((legend_x + 148, row_y + 34), "ordinary normal read", font=note, fill=(91, 99, 112))

    row_y += 112
    draw.text((legend_x - 2, row_y - 7), "q_opp", font=note, fill=(91, 99, 112))
    draw_arrow(draw, (legend_x + 48, row_y + 17), (legend_x + 117, row_y + 17), (255, 43, 67), width=7, head=14)
    draw.text((legend_x + 121, row_y - 7), "p", font=note, fill=(91, 99, 112))
    draw.text((legend_x + 173, row_y), "q_opp → p", font=label, fill=(27, 33, 44))
    draw.text((legend_x + 173, row_y + 34), "terminal opposite read", font=note, fill=(91, 99, 112))

    row_y += 120
    draw_arrow(draw, (legend_x, row_y + 17), (legend_x + 92, row_y + 17), (23, 132, 210), width=7, head=14)
    draw.text((legend_x + 118, row_y), "CW contour scan", font=label, fill=(27, 33, 44))

    row_y += 83
    draw_arrow(draw, (legend_x + 92, row_y + 17), (legend_x, row_y + 17), (142, 77, 213), width=7, head=14)
    draw.text((legend_x + 118, row_y), "CCW contour scan", font=label, fill=(27, 33, 44))

    row_y += 105
    bar_left, bar_top, bar_width, bar_height = legend_x, row_y, 42, 170
    gradient = heatmap_rgb(np.linspace(1.0, 0.0, 180, dtype=float)[:, None])
    gradient_image = Image.fromarray(gradient, mode="RGB").resize((bar_width, bar_height), Image.Resampling.BILINEAR)
    image.paste(gradient_image, (bar_left, bar_top))
    draw = ImageDraw.Draw(image)
    draw.rectangle((bar_left, bar_top, bar_left + bar_width, bar_top + bar_height), outline=(43, 49, 59), width=1)
    draw_arrow(draw, (bar_left + 70, bar_top + bar_height), (bar_left + 70, bar_top), (52, 59, 72), width=4, head=11)
    draw.text((bar_left + 100, bar_top - 5), "high λ", font=label, fill=(27, 33, 44))
    draw.text((bar_left + 100, bar_top + bar_height - 29), "low λ", font=label, fill=(27, 33, 44))
    draw.text((bar_left + 100, bar_top + 69), "Tree order", font=note, fill=(91, 99, 112))
    draw.text((bar_left + 100, bar_top + 95), "low → high", font=note, fill=(91, 99, 112))

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=98, dpi=(300, 300))


def build_records(field: np.ndarray, level_count: int, delta0: float, n_min: int) -> tuple[np.ndarray, list[dict]]:
    gx, gy = sobel_field(field)
    levels = np.linspace(0.18, 0.82, level_count)
    records: list[dict] = []
    for level in levels:
        for poly in marching_squares(field, float(level)):
            poly = orient_clockwise(poly)
            if perimeter(poly) < 12.0 or abs(polygon_area(poly)) < 8.0:
                continue
            record = contour_sampling(poly, gx, gy, delta0=delta0, n_min=n_min)
            record["level"] = float(level)
            records.append(record)
    if not records:
        raise RuntimeError("No valid closed contour was produced; try another seed or fewer levels.")
    # A retained contour is terminal when no higher retained contour continues
    # inside it.  This marks the last retained loop of each visible branch, not
    # merely every loop at the globally highest scalar level.
    for record in records:
        level = float(record["level"])
        poly = np.asarray(record["poly"])
        has_higher_successor = any(
            float(other["level"]) > level
            and point_in_polygon(np.asarray(other["poly"])[0], poly)
            for other in records
        )
        record["terminal"] = not has_higher_successor
    return levels, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=5, help="random heatmap seed (default: 5)")
    parser.add_argument("--grid-size", type=int, default=96, help="route-grid width and height")
    parser.add_argument("--levels", type=int, default=6, help="number of retained scalar levels")
    parser.add_argument("--delta0", type=float, default=2.0, help="perimeter cells per final point")
    parser.add_argument("--n-min", type=int, default=16, help="minimum final points per retained contour")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/figures/topocontour_paper_figure.png"),
        help="output PNG path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid_size < 32:
        raise ValueError("--grid-size must be at least 32")
    if args.levels < 2:
        raise ValueError("--levels must be at least 2")
    field = random_heatmap(args.grid_size, args.seed)
    levels, records = build_records(field, args.levels, args.delta0, args.n_min)
    render(field, levels, records, args.output, args.seed)
    total_points = sum(int(record["n_final"]) for record in records)
    print(f"saved: {args.output.resolve()}")
    print(f"retained contours: {len(records)}; final contour points: {total_points}")


if __name__ == "__main__":
    main()
