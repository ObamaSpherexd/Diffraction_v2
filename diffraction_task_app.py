#!/usr/bin/env python3


import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
import tkinter as tk
from tkinter import ttk, messagebox
from enum import Enum
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
NM = 1e-9  # нанометры → метры
MM = 1e-3  # миллиметры → метры


# ---------------------------------------------------------------------------
# Типы апертур
# ---------------------------------------------------------------------------
class ApertureType(Enum):
    SINGLE_SLIT = "Одиночная щель"
    DOUBLE_SLIT = "Двойная щель"
    CIRCULAR = "Круглое отверстие"
    RECTANGULAR = "Прямоугольное отверстие"
    SQUARE_OBSTACLE = "Непрозрачный экран (диск)"
    TRIANGLE = "Треугольное отверстие"
    DOUBLE_RECT = "2 прямоугольных отверстия"
    DOUBLE_CIRC = "2 круглых отверстия"


# ---------------------------------------------------------------------------
# Стандартные примеры (пресеты)
# ---------------------------------------------------------------------------
@dataclass
class Preset:
    name: str
    wavelength_nm: float       # длина волны (нм)
    a: float                   # расстояние источник → апертура (м)
    b: float                   # расстояние апертура → экран (м)
    aperture: ApertureType
    params: dict               # параметры апертуры (в метрах)
    screen_size_mm: float      # размер области наблюдения на экране (мм)
    grid_points: int           # число точек сетки


PRESETS = [
    Preset(
        name="Фраунгофер: щель 0.1 мм, λ=632 нм",
        wavelength_nm=632.8,
        a=1.0,
        b=2.0,
        aperture=ApertureType.SINGLE_SLIT,
        params={"width": 0.1e-3},
        screen_size_mm=20.0,
        grid_points=512,
    ),
    Preset(
        name="Френель: круглое отверстие 1 мм, λ=532 нм",
        wavelength_nm=532.0,
        a=0.3,
        b=0.3,
        aperture=ApertureType.CIRCULAR,
        params={"radius": 1.0e-3},
        screen_size_mm=10.0,
        grid_points=512,
    ),
    Preset(
        name="Фраунгофер: двойная щель, λ=650 нм",
        wavelength_nm=650.0,
        a=1.0,
        b=3.0,
        aperture=ApertureType.DOUBLE_SLIT,
        params={"slit_width": 0.05e-3, "slit_separation": 0.2e-3},
        screen_size_mm=30.0,
        grid_points=512,
    ),
    Preset(
        name="Френель: прямоугольное отверстие, λ=500 нм",
        wavelength_nm=500.0,
        a=0.5,
        b=0.5,
        aperture=ApertureType.RECTANGULAR,
        params={"width_x": 1.5e-3, "width_y": 0.5e-3},
        screen_size_mm=15.0,
        grid_points=512,
    ),
    Preset(
        name="Фраунгофер: круглое отверстие (диск Эйри), λ=550 нм",
        wavelength_nm=550.0,
        a=10.0,
        b=10.0,
        aperture=ApertureType.CIRCULAR,
        params={"radius": 0.5e-3},
        screen_size_mm=10.0,
        grid_points=512,
    ),
]


# ---------------------------------------------------------------------------
# Создание масок апертур
# ---------------------------------------------------------------------------
def make_single_slit_mask(N, size, width):
    """Маска одиночной щели (вертикальная щель)."""
    x = np.linspace(-size / 2, size / 2, N)
    mask = np.abs(x[np.newaxis, :]) <= width / 2
    return np.ones((N, N)) * mask


def make_double_slit_mask(N, size, slit_width, slit_separation):
    """Маска двойной щели."""
    x = np.linspace(-size / 2, size / 2, N)
    center1 = -slit_separation / 2
    center2 = slit_separation / 2
    mask1 = np.abs(x[np.newaxis, :] - center1) <= slit_width / 2
    mask2 = np.abs(x[np.newaxis, :] - center2) <= slit_width / 2
    return np.ones((N, N)) * (mask1 | mask2)


def make_circular_mask(N, size, radius):
    """Маска круглого отверстия."""
    x = np.linspace(-size / 2, size / 2, N)
    y = np.linspace(-size / 2, size / 2, N)
    X, Y = np.meshgrid(x, y)
    return (X ** 2 + Y ** 2) <= radius ** 2


def make_rectangular_mask(N, size, width_x, width_y):
    """Маска прямоугольного отверстия."""
    x = np.linspace(-size / 2, size / 2, N)
    y = np.linspace(-size / 2, size / 2, N)
    X, Y = np.meshgrid(x, y)
    return (np.abs(X) <= width_x / 2) & (np.abs(Y) <= width_y / 2)


def make_square_obstacle_mask(N, size, radius):
    """Непрозрачный круглый диск на прозрачном фоне."""
    x = np.linspace(-size / 2, size / 2, N)
    y = np.linspace(-size / 2, size / 2, N)
    X, Y = np.meshgrid(x, y)
    return ~((X ** 2 + Y ** 2) <= radius ** 2)


def make_triangle_mask(N, size):
    """Маска равностороннего треугольного отверстия."""
    x = np.linspace(-size / 2, size / 2, N)
    y = np.linspace(-size / 2, size / 2, N)
    X, Y = np.meshgrid(x, y)
    h = size * 0.4
    # Вершины треугольника
    v0 = np.array([0, h / 2])
    v1 = np.array([-h / np.sqrt(3), -h / 2])
    v2 = np.array([h / np.sqrt(3), -h / 2])

    def sign(p1, p2, p3):
        return (p1[:, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[:, 1] - p3[1])

    points = np.column_stack([X.ravel(), Y.ravel()])
    d1 = sign(points, v0, v1)
    d2 = sign(points, v1, v2)
    d3 = sign(points, v2, v0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    mask = ~(has_neg & has_pos)
    return mask.reshape(N, N)


def make_double_rect_mask(N, size, width_x, width_y, separation):
    """Маска двух прямоугольных отверстий."""
    x = np.linspace(-size / 2, size / 2, N)
    y = np.linspace(-size / 2, size / 2, N)
    X, Y = np.meshgrid(x, y)
    center1 = -separation / 2
    center2 = separation / 2
    mask1 = (np.abs(X - center1) <= width_x / 2) & (np.abs(Y) <= width_y / 2)
    mask2 = (np.abs(X - center2) <= width_x / 2) & (np.abs(Y) <= width_y / 2)
    return mask1 | mask2


def make_double_circ_mask(N, size, radius, separation):
    """Маска двух круглых отверстий."""
    x = np.linspace(-size / 2, size / 2, N)
    y = np.linspace(-size / 2, size / 2, N)
    X, Y = np.meshgrid(x, y)
    center1 = -separation / 2
    center2 = separation / 2
    mask1 = np.sqrt((X - center1) ** 2 + Y ** 2) <= radius
    mask2 = np.sqrt((X - center2) ** 2 + Y ** 2) <= radius
    return mask1 | mask2


def make_aperture(aperture_type, N, aperture_size, params):
    """Создаёт маску апертуры заданного типа."""
    if aperture_type == ApertureType.SINGLE_SLIT:
        w = params.get("width", aperture_size / 10)
        return make_single_slit_mask(N, aperture_size, w)
    elif aperture_type == ApertureType.DOUBLE_SLIT:
        sw = params.get("slit_width", aperture_size / 20)
        ss = params.get("slit_separation", aperture_size / 5)
        return make_double_slit_mask(N, aperture_size, sw, ss)
    elif aperture_type == ApertureType.CIRCULAR:
        r = params.get("radius", aperture_size / 6)
        return make_circular_mask(N, aperture_size, r)
    elif aperture_type == ApertureType.RECTANGULAR:
        wx = params.get("width_x", aperture_size / 4)
        wy = params.get("width_y", aperture_size / 8)
        return make_rectangular_mask(N, aperture_size, wx, wy)
    elif aperture_type == ApertureType.SQUARE_OBSTACLE:
        r = params.get("radius", aperture_size / 10)
        return make_square_obstacle_mask(N, aperture_size, r)
    elif aperture_type == ApertureType.TRIANGLE:
        return make_triangle_mask(N, aperture_size)
    elif aperture_type == ApertureType.DOUBLE_RECT:
        wx = params.get("width_x", aperture_size / 8)
        wy = params.get("width_y", aperture_size / 6)
        sep = params.get("separation", aperture_size / 3)
        return make_double_rect_mask(N, aperture_size, wx, wy, sep)
    elif aperture_type == ApertureType.DOUBLE_CIRC:
        r = params.get("radius", aperture_size / 8)
        sep = params.get("separation", aperture_size / 3)
        return make_double_circ_mask(N, aperture_size, r, sep)
    else:
        return np.ones((N, N))


# ---------------------------------------------------------------------------
# Расчёт дифракции Френеля
# ---------------------------------------------------------------------------
def fresnel_diffraction(aperture, wavelength, a, b, aperture_size, screen_size, N):
    """
    Численный расчёт дифракции Френеля методом БПФ (свёрточный подход).

    Используется формула Френеля-Кирхгофа с квадратичным фазовым множителем:

        U(x, y) = exp(ikb)/(i·λ·b) · exp(ik·(x²+y²)/(2b)) ·
                  FFT{ A(ξ,η) · exp(ik·(ξ²+η²)/(2b)) · exp(ik·(ξ²+η²)/(2a)) }

    где FFT берётся на сетке, согласованной с экраном.
    """
    k = 2 * np.pi / wavelength

    # --- Шаг 1: координаты на экране ---
    x = np.linspace(-screen_size / 2, screen_size / 2, N)
    y = np.linspace(-screen_size / 2, screen_size / 2, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # --- Шаг 2: координаты в апертуре (те же размеры, т.к. свёртка) ---
    # При свёрточном подходе экран и апертура имеют одинаковую сетку
    xi = x
    eta = y
    XI, ETA = np.meshgrid(xi, eta)

    # --- Шаг 3: фаза на экране ---
    phase_screen = np.exp(1j * k * (X ** 2 + Y ** 2) / (2 * b))

    # --- Шаг 4: эффективная апертура ---
    # Квадратичная фаза от расстояния b
    phase_b = np.exp(1j * k * (XI ** 2 + ETA ** 2) / (2 * b))
    # Квадратичная фаза от расстояния a (сферическая волна)
    if np.isfinite(a) and a > 0:
        phase_a = np.exp(1j * k * (XI ** 2 + ETA ** 2) / (2 * a))
    else:
        phase_a = np.ones_like(XI)

    # Масштабируем апертуру до размера сетки (если нужно)
    if aperture.shape != (N, N):
        from scipy.ndimage import zoom
        zy, zx = N / aperture.shape[0], N / aperture.shape[1]
        aperture_resampled = zoom(aperture.real, (zy, zx), order=1)
        aperture_resampled = (aperture_resampled > 0.5).astype(complex)
    else:
        aperture_resampled = aperture.astype(complex)

    U_eff = aperture_resampled * phase_b * phase_a

    # --- Шаг 5: БПФ свёртка ---
    prefactor = np.exp(1j * k * b) / (1j * wavelength * b)

    U_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_eff)))
    U_fft *= dx * dy

    U = prefactor * phase_screen * U_fft

    intensity = np.abs(U) ** 2

    max_I = np.max(intensity)
    if max_I > 0:
        intensity /= max_I

    # actual FFT screen coords
    x_nat = np.fft.fftshift(np.fft.fftfreq(N, dx)) * wavelength * b
    y_nat = np.fft.fftshift(np.fft.fftfreq(N, dy)) * wavelength * b
    return intensity, x_nat, y_nat


# ---------------------------------------------------------------------------
# Расчёт дифракции Фраунгофера
# ---------------------------------------------------------------------------
def fraunhofer_diffraction(aperture, wavelength, b, aperture_size, screen_size, N):
    """
    Расчёт дифракции Фраунгофера (дальняя зона) через БПФ.

    U(x, y) ∝ FT{A(ξ,η)} evaluated at fx = x/(λb), fy = y/(λb)

    Возвращает интенсивность на естественной сетке БПФ,
    ограниченной запрошенным screen_size.
    """
    # Координаты в плоскости апертуры
    xi = np.linspace(-aperture_size / 2, aperture_size / 2, N)
    eta = np.linspace(-aperture_size / 2, aperture_size / 2, N)
    dxi = xi[1] - xi[0]
    deta = eta[1] - eta[0]

    # БПФ апертуры
    A_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))

    # Естественные экранные координаты БПФ:
    x_nat = np.fft.fftshift(np.fft.fftfreq(N, dxi)) * wavelength * b
    y_nat = np.fft.fftshift(np.fft.fftfreq(N, deta)) * wavelength * b

    intensity = np.abs(A_fft) ** 2
    max_I = np.max(intensity)
    if max_I > 0:
        intensity /= max_I

    return intensity, x_nat, y_nat


# ---------------------------------------------------------------------------
# Характерный размер апертуры для числа Френеля
# ---------------------------------------------------------------------------
def get_aperture_char_size(aperture_type, params, aperture_size):
    if aperture_type == ApertureType.SINGLE_SLIT:
        return params.get("width", aperture_size / 10)
    elif aperture_type == ApertureType.DOUBLE_SLIT:
        sw = params.get("slit_width", aperture_size / 20)
        ss = params.get("slit_separation", aperture_size / 5)
        return ss + sw
    elif aperture_type == ApertureType.CIRCULAR:
        return params.get("radius", aperture_size / 6) * 2
    elif aperture_type == ApertureType.RECTANGULAR:
        wx = params.get("width_x", aperture_size / 4)
        wy = params.get("width_y", aperture_size / 8)
        return max(wx, wy)
    elif aperture_type == ApertureType.SQUARE_OBSTACLE:
        return params.get("radius", aperture_size / 10) * 2
    elif aperture_type == ApertureType.TRIANGLE:
        return aperture_size * 0.4
    elif aperture_type == ApertureType.DOUBLE_RECT:
        wx = params.get("width_x", aperture_size / 8)
        sep = params.get("separation", aperture_size / 3)
        return sep + wx
    elif aperture_type == ApertureType.DOUBLE_CIRC:
        r = params.get("radius", aperture_size / 8)
        sep = params.get("separation", aperture_size / 3)
        return sep + 2 * r
    return aperture_size / 3

# ---------------------------------------------------------------------------
# Универсальный расчёт (автовыбор Френель/Фраунгофер)
# ---------------------------------------------------------------------------
def compute_diffraction(aperture, wavelength, a, b, aperture_size, screen_size, N,
                        mode="auto", d_char=None):
    """
    mode = "fresnel" | "fraunhofer" | "auto"
    В режиме auto выбирается по числу Френеля:
        N_F = d² / (λ·b),  где d — характерный размер апертуры.
        N_F > 1 → Френель, N_F < 1 → Фраунгофер.
    """
    if d_char is None:
        d_char = aperture_size / 3
    N_F = d_char ** 2 / (wavelength * b)

    if mode == "auto":
        mode = "fresnel" if N_F > 0.5 else "fraunhofer"

    if mode == "fresnel":
        intensity, x, y = fresnel_diffraction(
            aperture, wavelength, a, b, aperture_size, screen_size, N)
        return intensity, x, y, "fresnel", N_F
    else:
        intensity, x, y = fraunhofer_diffraction(
            aperture, wavelength, b, aperture_size, screen_size, N)
        return intensity, x, y, "fraunhofer", N_F


# ---------------------------------------------------------------------------
# GUI-приложение
# ---------------------------------------------------------------------------
class DiffractionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Дифракция Френеля и Фраунгофера")
        self.root.geometry("1200x800")

        self.fig = None
        self.current_mode = "auto"

        self._build_ui()
        self._apply_preset(0)

    def _build_ui(self):
        # ----- Панель управления -----
        ctrl_frame = ttk.Frame(self.root, padding=10)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Пресеты
        ttk.Label(ctrl_frame, text="Пресеты:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.preset_var = tk.StringVar()
        preset_names = [p.name for p in PRESETS]
        self.preset_combo = ttk.Combobox(ctrl_frame, textvariable=self.preset_var,
                                          values=preset_names, state="readonly", width=40)
        self.preset_combo.pack(anchor=tk.W, pady=2)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset)

        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Параметры
        ttk.Label(ctrl_frame, text="Параметры:", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        params = [
            ("wavelength", "Длина волны (нм):", "532"),
            ("a", "Расст. источник→отверстие (м):", "1.0"),
            ("b", "Расст. отверстие→экран (м):", "1.0"),
            ("aperture_size", "Размер апертуры (мм):", "5.0"),
            ("screen_size", "Размер экрана (мм):", "20.0"),
            ("grid", "Точки сетки:", "512"),
        ]

        self.entries = {}
        for key, label, default in params:
            row = ttk.Frame(ctrl_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=32).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(row, textvariable=var, width=12)
            entry.pack(side=tk.LEFT, padx=3)
            self.entries[key] = var

        # Тип апертуры
        row = ttk.Frame(ctrl_frame)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="Тип апертуры:", width=32).pack(side=tk.LEFT)
        self.aperture_var = tk.StringVar(value=ApertureType.SINGLE_SLIT.value)
        ap_types = [t.value for t in ApertureType]
        self.aperture_combo = ttk.Combobox(row, textvariable=self.aperture_var,
                                            values=ap_types, state="readonly", width=18)
        self.aperture_combo.pack(side=tk.LEFT, padx=3)

        # Доп. параметры апертуры
        self.extra_frame = ttk.Frame(ctrl_frame)
        self.extra_frame.pack(fill=tk.X, pady=4)
        self._build_extra_params()

        # Режим расчёта
        row = ttk.Frame(ctrl_frame)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="Режим расчёта:", width=32).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="auto")
        for m, label in [("auto", "Авто"), ("fresnel", "Френель"), ("fraunhofer", "Фраунгофер")]:
            ttk.Radiobutton(row, text=label, variable=self.mode_var, value=m).pack(side=tk.LEFT, padx=5)

        # Кнопка
        ttk.Button(ctrl_frame, text="Рассчитать", command=self._calculate).pack(pady=12, fill=tk.X)

        # ----- Область графика -----
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _build_extra_params(self):
        for w in self.extra_frame.winfo_children():
            w.destroy()

        ap_type = ApertureType(self.aperture_var.get())
        self.extra_entries = {}

        extras = {
            ApertureType.SINGLE_SLIT: [("width", "Ширина щели (мм):", "0.1")],
            ApertureType.DOUBLE_SLIT: [
                ("slit_width", "Ширина щели (мм):", "0.05"),
                ("slit_sep", "Расстояние между (мм):", "0.2"),
            ],
            ApertureType.CIRCULAR: [("radius", "Радиус (мм):", "0.5")],
            ApertureType.RECTANGULAR: [
                ("width_x", "Ширина X (мм):", "1.5"),
                ("width_y", "Ширина Y (мм):", "0.5"),
            ],
            ApertureType.SQUARE_OBSTACLE: [("radius", "Радиус диска (мм):", "0.5")],
            ApertureType.DOUBLE_RECT: [
                ("width_x", "Ширина X (мм):", "0.3"),
                ("width_y", "Высота Y (мм):", "0.4"),
                ("separation", "Расстояние между (мм):", "0.8"),
            ],
            ApertureType.DOUBLE_CIRC: [
                ("radius", "Радиус (мм):", "0.3"),
                ("separation", "Расстояние между (мм):", "0.8"),
            ],
        }

        for key, label, default in extras.get(ap_type, []):
            row = ttk.Frame(self.extra_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=28).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(row, textvariable=var, width=10)
            entry.pack(side=tk.LEFT, padx=3)
            self.extra_entries[key] = var

    def _on_aperture_change(self, event=None):
        self._build_extra_params()

    def _on_preset(self, event=None):
        idx = self.preset_combo.current()
        if idx >= 0:
            self._apply_preset(idx)

    def _apply_preset(self, idx):
        p = PRESETS[idx]
        self.entries["wavelength"].set(str(p.wavelength_nm))
        self.entries["a"].set(str(p.a))
        self.entries["b"].set(str(p.b))
        self.aperture_var.set(p.aperture.value)
        self.entries["aperture_size"].set(str(p.screen_size_mm))
        self.entries["screen_size"].set(str(p.screen_size_mm))
        self.entries["grid"].set(str(p.grid_points))

        # Доп. параметры
        for k, v in p.params.items():
            v_mm = v / MM
            if k == "width":
                self._build_extra_params()
                if "width" in self.extra_entries:
                    self.extra_entries["width"].set(f"{v_mm:.3f}")
            elif k == "radius":
                self._build_extra_params()
                if "radius" in self.extra_entries:
                    self.extra_entries["radius"].set(f"{v_mm:.3f}")
            elif k == "slit_width":
                self._build_extra_params()
                if "slit_width" in self.extra_entries:
                    self.extra_entries["slit_width"].set(f"{v_mm:.3f}")
            elif k == "slit_separation":
                self._build_extra_params()
                if "slit_sep" in self.extra_entries:
                    self.extra_entries["slit_sep"].set(f"{v_mm:.3f}")
            elif k == "width_x":
                self._build_extra_params()
                if "width_x" in self.extra_entries:
                    self.extra_entries["width_x"].set(f"{v_mm:.3f}")
            elif k == "width_y":
                self._build_extra_params()
                if "width_y" in self.extra_entries:
                    self.extra_entries["width_y"].set(f"{v_mm:.3f}")

        self._build_extra_params()

    def _get_float(self, key):
        return float(self.entries[key].get())

    def _calculate(self):
        try:
            wavelength_nm = self._get_float("wavelength")
            a = self._get_float("a")
            b = self._get_float("b")
            aperture_size_mm = self._get_float("aperture_size")
            screen_size_mm = self._get_float("screen_size")
            N = int(self._get_float("grid"))

            wavelength = wavelength_nm * NM
            aperture_size = aperture_size_mm * MM
            screen_size = screen_size_mm * MM

            ap_type = ApertureType(self.aperture_var.get())

            # Доп. параметры (в метрах)
            params = {}
            for k, var in self.extra_entries.items():
                val = float(var.get()) * MM
                if k == "slit_sep":
                    params["slit_separation"] = val
                else:
                    params[k] = val

            aperture = make_aperture(ap_type, N, aperture_size, params)

            d_char = get_aperture_char_size(ap_type, params, aperture_size)
            intensity, x, y, mode_used, N_F = compute_diffraction(
                aperture, wavelength, a, b, aperture_size, screen_size, N,
                mode=self.mode_var.get(), d_char=d_char)

            self._plot(aperture, intensity, x, y, wavelength_nm, a, b,
                       aperture_size_mm, screen_size_mm, mode_used, N_F, ap_type, params, aperture_size)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при расчёте:\n{e}")
            import traceback
            traceback.print_exc()

    def _plot(self, aperture, intensity, x, y, wavelength_nm, a, b,
              aperture_size_mm, screen_size_mm, mode_used, N_F, ap_type, params, aperture_size):
        if self.fig:
            plt.close(self.fig)

        self.fig = plt.Figure(figsize=(10, 7), dpi=100)
        gs = GridSpec(2, 2, figure=self.fig)

        # 1. Апертура
        ax1 = self.fig.add_subplot(gs[0, 0])
        extent = [-aperture_size_mm / 2, aperture_size_mm / 2,
                  -aperture_size_mm / 2, aperture_size_mm / 2]
        ax1.imshow(aperture, cmap="gray", extent=extent, origin="lower")
        ax1.set_xlabel("мм")
        ax1.set_ylabel("мм")
        ax1.set_title("Апертура")
        ax1.set_aspect("equal")

        # 2. Интенсивность на экране
        ax2 = self.fig.add_subplot(gs[0, 1])
        extent = [-screen_size_mm / 2, screen_size_mm / 2,
                  -screen_size_mm / 2, screen_size_mm / 2]
        im = ax2.imshow(intensity, cmap="hot", extent=extent, origin="lower",
                        vmin=0, vmax=1)
        ax2.set_xlabel("мм")
        ax2.set_ylabel("мм")
        mode_label = "Френель" if mode_used == "fresnel" else "Фраунгофер"
        ax2.set_title(f"Интенсивность ({mode_label}, N_F={N_F:.2f})")
        self.fig.colorbar(im, ax=ax2, label="I / I_max")

        # 3. Профиль интенсивности (горизонтальное сечение)
        ax3 = self.fig.add_subplot(gs[1, :])
        mid = len(y) // 2
        x_mm = x * 1000
        ax3.plot(x_mm, intensity[mid, :], "b-", linewidth=1)
        ax3.set_xlabel("Положение на экране (мм)")
        ax3.set_ylabel("I / I_max")
        ax3.set_title("Профиль интенсивности (центральное сечение)")
        ax3.grid(True, alpha=0.3)
        half = screen_size_mm / 2
        ax3.set_xlim(-half, half)

        self.fig.tight_layout()

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        for w in self.plot_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = DiffractionApp()
    app.root.mainloop()
