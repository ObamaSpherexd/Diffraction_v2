import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from enum import Enum
from dataclasses import dataclass
import streamlit as st
from scipy.special import j1, airy, fresnel

# constants
NM = 1e-9
MM = 1e-3


def parse_formula(formula: str) -> str:
    """Convert r^n to r**n for user formulas."""
    return formula.replace('^', '**')

# aperture types
class ApertureType(Enum):
    SINGLE_SLIT='Одиночная щель'
    DOUBLE_SLIT='Двойная щель'
    CIRCULAR='Круглое отверстие'
    RECTANGULAR='Прямоугольное отверстие'
    SQUARE_OBSTACLE='Непрозрочный экран'
    TRIANGLE='Треугольное отверстие'
    DIFFRACTION_GRATING='Дифракционная решетка'

    DOUBLE_RECT='2 прямоугольных отверстия'
    DOUBLE_CIRC='2 круглых отверстия'

    CUSTOM='Пользовательская'

# STANDART EXAMPLES - PRESETS

@dataclass
class Preset:
    name: str
    wavelength_nm: float
    a: float
    b: float
    aperture: 'ApertureType'
    params: dict
    screen_size_mm: float
    grid_points: int

CUSTOM_TEMPLATES = [
    {'name': 'Квадратичная линза (φ=πr²/λf)', 
     'amp': '1', 'phase': 'pi*r^2/(wavelength*f)', 
     'params': {'lambda': 500e-9, 'f': 0.5}},
    {'name': 'Зонная пластина Френеля', 
     'amp': '1', 'phase': 'pi*r^2/(wavelength*f)*(1-2*floor(r/sqrt(wavelength*f)))', 
     'params': {'lambda': 500e-9, 'f': 0.5}},
    {'name': 'Спираль Френеля', 
     'amp': '1', 'phase': 'atan2(y,x)', 
     'params': {}},
]

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
    Preset(
        name="Фраунгофер: дифракционная решетка, λ=550 нм",
        wavelength_nm=550.0,
        a=1.0,
        b=2.0,
        aperture=ApertureType.DIFFRACTION_GRATING,
        params={"period": 0.1e-3, "duty_cycle": 0.5},
        screen_size_mm=20.0,
        grid_points=512,
    ),
]

# PARSING CUSTOM MASKS
# parse_formula is already defined at the top of the file

# CREATING APERTURE MASKS
def make_single_slit_mask(N,size,width):
    '''single slit mask (vertival)'''
    x=np.linspace(-size/2,size/2,N)
    mask=np.abs(x[np.newaxis,:])<=width/2
    return np.ones((N,N))*mask

def make_double_slit_mask(N,size,slit_width,slit_separation):
    '''double slit'''
    x=np.linspace(-size/2,size/2,N)
    center1=-slit_separation/2
    center2=slit_separation/2
    mask1=np.abs(x[np.newaxis,:]-center1)<=slit_width/2
    mask2=np.abs(x[np.newaxis,:]-center2)<=slit_width/2
    return np.ones((N,N))*(mask1|mask2)

def make_circular_mask(N,size,radius):
    '''circular hole mask'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    return (X**2+Y**2)<=radius**2

def make_rectangular_mask(N,size,width_x,width_y):
    '''rectangular mask'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    return (np.abs(X)<=width_x/2) &(np.abs(Y)<=width_y/2)

def make_square_obstacle_mask(N,size,radius):
    '''NOT CLEAR disk on a clear background'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    return ~((X**2+Y**2)<=radius**2) # wavy thing is used to negate so that true -> false

def make_triangle_mask(N,size):
    '''sides equal rectangle'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    h=size*0.4  
    v0=np.array([0,h/2])
    v1=np.array([-h/np.sqrt(3),-h/2])
    v2=np.array([h/np.sqrt(3),-h/2])

    points=np.column_stack([X.ravel(),Y.ravel()])
    d1=(v1[0]-v0[0])*(points[:,1]-v0[1])-(v1[1]-v0[1])*(points[:,0]-v0[0])
    d2=(v2[0]-v1[0])*(points[:,1]-v1[1])-(v2[1]-v1[1])*(points[:,0]-v1[0])
    d3=(v0[0]-v2[0])*(points[:,1]-v2[1])-(v0[1]-v2[1])*(points[:,0]-v2[0])
    has_neg=(d1<0)|(d2<0)|(d3<0)
    has_pos=(d1>0)|(d2>0)|(d3>0)
    mask=~(has_neg&has_pos)
    return mask.reshape(N,N)

def make_diffraction_grating_mask(N,size,period,duty_cycle):
    '''diffraction grating mask'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    slit_width=period*duty_cycle
    mask=(X%period)<=slit_width
    return mask

def make_double_rect_mask(N,size,width_x,width_y,separation):
    '''two rectangular holes'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    center1=-separation/2
    center2=separation/2
    mask1=(np.abs(X-center1)<=width_x/2) & (np.abs(Y)<=width_y/2)
    mask2=(np.abs(X-center2)<=width_x/2) & (np.abs(Y)<=width_y/2)
    return mask1 | mask2

def make_double_circ_mask(N,size,radius,separation):
    '''two circular holes'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    center1=-separation/2
    center2=separation/2
    mask1=np.sqrt((X-center1)**2+Y**2)<=radius
    mask2=np.sqrt((X-center2)**2+Y**2)<=radius
    return mask1 | mask2

def make_custom_mask(N,size,amp_formula,phase_formula,params):
    '''custom aperture from user formulas A(r) and phi(r)'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    R=np.sqrt(X**2+Y**2)

    from numpy import sin,cos,exp,log,sqrt,pi,zeros,ones,ones_like,floor,arctan2
    env={'sin':sin,'cos':cos,'exp':exp,'log':log,'sqrt':sqrt,'pi':pi,'floor':floor,'atan2':arctan2,'r':R,'R':R,'x':X,'X':X,'y':Y,'Y':Y}
    env.update(params)
    if 'lambda' in env:
        env['wavelength'] = env.pop('lambda')
    
    amp=eval(parse_formula(amp_formula),env) if amp_formula else ones_like(R)
    phi=eval(parse_formula(phase_formula),env) if phase_formula else zeros_like(R)
    return amp*np.exp(1j*phi)


def make_aperture(aperture_type,N,aperture_size,params):
    '''Creating aperture mask of a given type'''
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
    elif aperture_type == ApertureType.DIFFRACTION_GRATING:
        period = params.get("period", aperture_size / 20)
        duty = params.get("duty_cycle", 0.5)
        return make_diffraction_grating_mask(N, aperture_size, period, duty)
    elif aperture_type == ApertureType.DOUBLE_RECT:
        wx = params.get("width_x", aperture_size / 8)
        wy = params.get("width_y", aperture_size / 6)
        sep = params.get("separation", aperture_size / 3)
        return make_double_rect_mask(N, aperture_size, wx, wy, sep)
    elif aperture_type == ApertureType.DOUBLE_CIRC:
        r = params.get("radius", aperture_size / 8)
        sep = params.get("separation", aperture_size / 3)
        return make_double_circ_mask(N, aperture_size, r, sep)
    elif aperture_type==ApertureType.CUSTOM:
        extra_params = {k: v for k, v in params.items() if k not in ('amp', 'phase')}
        return make_custom_mask(N,aperture_size,params.get('amp','1'),
                                                params.get('phase','0'),
                                                extra_params)
    else:
        return np.ones((N, N))

def theoretical_slit_profile(x, slit_width, wavelength, b):
    alpha = np.pi * slit_width * x / (wavelength * b)
    return (np.sin(alpha + 1e-10) / (alpha + 1e-10))**2

def theoretical_circular_profile(r, radius, wavelength, b):
    k = 2 * np.pi / wavelength
    rho = k * radius * r / b
    j = j1(rho + 1e-10)
    return (2 * j / (rho + 1e-10))**2

def theoretical_double_slit_profile(x, slit_width, slit_separation, wavelength, b):
    alpha = np.pi * slit_width * x / (wavelength * b)
    beta = np.pi * slit_separation * x / (wavelength * b)
    return ((np.sin(alpha + 1e-10) / (alpha + 1e-10))**2
            * (np.cos(beta + 1e-10))**2)

def theoretical_rectangular_profile(x, width_x, wavelength, b):
    alpha = np.pi * width_x * x / (wavelength * b)
    return (np.sin(alpha + 1e-10) / (alpha + 1e-10))**2

def theoretical_grating_profile(x, period, duty_cycle, wavelength, b, N_lines=10):
    slit_width = period * duty_cycle
    alpha = np.pi * slit_width * x / (wavelength * b)
    beta = np.pi * period * x / (wavelength * b)
    single = (np.sin(alpha + 1e-10) / (alpha + 1e-10))**2
    with np.errstate(divide='ignore', invalid='ignore'):
        grating = (np.sin(N_lines * beta + 1e-10) / (N_lines * np.sin(beta + 1e-10)))**2
    grating = np.nan_to_num(grating, nan=1.0)
    return single * grating

def theoretical_double_rect_profile(x, width_x, separation, wavelength, b):
    alpha = np.pi * width_x * x / (wavelength * b)
    beta = np.pi * separation * x / (wavelength * b)
    return (np.sin(alpha + 1e-10) / (alpha + 1e-10))**2 * (np.cos(beta + 1e-10))**2

def theoretical_double_circular_profile(x, radius, separation, wavelength, b):
    k = 2 * np.pi / wavelength
    rho = k * radius * np.abs(x) / b
    j = j1(rho + 1e-10)
    airy = (2 * j / (rho + 1e-10))**2
    beta = np.pi * separation * x / (wavelength * b)
    return airy * (np.cos(beta + 1e-10))**2

def find_minima_positions(slit_width, wavelength, b, screen_size, n_points=5):
    x = np.linspace(0, screen_size/2, 1000)
    alpha = np.pi * slit_width * x / (wavelength * b)
    vals = (np.sin(alpha + 1e-10) / (alpha + 1e-10))**2
    minima_pos = []
    for i in range(1, len(vals)-1):
        if vals[i] < vals[i-1] and vals[i] < vals[i+1] and vals[i] < 0.1:
            minima_pos.append(x[i])
            if len(minima_pos) >= n_points:
                break
    return np.array(minima_pos) * 1000
    
# CALCULATING FRENsEL DIFFRACTION
def frensel_diffraction(aperture,wavelength,a,b,aperture_size,screen_size,N):
    '''
    numerical evaluation of frensel diffraction using FFT'''
    k=2*np.pi/wavelength

    x=np.linspace(-screen_size/2,screen_size/2,N)
    y=np.linspace(-screen_size/2,screen_size/2,N)
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    X,Y=np.meshgrid(x,y)

    xi=x
    eta=y
    XI,ETA=np.meshgrid(xi,eta)

    phase_screen =np.exp(1j*k*(X**2+Y**2)/(2*b))
    phase_b=np.exp(1j*k*(XI**2+ETA**2)/(2*b))

    if np.isfinite(a) and a>0:
        phase_a=np.exp(1j*k*(XI**2+ETA**2)/(2*b))
    else:
        phase_a=np.ones_like(XI)

    if aperture.shape!=(N,N):
        from scipy.ndimage import zoom
        zy,zx=N/aperture.shape[0],N/aperture.shape[1]
        aperture_resampled=zoom(aperture.real,(zx,zy),order=1)
        aperture_resampled=(aperture_resampled>0.5).astype(complex)
    else:
        aperture_resampled=aperture.astype(complex)
    U_eff=aperture_resampled*phase_b*phase_a

    prefactor=np.exp(1j*k*b)/(1j*wavelength*b)

    U_fft=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_eff)))
    U_fft*=dx*dy
    
    U=prefactor*phase_screen*U_fft

    intensity=np.abs(U)**2

    max_I=np.max(intensity)
    if max_I>0:
        intensity/=max_I
    return intensity

# FRAUNHOFER DIFFRACTION 
def fraunhofer_diffraction(aperture,wavelength,b,aperture_size,screen_size,N):
    '''Fraunhofer calculation (far away) using fft once again boring'''
    xi=np.linspace(-aperture_size/2,aperture_size/2,N)
    eta=np.linspace(-aperture_size/2,aperture_size/2,N)
    dxi=xi[1]-xi[0]
    deta=eta[1]-eta[0]

    A_fft=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))

    x_nat=np.fft.fftshift(np.fft.fftfreq(N,dxi))*wavelength*b
    y_nat=np.fft.fftshift(np.fft.fftfreq(N,deta))*wavelength*b

    intensity=np.abs(A_fft)**2
    max_I=np.max(intensity)
    if max_I>0:
        intensity/=max_I
    return intensity,x_nat,y_nat

def get_aperture_char_size(aperture_type, params, aperture_size):
    '''Characteristic size of the aperture for Fresnel number calculation.'''
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
    elif aperture_type == ApertureType.DIFFRACTION_GRATING:
        return params.get("period", aperture_size / 20)
    elif aperture_type == ApertureType.DOUBLE_RECT:
        wx = params.get("width_x", aperture_size / 8)
        sep = params.get("separation", aperture_size / 3)
        return sep + wx
    elif aperture_type == ApertureType.DOUBLE_CIRC:
        r = params.get("radius", aperture_size / 8)
        sep = params.get("separation", aperture_size / 3)
        return sep + 2 * r
    elif aperture_type == ApertureType.CUSTOM:
        return aperture_size / 3
    return aperture_size / 3

def compute_diffraction(aperture,wavelength,a,b,aperture_size,screen_size,N,mode='auto',d_char=None):
    '''
    mode="frensel"/"fraunhofer"/"auto" 
    d_char: characteristic aperture size for Fresnel number (auto if None)
    '''
    if d_char is None:
        d_char = aperture_size / 3
    N_F = d_char**2 / (wavelength * b)
    if mode=='auto':
            mode='frensel' if N_F>0.5 else 'fraunhofer'
    if mode=='frensel':
        intensity=frensel_diffraction(
            aperture,wavelength,a,b,aperture_size,screen_size,N)
        x=np.linspace(-screen_size/2,screen_size/2,N)
        y=np.linspace(-screen_size/2,screen_size/2,N)
        return intensity,x,y,'frensel',N_F
    else:
        intensity,x,y=fraunhofer_diffraction(
            aperture,wavelength,b,aperture_size,screen_size,N
        )
        return intensity,x,y,'fraunhofer',N_F
    
# STREAMLIT VISUAL FOR PICKMES
def main():
    st.set_page_config(page_title='Дифракция Френеля и Фраунгофера',layout='wide')
    st.title('Дифракция Френеля и Фраунгофера')
    
    with st.expander("Что делает FFT в этой программе?", expanded=False):
        st.markdown("""
        **Быстрое преобразование Фурье (FFT)** используется для численного расчёта дифракции:
        
        - **Дифракция Фраунгофера** (в дальней зоне): поле на экране = Фурье-образ апертуры
          $$U(\\xi,\\eta) \\propto \\iint A(x,y) e^{-i\\frac{2\\pi}{\\lambda b}(x\\xi + y\\eta)} dx\\,dy$$
          Это просто `fft2(aperture)` — FFT напрямую даёт угловой спектр.
        
        - **Дифракция Френеля** (ближняя зона): к результату FFT добавляются квадратичные фазовые множители (чирпы), учитывающие кривизну волнового фронта на расстоянии `b`.
        
        FFT позволяет вычислить эти интегралы за $O(N^2 \\log N)$ операций вместо $O(N^4)$ при прямом интегрировании.
        """)

    col1,col2=st.columns([1,2])
    
    with col1:
        st.subheader('Параметры')

        preset_names = ['---Выбор Пресета---'] + [p.name for p in PRESETS]
        selected_preset = st.selectbox('Пресет', preset_names, index=0)
        if selected_preset != '---Выбор Пресета--':
            preset_idx = preset_names.index(selected_preset) - 1
            p = PRESETS[preset_idx]
            default_wavelength = p.wavelength_nm
            default_a = p.a
            default_b = p.b
            default_aperture_size = p.screen_size_mm
            default_screen_size = p.screen_size_mm
            default_grid = p.grid_points
            default_aperture_type = p.aperture
            default_params = {k: v * 1000 for k, v in p.params.items()}
        else:
            default_wavelength = 532.0
            default_a = 1.0
            default_b = 1.0
            default_aperture_size = 5.0
            default_screen_size = 20.0
            default_aperture_type = ApertureType.SINGLE_SLIT
            default_params = {}

        wavelength_nm = st.number_input("Длина волны (нм)", value=default_wavelength, min_value=1.0)
        a = st.number_input("Расст. источник → отверстие (м)", value=default_a, min_value=0.01)
        b = st.number_input("Расст. отверстие → экран (м)", value=default_b, min_value=0.01)
        aperture_size_mm = st.number_input("Размер апертуры (мм)", value=default_aperture_size, min_value=0.1)
        screen_size_mm = st.number_input("Размер экрана (мм)", value=default_screen_size, min_value=0.1)
        grid_points = st.number_input("Точки сетки", value=default_grid, min_value=64, max_value=2048, step=64)

        st.subheader("Тип апертуры")
        aperture_type = st.selectbox(
            'aperture type',
            [t.value for t in ApertureType],
            index=list(ApertureType).index(default_aperture_type)
        )
        ap_type = ApertureType(aperture_type)

        st.subheader('Параметры апертуры')
        params = {}

        if ap_type == ApertureType.SINGLE_SLIT:
            params["width"] = st.number_input("Ширина щели (мм)", value=default_params.get("width", 0.1), min_value=0.001)
        elif ap_type == ApertureType.DOUBLE_SLIT:
            params["slit_width"] = st.number_input("Ширина щели (мм)", value=default_params.get("slit_width", 0.05), min_value=0.001)
            params["slit_separation"] = st.number_input("Расстояние между щелями (мм)", value=default_params.get("slit_separation", 0.2), min_value=0.001)
        elif ap_type == ApertureType.CIRCULAR:
            params["radius"] = st.number_input("Радиус (мм)", value=default_params.get("radius", 0.5), min_value=0.001)
        elif ap_type == ApertureType.RECTANGULAR:
            params["width_x"] = st.number_input("Ширина X (мм)", value=default_params.get("width_x", 1.5), min_value=0.001)
            params["width_y"] = st.number_input("Ширина Y (мм)", value=default_params.get("width_y", 0.5), min_value=0.001)
        elif ap_type == ApertureType.SQUARE_OBSTACLE:
            params["radius"] = st.number_input("Радиус диска (мм)", value=default_params.get("radius", 0.5), min_value=0.001)
        elif ap_type == ApertureType.DIFFRACTION_GRATING:
            params["period"] = st.number_input("Период решётки (мм)", value=default_params.get("period", 0.05), min_value=0.001)
            params["duty_cycle"] = st.slider("Заполнение (0-1)", 0.1, 0.9, default_params.get("duty_cycle", 0.5))
        elif ap_type == ApertureType.DOUBLE_RECT:
            params["width_x"] = st.number_input("Ширина X (мм)", value=default_params.get("width_x", 0.3), min_value=0.001)
            params["width_y"] = st.number_input("Высота Y (мм)", value=default_params.get("width_y", 0.4), min_value=0.001)
            params["separation"] = st.number_input("Расстояние между (мм)", value=default_params.get("separation", 0.8), min_value=0.001)
        elif ap_type == ApertureType.DOUBLE_CIRC:
            params["radius"] = st.number_input("Радиус (мм)", value=default_params.get("radius", 0.3), min_value=0.001)
            params["separation"] = st.number_input("Расстояние между (мм)", value=default_params.get("separation", 0.8), min_value=0.001)
        elif ap_type == ApertureType.CUSTOM:
            template = st.selectbox('Шаблон', [t['name'] for t in CUSTOM_TEMPLATES] + ['Своя формула'])
            if template == 'Своя формула':
                amp_formula = st.text_input('A(r) =', value='1')
                phase_formula = st.text_input('φ(r) =', value='pi*r^2/(wavelength*f)')
            else:
                t = next(x for x in CUSTOM_TEMPLATES if x['name'] == template)
                amp_formula = t['amp']
                phase_formula = t['phase']
            params['amp'] = amp_formula
            params['phase'] = phase_formula
            params['lambda'] = st.number_input('λ (нм)', value=500)
            params['f'] = st.number_input('f (мм)', value=500)
        st.subheader('Режим расчёта')
        mode = st.radio('Режим', ['auto', 'frensel', 'fraunhofer'], horizontal=True)
        show_theory = st.checkbox('Показать теорию', value=True)
        show_minima = st.checkbox('Показать минимумы', value=True)

    with col2:
        if True:
            try:
                wavelength = wavelength_nm * NM
                aperture_size = aperture_size_mm * MM
                screen_size = screen_size_mm * MM
                N = int(grid_points)

                params_m = {k: (v * MM if isinstance(v, (int, float)) else v) for k, v in params.items()}
                aperture = make_aperture(ap_type, N, aperture_size, params_m)

                d_char = get_aperture_char_size(ap_type, params_m, aperture_size)
                intensity, x, y, mode_used, N_F = compute_diffraction(aperture, wavelength, a, b, aperture_size, screen_size, N, mode=mode, d_char=d_char)

                fig = plt.figure(figsize=(12.8, 8.0))

                if np.iscomplexobj(aperture):
                    gs = GridSpec(2, 3, fig, hspace=0.3, wspace=0.3)
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1a = fig.add_subplot(gs[0, 1])
                    ax2 = fig.add_subplot(gs[0, 2])
                else:
                    gs = GridSpec(2, 2, fig, hspace=0.3, wspace=0.3)
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax2 = fig.add_subplot(gs[0, 1])

                extent = [-aperture_size_mm / 2, aperture_size_mm / 2,
                          -aperture_size_mm / 2, aperture_size_mm / 2]
                if np.iscomplexobj(aperture):
                    ax1.imshow(np.abs(aperture), cmap="gray", extent=extent, origin="lower")
                    ax1.set_title("Апертура (амплитуда)")

                    im_a = ax1a.imshow(np.angle(aperture), cmap="hsv", extent=extent, origin="lower")
                    ax1a.set_title("Апертура (фаза)")
                    fig.colorbar(im_a, ax=ax1a, label="фаза (рад)")
                else:
                    ax1.imshow(aperture, cmap="gray", extent=extent, origin="lower")
                    ax1.set_title("Апертура")

                ax1.set_xlabel("мм")
                ax1.set_ylabel("мм")
                ax1.set_aspect("equal")
                extent = [-screen_size_mm / 2, screen_size_mm / 2,
                          -screen_size_mm / 2, screen_size_mm / 2]
                im = ax2.imshow(intensity, cmap="hot", extent=extent, origin="lower", vmin=0, vmax=1)
                ax2.set_xlabel("мм")
                ax2.set_ylabel("мм")
                mode_label = "Френель" if mode_used == "frensel" else "Фраунгофер"
                ax2.set_title(f"Интенсивность ({mode_label}, λ={wavelength_nm}нм, b={b}м, N_F={N_F:.2f})", y=1)
                fig.colorbar(im, ax=ax2, label="I / I_max")

                ax3 = fig.add_subplot(gs[1, :])
                mid = len(y) // 2
                x_mm = np.linspace(-screen_size_mm / 2, screen_size_mm / 2, len(x))
                ax3.plot(x_mm, intensity[mid, :], "b-", linewidth=1, label='Численно (FFT)')
                ax3.set_xlabel("Положение на экране (мм)")
                ax3.set_ylabel("I / I_max")
                ax3.set_title("Профиль интенсивности (центральное сечение)")
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim(-screen_size_mm / 2, screen_size_mm / 2)

                if show_theory and mode_used == 'fraunhofer':
                    x_th = x_mm * 1e-3
                    if ap_type == ApertureType.SINGLE_SLIT:
                        w = params_m.get("width", 0.1e-3)
                        I_th = theoretical_slit_profile(x_th, w, wavelength, b)
                        ax3.plot(x_mm, I_th, "r--", lw=1.5, alpha=0.7, label='Теория (sinc²)')
                    elif ap_type == ApertureType.DOUBLE_SLIT:
                        sw = params_m.get("slit_width", 0.05e-3)
                        ss = params_m.get("slit_separation", 0.2e-3)
                        I_th = theoretical_double_slit_profile(x_th, sw, ss, wavelength, b)
                        ax3.plot(x_mm, I_th, "r--", lw=1.5, alpha=0.7, label='Теория (sinc²·cos²)')
                    elif ap_type == ApertureType.CIRCULAR:
                        radius = params_m.get("radius", 0.5e-3)
                        I_th = theoretical_circular_profile(np.abs(x_th), radius, wavelength, b)
                        ax3.plot(x_mm, I_th, "r--", lw=1.5, alpha=0.7, label='Теория (Эйри)')
                    elif ap_type == ApertureType.RECTANGULAR:
                        wx = params_m.get("width_x", 1.5e-3)
                        I_th = theoretical_rectangular_profile(x_th, wx, wavelength, b)
                        ax3.plot(x_mm, I_th, "r--", lw=1.5, alpha=0.7, label='Теория (sinc²)')
                    elif ap_type == ApertureType.DIFFRACTION_GRATING:
                        period = params_m.get("period", 0.1e-3)
                        duty = params_m.get("duty_cycle", 0.5)
                        N_lines = max(2, int(aperture_size / period))
                        I_th = theoretical_grating_profile(x_th, period, duty, wavelength, b, N_lines)
                        ax3.plot(x_mm, I_th, "r--", lw=1.5, alpha=0.7, label=f'Теория (решётка N={N_lines})')
                    elif ap_type == ApertureType.DOUBLE_RECT:
                        wx = params_m.get("width_x", 0.3e-3)
                        sep = params_m.get("separation", 0.8e-3)
                        I_th = theoretical_double_rect_profile(x_th, wx, sep, wavelength, b)
                        ax3.plot(x_mm, I_th, "r--", lw=1.5, alpha=0.7, label='Теория (sinc²·cos²)')
                    elif ap_type == ApertureType.DOUBLE_CIRC:
                        radius = params_m.get("radius", 0.3e-3)
                        sep = params_m.get("separation", 0.8e-3)
                        I_th = theoretical_double_circular_profile(x_th, radius, sep, wavelength, b)
                        ax3.plot(x_mm, I_th, "r--", lw=1.5, alpha=0.7, label='Теория (Эйри·cos²)')

                if show_minima and (ap_type == ApertureType.SINGLE_SLIT or ap_type == ApertureType.CIRCULAR):
                    slit_width = params_m.get("width", params_m.get("radius", 0.1e-3)) * 2
                    minima_pos = find_minima_positions(slit_width, wavelength, b, screen_size)
                    for pos in minima_pos:
                        if pos < screen_size_mm / 2:
                            ax3.axvline(x=pos, color='green', linestyle=':', alpha=0.5, linewidth=0.8)
                            ax3.axvline(x=-pos, color='green', linestyle=':', alpha=0.5, linewidth=0.8)

                ax3.legend(loc='upper right', fontsize=9)

                st.pyplot(fig)
            except Exception as e:
                st.error(f"Ошибка при расчёте: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    print(' run python -m streamlit run diffraction_task_2.py')
    main()