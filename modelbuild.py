# ==========================
# 模型定义
# ==========================
from astromodels import (
    Band,
    Blackbody,
    Cutoff_powerlaw,
    Function1D,
    FunctionMeta,
    Log_uniform_prior,
    Model,
    NonDissipativePhotosphere,
    PointSource,
    Powerlaw,
    Uniform_prior,
)
import astropy.units as astropy_units
import numpy as np
from scipy.integrate import quad
from threeML import *

GBM_ENERGY_BOUNDS_KEV = (8.0, 4.0e4)
GBM_LAT_ENERGY_BOUNDS_KEV = (8.0, 1.0e8)


def _energy_bounds_for_mode(analysis_mode):
    if "lat" in str(analysis_mode or "").lower():
        return GBM_LAT_ENERGY_BOUNDS_KEV
    return GBM_ENERGY_BOUNDS_KEV


def _set_energy_prior(parameter, bounds):
    lower = float(bounds[0])
    upper = float(bounds[1])
    current_value = getattr(parameter, "value", None)
    if current_value is not None:
        if current_value < lower:
            parameter.value = lower
        elif current_value > upper:
            parameter.value = upper
    parameter.min_value = lower
    parameter.max_value = upper
    parameter.prior = Log_uniform_prior(lower_bound=lower, upper_bound=upper)


class MultiColorBlackBody(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        A multi-color blackbody (mBB) model assuming a power-law distribution 
        of thermal luminosities with temperature. 
        Based on Hou et al. (2018) for GRB prompt emission.

    latex : $ N(E)=\frac{8.0525(m+1)K}{[ ( \frac{T_{max}}{T_{min}} )^{m+1}-1 ]} ( \frac{kT_{min}}{\text{keV}} )^{-2} I(E) $

    parameters :
        K :
            desc : Normalization (related to total luminosity L_mBB)
            initial value : 1.0
            is_normalization : True
            transformation : log10
            min : 1e-30
            max : 1e5
        kT_min :
            desc : Minimum temperature
            initial value : 5.0
            min : 0.1
            max : 100.0
            unit : keV
        kT_max :
            desc : Maximum temperature
            initial value : 50.0
            min : 1.0
            max : 500.0
            unit : keV
        m :
            desc : Power law index of the temperature distribution
            initial value : -0.5
            min : -5.0
            max : 1.9  # Physical constraint: m < 2
    """

    def _set_units(self, x_unit, y_unit):
        # Temperatures have the same units as x (typically keV)
        self.kT_min.unit = x_unit
        self.kT_max.unit = x_unit

        # m is dimensionless
        self.m.unit = astropy_units.dimensionless_unscaled

        # Normalization unit
        self.K.unit = y_unit

    def evaluate(self, x, K, kT_min, kT_max, m):
        # Handle units if present
        if isinstance(x, astropy_units.Quantity):
            x_ = x.value
            K_ = K.value
            kT_min_ = kT_min.value
            kT_max_ = kT_max.value
            m_ = m.value
            unit_ = self.y_unit
        else:
            x_, K_, kT_min_, kT_max_, m_ = x, K, kT_min, kT_max, m
            unit_ = 1.0

        # Constants from the paper
        const = 8.0525
        
        # Pre-factor calculation
        # [ (Tmax/Tmin)^(m+1) - 1 ]
        ratio_term = np.power(kT_max_ / kT_min_, m_ + 1) - 1.0
        
        prefactor = (const * (m_ + 1) * K_) / (ratio_term * np.power(kT_min_, 2))

        # Define the integrand for I(E)
        # x_int = E / kT
        def integrand(x_int, m_val):
            # Avoid overflow in exp
            if x_int > 700:
                return 0.0
            return np.power(x_int, 2.0 - m_val) / (np.exp(x_int) - 1.0)

        # Vectorized integration for I(E)
        def compute_I(energy):
            lower = energy / kT_max_
            upper = energy / kT_min_
            
            # Numerical integration of equation (5)
            res, _ = quad(integrand, lower, upper, args=(m_,))
            
            return np.power(energy / kT_min_, m_ - 1.0) * res

        # Apply to all energy values
        if isinstance(x_, np.ndarray):
            i_e = np.array([compute_I(e) for e in x_])
        else:
            i_e = compute_I(x_)

        return prefactor * i_e * unit_


def build_model(mstr, parameters=None, analysis_mode="gbm"):
    """
    支持单模型或双模型组合：
    例：
        'band'
        'comp'
        'blackbody'
        'band+blackbody'
        'comp+pl'

    能量量纲参数的先验和硬边界按实际拟合模式设置：
        GBM: 8-40000 keV
        GBM+LAT: 8-1e8 keV
    """
    energy_bounds = _energy_bounds_for_mode(analysis_mode)
    # def make_one(mstr):
    if mstr == "band":
        m = Band(piv=1E2)
        parameters = [1E-4,-1.0,500.0,-2.0]
        # m.piv.free = True
        # m.piv = 1E2
        m.K.prior = Log_uniform_prior(lower_bound=parameters[0]*1E-3, upper_bound=1E1)
        m.alpha.prior = Uniform_prior(lower_bound=parameters[1]-0.5, upper_bound=parameters[1]+2)
        m.beta.prior = Uniform_prior(lower_bound=parameters[3]-3, upper_bound=parameters[3]+0.4)
        # m.piv.prior = Uniform_prior(lower_bound=1, upper_bound=1e6)

        m.K, m.alpha, m.xp, m.beta = parameters[0], parameters[1], parameters[2], parameters[3]
        # 设置参数边界以避免采样问题
        m.alpha.min_value, m.alpha.max_value = -1.5, 2.0
        m.beta.min_value, m.beta.max_value = -5, -1.6
        _set_energy_prior(m.xp, energy_bounds)

    elif mstr == "comp":
        m = Cutoff_powerlaw(piv=1E2)
        parameters = [1E-4, -2.0, 10.0]
        m.K.prior = Log_uniform_prior(lower_bound=1E-9, upper_bound=1E3)
        m.index.prior = Uniform_prior(lower_bound=-2, upper_bound=0)
        m.K, m.index, m.xc = parameters[0], parameters[1], parameters[2]
        # 设置参数边界以避免采样问题
        m.index.min_value, m.index.max_value = -5.0, 0
        m.K.min_value, m.K.max_value = 1e-12, 1e6
        _set_energy_prior(m.xc, energy_bounds)

    elif mstr == "blackbody" or mstr == 'bb':
        
        m = Blackbody()
        parameters = [1E-4, 30.0]
        m.K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
        # 设置参数边界
        m.K, m.kT = parameters[0], parameters[1]
        m.K.min_value, m.K.max_value = 1e-12, 1e3
        _set_energy_prior(m.kT, energy_bounds)

    elif mstr == 'NDP':
        m = NonDissipativePhotosphere(piv=1E2)
        parameters = [1E-4, 200.0]
        m.K.prior = Log_uniform_prior(lower_bound=1E-9, upper_bound=1E1)
        m.K, m.ec = parameters[0], parameters[1]
        _set_energy_prior(m.ec, energy_bounds)

    elif mstr == "pl":
        m = Powerlaw(piv=1E2)             #Powerlaw()
        parameters = [1E-4, -2.0]
        # m.piv.free = True
        m.K.prior = Log_uniform_prior(lower_bound=1E-7, upper_bound=1)
        # m.piv.prior = Uniform_prior(lower_bound=1, upper_bound=1e6)
        m.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=0.0)
        m.K,  m.index = parameters[:2]
        m.K.min_value, m.K.max_value = 1e-8, 1e4
        # m.piv.min_value, m.piv.max_value = 1.0, 1e6

    elif mstr == "SBPL":
        m = SmoothlyBrokenPowerLaw()
        parameters = [1E-4, -1.0, 500.0, -2.0, 0.5]  # 新增一个参数控制“平滑度”或“转折锐度”
        m.K.prior = Log_uniform_prior(lower_bound=parameters[0]*1E-3, upper_bound=parameters[0]*1E3)
        m.beta.prior = Truncated_gaussian(lower_bound=-5.0, upper_bound=-1.6, mu=parameters[3], sigma=0.5)
        m.alpha.prior = Truncated_gaussian(lower_bound=-1.5, upper_bound=3.0, mu=parameters[1], sigma=0.5)
        # 初始化参数值
        m.K, m.alpha, m.break_energy, m.beta, m.break_scale = (
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
        )
        m.alpha.min_value, m.alpha.max_value = -2, 5.0
        m.beta.min_value, m.beta.max_value = -10, -1.5
        _set_energy_prior(m.break_energy, energy_bounds)

    elif mstr == "band+bb":
        parameters = [1E-3, -1.0, 500.0, -2.0,# Band: K, alpha, xp, beta，piv
                      1E-4, 30.0]                 # BB: K_bb(keV), T_bb(keV)
        # m.K, m.alpha, m.xp, m.beta = parameters[0], parameters[1], parameters[2], parameters[3]
        # m.K_bb, m.T_bb = parameters[4], parameters[5]

        band = Band(piv=1E2)
        bb = Blackbody()

        # band.piv.free = True
        band.K.prior = Log_uniform_prior(lower_bound=parameters[0]*1E-3, upper_bound=parameters[0]*1E3)
        band.alpha.prior = Uniform_prior(lower_bound=parameters[1]-0.5, upper_bound=parameters[1]+3)
        band.beta.prior =  Truncated_gaussian(lower_bound=-5.0, upper_bound=-1.6, mu=parameters[3], sigma=0.5)
        # band.piv.prior = Uniform_prior(lower_bound=1, upper_bound=1e8)
        # band.K.min_value, band.K.max_value = 1e-7, 1
        # band.alpha.min_value, band.alpha.max_value = -1.6, 2.0
        # band.beta.min_value, band.beta.max_value = -5, -1.6
        
        # band.piv.min_value, band.piv.max_value = 1.0, 1e6
        band.K, band.alpha, band.xp, band.beta  = parameters[0], parameters[1], parameters[2], parameters[3]
        _set_energy_prior(band.xp, energy_bounds)
        

        # Blackbody部分参数设置
        bb.K.prior = Log_uniform_prior(lower_bound=1e-7, upper_bound=1)
        bb.K, bb.kT = parameters[4], parameters[5]
        _set_energy_prior(bb.kT, energy_bounds)
        # bb.K.min_value, bb.K.max_value = 1e-7, 1
        
        m = band + bb

    elif mstr == 'mbb':
        m = MultiColorBlackBody()
        parameters = [1E-6, 8, 100 ,0.0]             # MBB: K_mbb, kmin, kmax, m
        m.K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
        m.m.prior = Uniform_prior(lower_bound=-2.5, upper_bound=1)
        m.K, m.kT_min, m.kT_max, m.m = parameters[0], parameters[1], parameters[2], parameters[3]
        _set_energy_prior(m.kT_min, energy_bounds)
        _set_energy_prior(m.kT_max, energy_bounds)

    elif mstr == 'band+mbb':
        band = Band(piv=1E2)
        mbb = MultiColorBlackBody()

        parameters = [1E-5, -1.0, 1000.0, -2.0,  # Band: K, alpha, xp, beta
                      1E-6, 8, 200 ,0.0]             # MBB: K_mbb, kmin, kmax, m

        band.K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
        band.alpha.prior = Uniform_prior(lower_bound=parameters[1]-0.5, upper_bound=parameters[1]+2)
        band.beta.prior =  Uniform_prior(lower_bound=parameters[3]-2, upper_bound=parameters[3]+0.4)
        # band.piv.prior = Uniform_prior(lower_bound=1, upper_bound=1e8)
        band.K.min_value, band.K.max_value = 1e-50, 1e3
        band.alpha.min_value, band.alpha.max_value = -1.5, 3.0
        band.beta.min_value, band.beta.max_value = -5, -1.6
        
        # band.piv.min_value, band.piv.max_value = 1.0, 1e6
        band.K, band.alpha, band.xp, band.beta  = parameters[0], parameters[1], parameters[2], parameters[3]
        _set_energy_prior(band.xp, energy_bounds)

        mbb.K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
        mbb.m.prior = Uniform_prior(lower_bound=-2.5, upper_bound=1)
        mbb.K, mbb.kT_min, mbb.kT_max, mbb.m = parameters[4], parameters[5], parameters[6], parameters[7]
        _set_energy_prior(mbb.kT_min, energy_bounds)
        _set_energy_prior(mbb.kT_max, energy_bounds)

        m = band + mbb


    elif mstr == "mbb+pl":
        mbb = MultiColorBlackBody()
        pl = Powerlaw(piv=1E2)
        parameters = [1E-6, 8, 500, 0.0, 1e-4, -2.0]
        mbb.K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
        mbb.m.prior = Uniform_prior(lower_bound=-2.5, upper_bound=1)
        mbb.K, mbb.kT_min, mbb.kT_max, mbb.m = (
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
        )
        _set_energy_prior(mbb.kT_min, energy_bounds)
        _set_energy_prior(mbb.kT_max, energy_bounds)
        pl.K.prior = Log_uniform_prior(lower_bound=1e-8, upper_bound=1e1)
        pl.index.prior = Truncated_gaussian(
            lower_bound=-10.0, upper_bound=10.0, mu=parameters[5], sigma=0.5
        )
        pl.K.min_value, pl.K.max_value = 1e-8, 1e1
        pl.index.min_value, pl.index.max_value = -10, 10
        pl.K, pl.index = parameters[4], parameters[5]
        m = mbb + pl

    elif mstr == "band+pl":
        # 组合模型：Band + Powerlaw
        band = Band(piv=1e2)
        pl = Powerlaw(piv=1e2)
        parameters = [1E-2,-1.0,500.0,-2.0, 1e-4, -2.0]
        # m.piv_1 = 1E2
        # m.piv_1.free = True
        band.K.prior = Log_uniform_prior(lower_bound=1e-7, upper_bound=1e3)
        band.alpha.prior = Uniform_prior(lower_bound=parameters[1]-0.5, upper_bound=parameters[1]+2)
        band.beta.prior =  Uniform_prior(lower_bound=parameters[3]-2, upper_bound=parameters[3]+0.4)
        # band.piv.prior = Uniform_prior(lower_bound=1, upper_bound=1e8)
        band.K.min_value, band.K.max_value = 1e-50, 1e3
        band.alpha.min_value, band.alpha.max_value = -1.5, 3.0
        band.beta.min_value, band.beta.max_value = -5, -1.6

        pl.K.prior = Log_uniform_prior(lower_bound=1e-8, upper_bound=1e1)
        pl.index.prior = Truncated_gaussian(lower_bound=-10.0, upper_bound=10.0, mu=parameters[5], sigma=0.5)

        pl.K.min_value,pl.K.max_value = 1e-8 ,1e1
        pl.index.min_value,pl.index.max_value = -10, 10
        band.K, band.alpha, band.xp, band.beta, pl.K, pl.index = parameters[:6]
        _set_energy_prior(band.xp, energy_bounds)
        m = band + pl

        
    elif mstr == "comp+bb":
        # 组合模型：Cutoff_powerlaw + Blackbody
        comp = Cutoff_powerlaw(piv=1E2)
        bb = Blackbody()
        parameters = [1E-4, -2.0, 10.0, 1E-3, 30.0]
        # Cutoff_powerlaw部分参数设置
        comp.K.prior = Log_uniform_prior(lower_bound=1e-7, upper_bound=1e1)
        comp.index.prior = Uniform_prior(lower_bound=parameters[1]-0.5, upper_bound=parameters[1]+1)
        comp.K.min_value, comp.K.max_value = 1e-12, 1e1
        comp.index.min_value, comp.index.max_value = -5.0, 0.0
        comp.K, comp.index, comp.xc = parameters[0], parameters[1], parameters[2]
        _set_energy_prior(comp.xc, energy_bounds)

        # Blackbody部分参数设置
        bb.K.prior = Uniform_prior(lower_bound=1E-9,
                                        upper_bound=1E1)
        bb.K, bb.kT = parameters[3], parameters[4]
        bb.K.min_value, bb.K.max_value = 1e-9, 1e1
        _set_energy_prior(bb.kT, energy_bounds)
        
        m = comp + bb
    elif mstr == "comp+pl":
        # 组合模型：Cutoff_powerlaw + Powerlaw
        m = Cutoff_powerlaw(piv=1E2)+Powerlaw(piv=1E2)
        # 根据贝叶斯结果调整初始参数值和先验设置
        parameters = [1E-4,  -1.0, 100.0 ,1E-4, -2.0]

        # m.piv_1.free = True
        m.K_1.prior = Log_uniform_prior(lower_bound=1e-7, upper_bound=1e1)
        # m.piv_1.prior = Uniform_prior(lower_bound=1, upper_bound=1e6)
        m.index_1.prior = Uniform_prior(lower_bound=parameters[1]-0.5, upper_bound=parameters[1]+0.5)
        m.K_2.prior = Log_uniform_prior(lower_bound=1e-8, upper_bound=1e1)
        m.index_2.prior = Truncated_gaussian(lower_bound=-10.0, upper_bound=10.0, mu=parameters[4], sigma=0.5)
 

        # 参数边界设置
        m.K_1.min_value, m.K_1.max_value = 1e-8, 1e1
        # m.piv_1.min_value, m.piv_1.max_value = 1, 1E6
        m.index_1.min_value, m.index_1.max_value = -5.0, 0
        m.K_2.min_value, m.K_2.max_value = 1e-8, 1e1
        m.index_2.min_value, m.index_2.max_value = -10.0, 10.0


        m.K_1,  m.index_1, m.xc_1, m.K_2, m.index_2  = parameters[:5]
        _set_energy_prior(m.xc_1, energy_bounds)

    elif mstr == "pl+bb":
        # 组合模型：Powerlaw + Blackbody
        m = Blackbody()+Powerlaw(piv=1E2)
        parameter_values = [1E-4, 30.0, 1E-4, -2.0]
        parameters = parameter_values
        m.K_1.prior = Log_uniform_prior(lower_bound=1E-9, upper_bound=1E1)
        # if GRBname == '140402A':
        #     m.K_1.prior = Log_uniform_prior(lower_bound=parameters[0]*1E-5, upper_bound=parameters[0]*1E5)   ## Only for 140402A BB+PL

        m.K_2.prior = Log_uniform_prior(lower_bound=1E-9, upper_bound=1E1)
        m.index_2.prior = Uniform_prior(lower_bound=-10.0, upper_bound=10.0)

        m.K_1,m.kT_1,m.K_2,m.index_2 = parameters[0], parameters[1], parameters[2], parameters[3]
        _set_energy_prior(m.kT_1, energy_bounds)

    elif mstr == "band+bb+pl":
        band = Band(piv=1E2)
        bb = Blackbody()
        pl = Powerlaw(piv=1E2)
        parameters = [1E-5, -1.0, 1000.0, -2.0,  # Band: K, alpha, xp, beta
                      1E-6, 30.0,              # BB: K_bb(keV), T_bb(keV)
                      1E-4, -2.0]               # PL: K_pl, index
        band.K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
        band.alpha.prior = Truncated_gaussian(lower_bound=parameters[1]-0.5, upper_bound=parameters[1]+0.5, mu=-1, sigma=0.5)
        band.beta.prior =  Uniform_prior(lower_bound=parameters[3]-2, upper_bound=parameters[3]+0.4)       
        band.K.min_value, band.K.max_value = 1e-10, 1e3
        band.alpha.min_value, band.alpha.max_value = -1.5, 0.0
        band.beta.min_value, band.beta.max_value = -5, -1.6
        band.K,band.alpha,band.xp,band.beta = parameters[:4]
        _set_energy_prior(band.xp, energy_bounds)

        bb.K.prior = Uniform_prior(lower_bound=1E-9,
                                        upper_bound=1E1)
        bb.K, bb.kT = parameters[4], parameters[5]
        bb.K.min_value, bb.K.max_value = 1e-9, 1e1
        _set_energy_prior(bb.kT, energy_bounds)


        pl.K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e3)
        pl.index.prior =  Truncated_gaussian(lower_bound=parameters[7]-0.5, upper_bound=parameters[7]+0.5, mu=-2, sigma=0.5)           
        pl.K.min_value, pl.K.max_value = 1e-10, 1e3
        pl.index.min_value, pl.index.max_value = -5, 5
        pl.K, pl.index = parameters[6], parameters[7]
        m = band + bb + pl


    elif mstr == "band+gauss":
        band = Band(piv=1E2)
        gauss = Gaussian()
        parameters = [1E-5, -1.0, 1000.0, -2.0, 1 , 1E4, 200.0]
        band.K.prior = Log_uniform_prior(lower_bound=1e-7, upper_bound=1)
        band.alpha.prior = Uniform_prior(lower_bound=parameters[1]-0.5, upper_bound=parameters[1]+1)
        band.beta.prior =  Uniform_prior(lower_bound=parameters[3]-2, upper_bound=parameters[3]+0.4)           
        band.K.min_value, band.K.max_value = 1e-10, 1e3
        band.alpha.min_value, band.alpha.max_value = -1.5, 0.0
        band.beta.min_value, band.beta.max_value = -5, -1.6
        band.K,band.alpha,band.xp,band.beta = parameters[:4]
        _set_energy_prior(band.xp, energy_bounds)
        gauss.F.prior = Log_uniform_prior(lower_bound=1e-8, upper_bound=1)
        # gauss.F.min_value, gauss.F.max_value = 1e-10, 1e4
        gauss.F,gauss.mu, gauss.sigma = parameters[4], parameters[5], parameters[6]
        _set_energy_prior(gauss.mu, energy_bounds)
        _set_energy_prior(gauss.sigma, energy_bounds)
        m = band + gauss
    # elif mstr == "band+bb+pl":
    # elif mstr == "csbpl":
    #     csbpl = CSBPL()
    #     csbpl.K.value = 1e-2
    #     csbpl.E_b.value = 300          # keV
    #     csbpl.alpha.value = -0.5
    #     csbpl.beta.value = 2.0
    #     csbpl.gamma.value = -2.3
    
    else:
        raise ValueError(f"未知模型类型: {mstr}")
    # return m

    # parts = model_str.split("+")
    # if len(parts) > 1:
    #     # 处理组合模型
    #     combined = make_one(parts[0])
    #     for sub in parts[1:]:
    #         combined = combined + make_one(sub)
    #     m = combined
    # else:
    #     # 处理单一模型
    #     m = make_one(model_str)

    # print(f'加载模型: {str(m)}')

    # source = PointSource(f"GRB{name}", ra, dec, spectral_shape=m)   
    return m

# from astromodels import Function1D, FunctionMeta, Parameter
# import numpy as np

# class CSBPL(Function1D, metaclass=FunctionMeta):
#     def _set_units(self, x_unit, y_unit):
#         self.x_unit = x_unit
#         self.y_unit = y_unit

#     # 正确写法：value 放第一位，不用 default=
#     pl_index = Parameter(-2.0, min_value=-10, max_value=10, delta=0.1)
#     pl_K     = Parameter(1e-2, min_value=1e-10, max_value=1e10, delta=1e-3)
#     bb_T     = Parameter(30.0, min_value=0.1, max_value=1000, delta=1.0)
#     bb_K     = Parameter(1e-2, min_value=1e-10, max_value=1e10, delta=1e-3)
#     norm1    = Parameter(1.0,  min_value=0,     max_value=1e10, delta=0.1)
#     norm2    = Parameter(1.0,  min_value=0,     max_value=1e10, delta=0.1)

#     def evaluate(self, x, pl_index, pl_K, bb_T, bb_K, norm1, norm2):
#         f1 = pl_K * (x ** pl_index)
#         bb = bb_K * x**3 / (np.exp(x / bb_T) - 1.0)
#         return norm1 * f1 + norm2 * bb

#     def _fix_units(self, x, y):
#         return x * self.x_unit, y * self.y_unit
