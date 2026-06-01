### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 00000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ deps-cell
begin
    using Plots
    using PlutoUI
    using QuadGK
    using Roots
    using DataFrames
    using Statistics
end

# ╔═╡ title-cell
md"""
# Planetary Nebula [O III] λ5007 — Physical Luminosity Model

**Physics included:**
1. Saha ionization equilibrium (corrected CGS prefactor, T in Kelvin)
2. Collisional excitation rate q(T) with Boltzmann factor exp(−ΔE/kT)
3. Collisional de-excitation quenching: ε/(1 + nₑ/n\_crit(T))
4. Self-consistent Strömgren truncation — emissivity zeroed beyond r\_S,
   where r\_S is solved from the stellar ionizing photon budget Q(H⁰)
5. Post-AGB central star evolution: L★(t̄, M\_cs) and T★(t̄, M\_cs)
6. Wind-anchored density: n₀ from Ṁ/(4π r²\_★ vw μ mH); log(n₀) ~ 14
7. **Shock-derived σ(t) and b(t)**: shell width grows as the shock
   decelerates (Rankine-Hugoniot + Villaver+2002); ionization-front
   sharpness b falls as nₑ drops with expansion. Neither is a free
   parameter — both are computed from physical shock inputs.
"""

# ╔═╡ constants-cell
begin
    # ── Physical constants ───────────────────────────────────────────────────
    const kB_eV   = 8.617333e-5    # eV K⁻¹
    const eV2K    = 1.0 / kB_eV   # K per eV
    const eV2erg  = 1.602176e-12   # erg per eV

    # Saha prefactor: 2·(2π mₑ kB / h²)^(3/2)  [cm⁻³ K⁻³/²]
    const SAHA_A  = 4.8288e15

    # ── Oxygen atomic data ───────────────────────────────────────────────────
    const g_O   = [9, 4, 1, 2, 1, 2]
    const E_ion = [13.618, 35.121, 54.936, 77.413, 113.899]   # eV

    # ── [O III] λ5007 transition ─────────────────────────────────────────────
    const E5007_eV  = 2.4792
    const hν5007    = E5007_eV * eV2erg      # erg

    # Effective collision strength (Mendoza 1983; Lennon & Burke 1994)
    const Ω_OIII   = 2.29
    const g_upper  = 5

    # Einstein A coefficient for [O III] λ5007  (Storey & Zeippen 2000)
    const A_ul_5007 = 0.02101    # s⁻¹

    # ── Hydrogen recombination ───────────────────────────────────────────────
    # Case B at T ~ 10⁴ K
    const α_B     = 2.6e-13     # cm³ s⁻¹

    # ── Nebula length scale ──────────────────────────────────────────────────
    const r_star_cm = 1.0e17    # cm  (dimensionless r̄ unit)

    # ── PNLF empirical cutoff ────────────────────────────────────────────────
    const logL_sun        = 33.582          # log₁₀(L☉ / erg s⁻¹)
    const M_sun_5007      = 6.87            # absolute mag of Sun in [O III] band
    const M_star_ciardullo = -4.47          # Ciardullo et al. 2002
    const logL_PNLF_star  = logL_sun + (M_sun_5007 - M_star_ciardullo) / 2.5

    nothing
end

# ╔═╡ saha-cell
md"""
## 1 · Saha Ionization Equilibrium

$$\frac{n_{i+1}\,n_e}{n_i} = A\,T_K^{3/2}\,\frac{g_{i+1}}{g_i}\exp\!\left(-\frac{\chi_i}{kT}\right)$$

Prefactor $A = 4.829\times10^{15}\ \mathrm{cm^{-3}\,K^{-3/2}}$; energies in eV; temperature
converted to K before applying the prefactor.
"""

# ╔═╡ saha-fn-cell
"""
    saha_ratios(nₑ, T_eV) → Vector{Float64}

Number fractions [f₀…f₅] for O⁰ through O⁵ in Saha equilibrium.
"""
function saha_ratios(nₑ::Float64, T_eV::Float64)
    (T_eV <= 0.0 || nₑ <= 0.0) && return fill(NaN, 6)
    T_K    = T_eV * eV2K
    factor = SAHA_A * T_K^1.5 / nₑ
    chain  = ones(Float64, 6)
    for i in 1:5
        chain[i+1] = chain[i] * factor * (g_O[i+1] / g_O[i]) * exp(-E_ion[i] / T_eV)
    end
    Z_tot = sum(chain)
    return chain ./ Z_tot
end

# ╔═╡ qeff-cell
md"""
## 2 · Collisional Excitation Rate

$$q_{12}(T) = \frac{8.629\times10^{-6}}{\sqrt{T_K}}\cdot\frac{\Omega}{g_u}\cdot\exp\!\left(-\frac{\Delta E}{kT}\right)$$

The exponential **must** be present: this is the excitation coefficient.
The form without exp(−ΔE/kT) is the *de*-excitation coefficient q₂₁,
related by detailed balance q₁₂ = (gᵤ/gₗ)·q₂₁·exp(−ΔE/kT).
"""

# ╔═╡ qeff-fn-cell
"""
    q_eff(T_eV) → Float64

Collisional excitation rate coefficient for [O III] λ5007 (cm³ s⁻¹).
Includes the Boltzmann suppression factor exp(−E5007/T).
"""
function q_eff(T_eV::Float64)
    T_eV <= 0.0 && return 0.0
    T_K = T_eV * eV2K
    return 8.629e-6 / sqrt(T_K) * Ω_OIII / g_upper * exp(-E5007_eV / T_eV)
end

# ╔═╡ ncrit-cell
md"""
## 3 · Collisional De-excitation: Critical Density

At $n_e \gg n_\mathrm{crit}$ the upper level is thermalised by collisions before
it can radiate, suppressing the line: $\varepsilon \to \varepsilon/(1 + n_e/n_\mathrm{crit})$.

$$n_\mathrm{crit}(T) = \frac{A_{ul}}{q_{ul}(T)}$$

For [O III] λ5007: $A_{ul} = 0.021\ \mathrm{s^{-1}}$, giving
$n_\mathrm{crit} \approx 6.8\times10^5\ \mathrm{cm^{-3}}$ at $T = 10^4$ K.
The T-dependent form is used so the quenching threshold varies across the shell profile.
"""

# ╔═╡ ncrit-fn-cell
"""
    ne_crit(T_eV) → Float64

T-dependent critical density (cm⁻³) for [O III] λ5007.
n_crit = A_ul / q_ul  where q_ul is the de-excitation rate coefficient.
"""
function ne_crit(T_eV::Float64)
    T_eV <= 0.0 && return Inf
    T_K  = T_eV * eV2K
    # de-excitation rate: q_ul = 8.629e-6 * Ω / (g_upper * sqrt(T_K))
    # (no Boltzmann factor: this is the downward rate)
    q_ul = 8.629e-6 * Ω_OIII / (g_upper * sqrt(T_K))
    return A_ul_5007 / q_ul
end

# ╔═╡ profiles-cell
md"""
## 4 · Density and Temperature Profiles

**Wind-anchored density.** From mass continuity $\dot{M} = 4\pi r^2 \rho v_w$:

$$n_e(r) = \frac{\dot{M}}{4\pi r^2 v_w \mu m_H} \equiv \frac{n_0}{(r+\varepsilon)^2}$$

For $\dot{M}=10^{-5}\ M_\odot\ \mathrm{yr}^{-1}$, $v_w=10\ \mathrm{km\,s}^{-1}$,
$r_\star=10^{14}\ \mathrm{cm}$: $\log n_0 \approx 14$.
The swept shell compresses this into a Gaussian of width $\sigma(t)$ at $r=t$:

$$\bar{n}(r,t) = \frac{n_0}{(r+\varepsilon)^2}
  \cdot \frac{1}{\sqrt{\pi}\,\sigma(t)}\exp\!\left[-\frac{(r-t)^2}{\sigma(t)^2}\right]$$

**Shock-derived shell width $\sigma(t)$.**
For a strong adiabatic shock the compression ratio is $(\gamma+1)/(\gamma-1) = 4$,
giving $\Delta r/r \approx 1/4$ initially. As the shock decelerates into the
$\rho\propto r^{-2}$ AGB wind the compression weakens and the shell broadens
(Villaver et al. 2002):

$$\sigma(t) = \sigma_0 \left(1 + \frac{t}{t_\mathrm{thin}}\right)^\beta$$

$\sigma_0 \approx 0.02$–$0.05$ (strong-shock limit), $\beta \approx 0.3$–$0.5$.

**Shock-derived ionization-front sharpness $b(t)$.**
The sigmoid width $\sim 1/b$ equals the ionization-front thickness
$\ell_\mathrm{IF} = v_\mathrm{IF}/(\alpha_B n_e)$.
Since $n_e \propto n_0/(t+\varepsilon)^2$ at the shell peak, $b$ falls with time
as the density drops:

$$b(t) = \frac{r_\star \cdot \alpha_B \cdot n_e^\mathrm{peak}(t)}{v_\mathrm{IF}}$$

clamped to $[1, 50]$. A fast R-type front ($v_\mathrm{IF} \sim 100\ \mathrm{km\,s}^{-1}$)
gives a sharp early transition; the D-type phase ($v_\mathrm{IF} \sim 10\ \mathrm{km\,s}^{-1}$)
broadens it. Neither $\sigma$ nor $b$ is a free geometric parameter.
"""

# ╔═╡ profiles-fn-cell
"""
    n_profile(r, t, n0, σ; ε) → Float64

Wind-anchored electron density (cm⁻³). r⁻² wind × Gaussian shell at r = t.
ε softens the origin singularity; no ε² numerator (that was a normalization artefact).
"""
function n_profile(r::Float64, t::Float64, n0::Float64, σ::Float64; ε::Float64=0.02)
    r < 0.0 && return 0.0
    wind  = 1.0 / (r + ε)^2
    shell = exp(-(r - t)^2 / σ^2) / (sqrt(π) * σ)
    return n0 * wind * shell
end

function T_profile(r::Float64, t::Float64, T_in::Float64, T_out::Float64, b::Float64)
    return T_in - (T_in - T_out) / (1.0 + exp(-b * (r - t)))
end

# ╔═╡ shock-cell
md"## 4b · Shock-Derived σ(t) and b(t)"

# ╔═╡ shock-fn-cell
"""
    sigma_shell(t; sigma0, t_thin, beta) → Float64

Shell thickness (dimensionless) as a function of evolutionary time t̄.
Grows as the shock decelerates from the strong (thin) to weak (broad) limit.

  sigma0  : initial width at t → 0  (strong Rankine-Hugoniot shock, ≈ 0.02–0.05)
  t_thin  : shock-weakening timescale in t̄ units  (≈ 1–3)
  beta    : broadening exponent from hydro simulations  (≈ 0.3–0.5, Villaver+2002)
"""
function sigma_shell(t::Float64;
                     sigma0::Float64 = 0.03,
                     t_thin::Float64 = 1.5,
                     beta::Float64   = 0.4)
    return sigma0 * (1.0 + t / t_thin)^beta
end

"""
    b_transition(t, n0; v_IF_kms) → Float64

Temperature-transition sharpness b(t) from ionization-front physics.
b ~ r_★ · α_B · nₑ_peak(t) / v_IF, where nₑ_peak uses the current σ(t).
Falls with time as the shell density drops with expansion.
Clamped to [1, 50] for numerical stability.

  v_IF_kms : ionization-front speed (km/s)
             ~100  R-type (early, star heating up)
             ~10   D-type (settled, pressure-driven)
"""
function b_transition(t::Float64, n0::Float64;
                      sigma0::Float64   = 0.03,
                      t_thin::Float64   = 1.5,
                      beta::Float64     = 0.4,
                      v_IF_kms::Float64 = 20.0)
    σ_t    = sigma_shell(t; sigma0, t_thin, beta)
    # Peak density at shell centre r = t
    ne_pk  = n_profile(t, t, n0, σ_t)
    v_IF   = v_IF_kms * 1.0e5              # km/s → cm/s
    # IF thickness in dimensionless units: ℓ = v_IF / (α_B · nₑ · r_★)
    ne_pk  = max(ne_pk, 1.0)              # guard against zero
    ell_IF = v_IF / (α_B * ne_pk * r_star_cm)
    return clamp(1.0 / max(ell_IF, 1e-6), 1.0, 50.0)
end

# ╔═╡ postaGB-cell
md"""
## 5 · Post-AGB Stellar Evolution

Parameterised tracks following Schönberner (1983) and Vassiliadis & Wood (1994).

- Peak luminosity: $\log(L_\star/L_\odot) \approx 3.74 + 1.92\,(M_{cs}-0.55)$
- Crossing timescale: $\log(t_\mathrm{cross}/\mathrm{yr}) \approx 5.24 - 6.95\,(M_{cs}-0.55)$
- Higher mass → faster, hotter, brighter → larger $Q(H^0)$

The Lyman continuum photon rate uses a blackbody approximation with
ionizing fraction $f_\mathrm{ion} \approx \exp(-0.92\,x^{0.55})$
where $x = h\nu_\mathrm{Ly}/kT_\star$.
"""

# ╔═╡ postaGB-fn-cell
"""
    postaGB_track(M_cs, t_bar) → (L_star_erg, T_star_K)

Post-AGB central star luminosity (erg/s) and effective temperature (K)
at dimensionless evolutionary time t̄, for a CS of mass M_cs (M☉).

T_star_K is no longer clamped at 1e4 K — the caller (Q_H0) handles the
low-temperature regime smoothly, so clamping here was masking the physical
fade and causing r_S to stay falsely large at late t̄ before snapping to zero.
"""
function postaGB_track(M_cs::Float64, t_bar::Float64)
    logL_peak_sun = 3.74 + 1.92 * (M_cs - 0.55)
    L_peak   = 10.0^(logL_peak_sun + logL_sun)
    T_peak_K = 1.4e5 * (M_cs / 0.6)^2.0
    t_pk     = 0.8

    L_star = if t_bar <= t_pk
        L_peak * exp(-4.0 * ((t_bar - t_pk) / t_pk)^2)
    else
        L_peak * (t_bar / t_pk)^(-1.8)
    end

    T_star_K = if t_bar <= t_pk
        T_peak_K * exp(-2.0 * ((t_bar - t_pk) / t_pk)^2)
    else
        T_peak_K * (t_bar / t_pk)^(-0.9)
    end
    # No floor clamp: let T_star_K fall naturally.
    # Q_H0 evaluates smoothly to near-zero as T★ cools below ~3×10⁴ K.
    T_star_K = max(T_star_K, 1.0e3)   # only prevent exactly zero

    return L_star, T_star_K
end

"""
    Q_H0(T_star_K, L_star_erg) → photons s⁻¹

Lyman continuum photon rate from a blackbody central star.

Replaces the hard threshold at 2×10⁴ K with a smooth exponential suppression.
The old `T <= 2e4 → return 0` created a cliff: r_S was finite one step before
the threshold and exactly zero one step after, producing a discontinuous drop
in L(t̄). The corrected form uses the same blackbody fraction fit but lets it
decay smoothly to zero as T★ → 0, with no hard cutoff.
"""
function Q_H0(T_star_K::Float64, L_star_erg::Float64)
    T_star_K <= 0.0 && return 0.0
    x = 13.6 / (kB_eV * T_star_K)    # hν_Ly / kT★
    # Blackbody ionizing fraction: smooth for all x > 0, → 0 as x → ∞
    # exp(-0.92 * x^0.55) already goes to 0 continuously — no hard cutoff needed
    x > 150.0 && return 0.0           # numerical underflow guard only
    f_ion  = exp(-0.92 * x^0.55)
    E_mean = 2.0 * 13.6 * eV2erg
    return f_ion * L_star_erg / E_mean
end

# ╔═╡ stromgren-cell
md"""
## 6 · Strömgren Truncation

The ionization-bounded radius $r_S$ is found by integrating the recombination
rate outward until the stellar ionizing budget is exhausted:

$$Q(H^0) = \int_0^{r_S} \alpha_B\,n_e^2(r)\,4\pi r^2\,dr \cdot r_\star^3$$

The emissivity is zeroed for $r > r_S$. Three regimes:
- *Ionization-bounded* (dense shell): $r_S$ inside shell; inner face only.
- *Density-bounded* (dilute shell): $r_S$ beyond shell; full shell visible.
- *Cooling star* (late t̄): Q drops smoothly, $r_S$ retreats continuously.

**Late-time behaviour.** When $Q(H^0) \to 0$ as the star cools, $r_S \to 0$
and $L \to$ NaN (no ionized volume). This is correct physics — the PN fades.
The onset is now smooth because `Q_H0` has no hard threshold and
`stromgren_radius` returns `r_S > 0` for any `Q_ion > 0`.
"""

# ╔═╡ stromgren-fn-cell
"""
    stromgren_radius(t, n0, σ, Q_ion) → r_S (dimensionless)

Locate r_S by bisection on the cumulative recombination integral.
Returns r_max (density-bounded) if Q_ion exceeds total recombinations.
Returns 0.0 only if Q_ion is exactly zero (caller should check).

The Q_ion ≈ 0 early-exit that previously returned 0.0 for any small Q
is removed — at late t̄ when Q is small but nonzero the star still
ionizes a thin inner skin, which is physical and should be computed.
The bisection naturally finds a very small r_S in this regime.
"""
function stromgren_radius(t::Float64, n0::Float64, σ::Float64, Q_ion::Float64)
    Q_ion <= 0.0 && return 0.0

    r_max = t + 6.0 * σ + 0.3

    rec_integrand(r) = r <= 0.0 ? 0.0 :
        α_B * n_profile(r, t, n0, σ)^2 * 4π * r^2 * r_star_cm^3

    # Total recombinations in whole domain — pass shell peak as breakpoint
    Q_total, _ = quadgk(rec_integrand, 0.0, t, r_max;
                         rtol=1e-4, atol=0.0, order=21)

    # Density-bounded: star ionizes more than the whole shell
    Q_total <= Q_ion && return r_max

    # Bisection on cumulative integral — monotone, single zero guaranteed
    lo, hi = 0.0, r_max
    for _ in 1:60
        mid    = 0.5 * (lo + hi)
        # Pass shell peak as breakpoint if it lies in [0, mid]
        bp     = (t < mid) ? (0.0, t, mid) : (0.0, mid)
        Q_mid, _ = quadgk(rec_integrand, bp...; rtol=1e-3, atol=0.0, order=15)
        Q_mid < Q_ion ? (lo = mid) : (hi = mid)
        hi - lo < 1e-7 * r_max && break
    end

    return 0.5 * (lo + hi)
end

# ╔═╡ luminosity-cell
md"""
## 7 · Shell Luminosity with Full Physics

$$L_{5007}(t) = \int_0^{r_S} \varepsilon(r,t)\,4\pi r^2\,dr\cdot r_\star^3$$

$$\varepsilon = \frac{n_e^2 \cdot Z \cdot f_{\mathrm{O}^{2+}} \cdot q(T) \cdot h\nu_{5007}}
                    {1 + n_e/n_\mathrm{crit}(T)}$$

**Oscillation suppression.** Three measures are taken to produce a smooth
$L(\bar{t})$ curve:

1. `r_S` is solved via binary search on a `quadgk`-based cumulative integral,
   not a fixed Riemann sum, so it varies continuously with $\bar{t}$.
2. The luminosity integral passes `r_S` and the shell peak $r = t$ as
   explicit breakpoints to `quadgk`, so the discontinuity at $r_S$ and the
   Gaussian spike at $r = t$ are treated as interval boundaries rather than
   interior features the integrator must discover adaptively.
3. Integration order is raised to 21 (from the default 7) to resolve the
   narrow Gaussian shell at high $n_0$.
"""

# ╔═╡ luminosity-fn-cell
"""
    shell_luminosity(t, n0, T_in, T_out, σ, Z, b, Q_ion) → Float64

[O III] λ5007 luminosity (log₁₀ erg/s). Oscillation fixes applied:
  - r_S solved smoothly via bisection on quadgk cumulative integral
  - Explicit breakpoints at r=t (shell peak) and r=r_S (jump discontinuity)
  - Integration order=21 to resolve narrow Gaussian shell
"""
function shell_luminosity(t::Float64, n0::Float64, T_in::Float64, T_out::Float64,
                          σ::Float64, Z::Float64, b::Float64, Q_ion::Float64)

    r_S = stromgren_radius(t, n0, σ, Q_ion)
    (r_S <= 0.0) && return NaN

    # Upper limit: min of r_S and the effective shell tail
    r_max = min(r_S, t + 6.0 * σ + 0.3)
    r_max <= 0.0 && return NaN

    # Breakpoints: origin, shell peak, Strömgren surface
    # quadgk integrates each sub-interval separately — no discontinuity aliasing
    r_peak = clamp(t, 0.0, r_max)   # shell Gaussian peaks at r = t
    breakpoints = sort(unique(filter(r -> 0.0 < r < r_max,
                                     [r_peak, r_S])))

    emissivity(r) = begin
        r <= 0.0 && return 0.0
        nₑ = n_profile(r, t, n0, σ)
        nₑ < 1e-6 && return 0.0
        T  = T_profile(r, t, T_in, T_out, b)
        T  <= 0.0 && return 0.0
        fracs = saha_ratios(nₑ, T)
        any(isnan, fracs) && return 0.0
        f_O2p = fracs[3]
        q     = q_eff(T)
        n_c   = ne_crit(T)
        ε = nₑ^2 * Z * f_O2p * q * hν5007 / (1.0 + nₑ / n_c)
        return ε * 4π * r^2
    end

    # Integrate over [0, r_max] with explicit breakpoints
    val, err = quadgk(emissivity, 0.0, breakpoints..., r_max;
                       rtol=1e-5, atol=0.0, order=21)

    L = val * r_star_cm^3
    return (L > 0.0 && isfinite(L)) ? log10(L) : NaN
end

# ╔═╡ nebparams-cell
md"## 8 · Nebula Parameter Struct"

# ╔═╡ nebparams-fn-cell
"""
    NebParams

Physical parameters of a single nebula and its central star.
σ and b are NOT stored — they are computed at each t̄ from shock dynamics.
"""
struct NebParams
    n0     :: Float64   # wind density at r̄ = 1  (cm⁻³); from Ṁ/4πr²vw
    T_in   :: Float64   # post-shock inner temperature (eV)
    T_out  :: Float64   # outer (recombined) temperature (eV)
    Z      :: Float64   # oxygen abundance (O/H by number)
    M_cs   :: Float64   # central star mass (M☉)
    # Shock parameters — govern σ(t)
    sigma0 :: Float64   # initial shell width (strong-shock limit, ≈ 0.02–0.05)
    t_thin :: Float64   # shock-weakening timescale (t̄ units, ≈ 1–3)
    beta   :: Float64   # shell-broadening exponent (≈ 0.3–0.5)
    # Ionization-front parameter — governs b(t)
    v_IF   :: Float64   # ionization-front speed (km/s); ~100 R-type, ~10 D-type
end

"""
    luminosity_at(t, nb) → Float64

log₁₀ L([O III] 5007) at time t̄. Computes σ(t) and b(t) from shock
physics, then threads Q(H⁰) from the post-AGB track into shell_luminosity.
"""
function luminosity_at(t::Float64, nb::NebParams)
    σ = sigma_shell(t; sigma0=nb.sigma0, t_thin=nb.t_thin, beta=nb.beta)
    b = b_transition(t, nb.n0;
                     sigma0=nb.sigma0, t_thin=nb.t_thin, beta=nb.beta,
                     v_IF_kms=nb.v_IF)
    L_star, T_star_K = postaGB_track(nb.M_cs, t)
    Q = Q_H0(T_star_K, L_star)
    return shell_luminosity(t, nb.n0, nb.T_in, nb.T_out, σ, nb.Z, b, Q)
end

# ╔═╡ sliders-cell
md"""
## 9 · Interactive Explorer

Sliders are now **physical shock parameters** — σ and b are derived from
them at each t̄ rather than being set by hand.
"""

# ╔═╡ sl-n0
@bind log_n0 Slider(10.0:0.1:16.0; default=14.0, show_value=true)

# ╔═╡ sl-ti
@bind T_inner Slider(0.5:0.01:1.5; default=0.82, show_value=true)

# ╔═╡ sl-to
@bind T_outer Slider(0.3:0.01:0.9; default=0.66, show_value=true)

# ╔═╡ sl-z
@bind log_Z Slider(-3.0:0.05:-1.0; default=-1.7, show_value=true)

# ╔═╡ sl-mcs
@bind M_cs Slider(0.52:0.01:0.70; default=0.60, show_value=true)

# ╔═╡ sl-sigma0
@bind sigma0 Slider(0.01:0.005:0.10; default=0.03, show_value=true)

# ╔═╡ sl-tthin
@bind t_thin Slider(0.5:0.1:4.0; default=1.5, show_value=true)

# ╔═╡ sl-beta
@bind beta_exp Slider(0.1:0.05:0.8; default=0.4, show_value=true)

# ╔═╡ sl-vif
@bind v_IF Slider(5.0:5.0:150.0; default=20.0, show_value=true)

# ╔═╡ slider-labels
md"""
| Parameter | Value | Physical meaning |
|-----------|-------|-----------------|
| log₁₀(n₀ / cm⁻³) | $(log_n0) | Wind density at r̄ = 1 from Ṁ/4πr²vw |
| T\_inner (eV) | $(T_inner) | Post-shock ionized gas temperature |
| T\_outer (eV) | $(T_outer) | Outer recombined gas temperature |
| log₁₀(Z / Z☉) | $(log_Z) | Oxygen abundance |
| M\_cs (M☉) | $(M_cs) | Central star mass → Q(H⁰) and track speed |
| σ₀ | $(sigma0) | Initial shell width (strong-shock limit, Δr/r ~ 1/4) |
| t\_thin | $(t_thin) | Shock-weakening timescale (t̄ units) |
| β | $(beta_exp) | Shell-broadening exponent (Villaver+2002) |
| v\_IF (km/s) | $(v_IF) | Ionization-front speed (~100 R-type, ~10 D-type) |
"""

# ╔═╡ compute-cell
begin
    nb = NebParams(
        10.0^log_n0,
        T_inner,
        T_outer,
        10.0^log_Z,
        M_cs,
        sigma0,
        t_thin,
        beta_exp,
        v_IF
    )

    t_vec = collect(0.05:0.08:7.0)

    # σ(t) and b(t) across the evolution
    σ_vec = [sigma_shell(t; sigma0=nb.sigma0, t_thin=nb.t_thin, beta=nb.beta) for t in t_vec]
    b_vec = [b_transition(t, nb.n0;
                          sigma0=nb.sigma0, t_thin=nb.t_thin, beta=nb.beta,
                          v_IF_kms=nb.v_IF) for t in t_vec]

    # Luminosity curve
    L_curve = [luminosity_at(t, nb) for t in t_vec]

    # Central star track
    cs_track    = [postaGB_track(nb.M_cs, t) for t in t_vec]
    L_star_vec  = [cs[1] for cs in cs_track]
    T_star_vec  = [cs[2] for cs in cs_track]
    Q_vec       = Q_H0.(T_star_vec, L_star_vec)

    # Strömgren radii (using σ(t) at each step)
    r_S_vec        = [stromgren_radius(t, nb.n0, σ_vec[i], Q_vec[i])
                      for (i, t) in enumerate(t_vec)]
    r_shell_inner  = max.(t_vec .- σ_vec, 0.0)
    r_shell_outer  = t_vec .+ σ_vec

    # Peak luminosity
    valid_mask = .!isnan.(L_curve)
    logL_peak  = NaN; t_peak = NaN; f_O2p_peak = NaN

    if any(valid_mask)
        idx       = argmax(L_curve[valid_mask])
        valid_idx = findall(valid_mask)
        peak_abs  = valid_idx[idx]
        t_peak    = t_vec[peak_abs]
        logL_peak = L_curve[peak_abs]
        σ_peak    = σ_vec[peak_abs]
        b_peak    = b_vec[peak_abs]
        nₑ_pk     = n_profile(t_peak, t_peak, nb.n0, σ_peak)
        T_pk      = T_profile(t_peak, t_peak, nb.T_in, nb.T_out, b_peak)
        fr_pk     = nₑ_pk > 0 ? saha_ratios(nₑ_pk, T_pk) : fill(NaN, 6)
        f_O2p_peak = fr_pk[3]
    end

    ΔM_star = isnan(logL_peak) ? NaN : -(logL_peak - logL_PNLF_star) / 2.5

    nothing
end

# ╔═╡ summary-cell
md"""
### Peak luminosity summary

| Quantity | Value |
|----------|-------|
| Peak log₁₀(L₅₀₀₇ / erg s⁻¹) | $(round(logL_peak; digits=2)) |
| Peak time t̄ | $(round(t_peak; digits=2)) |
| f(O²⁺) at peak | $(round(f_O2p_peak * 100; digits=1)) % |
| Offset from PNLF M★ | ΔM = $(round(ΔM_star; digits=2)) mag |
| σ(t̄\_peak) — shock-derived | $(round(isnan(t_peak) ? NaN : sigma_shell(t_peak; sigma0=nb.sigma0, t_thin=nb.t_thin, beta=nb.beta); digits=4)) |
| b(t̄\_peak) — IF-derived | $(round(isnan(t_peak) ? NaN : b_transition(t_peak, nb.n0; sigma0=nb.sigma0, t_thin=nb.t_thin, beta=nb.beta, v_IF_kms=nb.v_IF); digits=2)) |
| n\_crit at T\_inner | $(round(ne_crit(T_inner); sigdigits=3)) cm⁻³ |
| log Q(H⁰) at peak | $(round(isnan(t_peak) ? NaN : log10(max(Q_vec[argmin(abs.(t_vec .- t_peak))], 1.0)); digits=2)) photons/s |
"""

# ╔═╡ plot-lum-cell
begin
    p1 = plot(;
        xlabel     = "Evolutionary time t̄",
        ylabel     = "log₁₀(L / erg s⁻¹)",
        title      = "[O III] λ5007 Luminosity Evolution",
        framestyle = :box, grid = true, gridalpha = 0.3,
        legend     = :topright)

    plot!(p1, t_vec[valid_mask], L_curve[valid_mask];
        label = "log L([O III] 5007)", lw = 2.5, color = :steelblue)

    if !isnan(logL_peak)
        scatter!(p1, [t_peak], [logL_peak];
            label = "Peak (t̄ = $(round(t_peak; digits=2)))",
            color = :firebrick, markersize = 8, markershape = :star5)

        hline!(p1, [logL_PNLF_star];
            label = "PNLF M★ cutoff", ls = :dash,
            color = :darkorange, lw = 1.5, alpha = 0.8)
    end

    p1
end

# ╔═╡ plot-stromgren-cell
begin
    # Strömgren radius vs shell extent over time
    p2 = plot(;
        xlabel     = "Evolutionary time t̄",
        ylabel     = "Radius (dimensionless)",
        title      = "Strömgren Radius vs Shell Extent",
        framestyle = :box, grid = true, gridalpha = 0.3,
        legend     = :topright)

    plot!(p2, t_vec, r_S_vec;
        label = "r_S (Strömgren)", lw = 2, color = :firebrick)
    plot!(p2, t_vec, r_shell_outer;
        label = "Shell outer (t̄ + σ)", lw = 1.5, ls = :dash, color = :steelblue)
    plot!(p2, t_vec, r_shell_inner;
        label = "Shell inner (t̄ − σ)", lw = 1.5, ls = :dot, color = :steelblue)

    # Shade region where rS < shell outer (ionization-bounded)
    ion_bounded = r_S_vec .< r_shell_outer
    if any(ion_bounded)
        t_ib  = t_vec[ion_bounded]
        rS_ib = r_S_vec[ion_bounded]
        ro_ib = r_shell_outer[ion_bounded]
        plot!(p2, t_ib, rS_ib;
            fillrange = ro_ib, fillalpha = 0.12,
            color = :firebrick, label = "Ionization-bounded zone", lw = 0)
    end

    p2
end

# ╔═╡ plot-cstar-cell
begin
    # Central star evolution
    logL_star_sun = log10.(max.(L_star_vec, 1.0)) .- logL_sun
    logT_star     = log10.(T_star_vec)

    p3a = plot(t_vec, logL_star_sun;
        xlabel = "t̄", ylabel = "log L★/L☉",
        title  = "CS Luminosity", lw = 2, color = :darkorchid,
        framestyle = :box, grid = true, gridalpha = 0.3, legend = false)

    p3b = plot(t_vec, logT_star;
        xlabel = "t̄", ylabel = "log T★ (K)",
        title  = "CS Temperature", lw = 2, color = :firebrick,
        framestyle = :box, grid = true, gridalpha = 0.3, legend = false)

    logQ = log10.(max.(Q_vec, 1.0))
    p3c = plot(t_vec, logQ;
        xlabel = "t̄", ylabel = "log Q(H⁰) (photons/s)",
        title  = "Ionizing Flux", lw = 2, color = :steelblue,
        framestyle = :box, grid = true, gridalpha = 0.3, legend = false)

    plot(p3a, p3b, p3c; layout = (1, 3), size = (900, 280),
         margin = 5Plots.mm, titlefont = font(10))
end

# ╔═╡ plot-shock-cell
begin
    # σ(t) and b(t): the two formerly-free parameters now derived from physics
    p_sig = plot(t_vec, σ_vec;
        xlabel = "Evolutionary time t̄", ylabel = "σ(t)",
        title  = "Shell Width (shock-derived)",
        lw = 2, color = :steelblue, framestyle = :box,
        grid = true, gridalpha = 0.3, legend = false)
    if !isnan(t_peak)
        vline!(p_sig, [t_peak]; ls = :dash, color = :firebrick, lw = 1, label = "Peak t̄")
    end

    p_b = plot(t_vec, b_vec;
        xlabel = "Evolutionary time t̄", ylabel = "b(t)",
        title  = "IF Sharpness (density-derived)",
        lw = 2, color = :darkorchid, framestyle = :box,
        grid = true, gridalpha = 0.3, legend = false)
    if !isnan(t_peak)
        vline!(p_b, [t_peak]; ls = :dash, color = :firebrick, lw = 1, label = "Peak t̄")
    end

    # Shell peak density over time (to show the quenching regime)
    ne_peak_vec = [n_profile(t, t, nb.n0, σ_vec[i]) for (i, t) in enumerate(t_vec)]
    p_ne = plot(t_vec, log10.(max.(ne_peak_vec, 1.0));
        xlabel = "Evolutionary time t̄", ylabel = "log₁₀ nₑ_peak (cm⁻³)",
        title  = "Shell Peak Density",
        lw = 2, color = :seagreen, framestyle = :box,
        grid = true, gridalpha = 0.3, legend = :topright)
    hline!(p_ne, [log10(ne_crit(T_inner))];
        label = "n_crit(T_in)", ls = :dash, color = :darkorange, lw = 1.5)
    if !isnan(t_peak)
        vline!(p_ne, [t_peak]; ls = :dash, color = :firebrick, lw = 1, label = "Peak t̄")
    end

    plot(p_sig, p_b, p_ne; layout = (1, 3), size = (900, 280),
         margin = 5Plots.mm, titlefont = font(10))
end

# ╔═╡ plot-radial-cell
begin
    # Use shock-derived σ and b at the peak time
    σ_pk  = isnan(t_peak) ? sigma0 :
            sigma_shell(t_peak; sigma0=nb.sigma0, t_thin=nb.t_thin, beta=nb.beta)
    b_pk  = isnan(t_peak) ? 4.0 :
            b_transition(t_peak, nb.n0;
                         sigma0=nb.sigma0, t_thin=nb.t_thin, beta=nb.beta,
                         v_IF_kms=nb.v_IF)

    r_vec = collect(0.001:0.008:(isnan(t_peak) ? 5.0 : t_peak + 8σ_pk))

    L_star_pk, T_star_pk = postaGB_track(nb.M_cs, isnan(t_peak) ? 1.0 : t_peak)
    Q_pk    = Q_H0(T_star_pk, L_star_pk)
    r_S_pk  = stromgren_radius(isnan(t_peak) ? 1.0 : t_peak, nb.n0, σ_pk, Q_pk)

    nₑ_r = [n_profile(r, t_peak, nb.n0, σ_pk) for r in r_vec]
    T_r  = [T_profile(r, t_peak, nb.T_in, nb.T_out, b_pk) for r in r_vec]
    f_r  = [begin
                nv = nₑ_r[k]; Tv = T_r[k]
                nv > 1.0 ? saha_ratios(nv, Tv)[3] : 0.0
            end for k in eachindex(r_vec)]
    q_r  = q_eff.(T_r)
    nc_r = ne_crit.(T_r)
    ε_r  = [nₑ_r[k]^2 * nb.Z * f_r[k] * q_r[k] * hν5007
              / (1.0 + nₑ_r[k] / nc_r[k])
            for k in eachindex(r_vec)]

    ε_phys = [r_vec[k] <= r_S_pk ? ε_r[k] : 0.0 for k in eachindex(r_vec)]
    εmax   = max(maximum(ε_phys), 1e-60)

    p4a = plot(r_vec, log10.(max.(nₑ_r, 1e-10));
        label = "log nₑ", color = :steelblue, lw = 2,
        ylabel = "log₁₀(nₑ / cm⁻³)", xlabel = "r̄",
        title  = "Radial profiles at t̄=$(round(t_peak;digits=2)), σ=$(round(σ_pk;digits=3)), b=$(round(b_pk;digits=1))",
        framestyle = :box, grid = true, gridalpha = 0.3)
    vline!(p4a, [r_S_pk]; ls = :dash, color = :firebrick, lw = 1.5, label = "r_S")
    hline!(p4a, [log10(ne_crit(T_inner))];
        ls = :dot, color = :darkorange, lw = 1, label = "n_crit")

    p4b = plot(r_vec, T_r;
        label = "T (eV)", color = :firebrick, lw = 2,
        ylabel = "T (eV)", xlabel = "r̄",
        framestyle = :box, grid = true, gridalpha = 0.3)
    vline!(p4b, [r_S_pk]; ls = :dash, color = :firebrick, lw = 1.5, label = "r_S")

    p4c = plot(r_vec, f_r;
        label = "f(O²⁺)", color = :seagreen, lw = 2,
        ylabel = "f(O²⁺)", xlabel = "r̄", ylims = (0, 1),
        framestyle = :box, grid = true, gridalpha = 0.3)
    vline!(p4c, [r_S_pk]; ls = :dash, color = :firebrick, lw = 1.5, label = "r_S")

    p4d = plot(r_vec, ε_phys ./ εmax;
        label = "ε (Strömgren-truncated)", color = :darkorchid, lw = 2,
        ylabel = "ε / ε_max", xlabel = "r̄",
        framestyle = :box, grid = true, gridalpha = 0.3)
    plot!(p4d, r_vec, ε_r ./ εmax;
        label = "ε (untruncated)", color = :gray60, lw = 1, ls = :dot)
    vline!(p4d, [r_S_pk]; ls = :dash, color = :firebrick, lw = 1.5, label = "r_S")

    plot(p4a, p4b, p4c, p4d;
        layout = (2, 2), size = (720, 520),
        margin = 5Plots.mm, legend = :topright, titlefont = font(9))
end

# ╔═╡ plot-ionfrac-cell
begin
    ion_labels = ["O⁰", "O⁺", "O²⁺", "O³⁺", "O⁴⁺"]
    ion_colors = [:gray60, :steelblue, :firebrick, :seagreen, :darkorchid]
    ion_styles = [:dash, :dot, :solid, :dashdot, :dashdotdot]

    frac_mat = zeros(Float64, length(t_vec), 5)
    for (k, t) in enumerate(t_vec)
        σ_t = σ_vec[k]
        b_t = b_vec[k]
        nₑ  = n_profile(t, t, nb.n0, σ_t)
        T   = T_profile(t, t, nb.T_in, nb.T_out, b_t)
        if nₑ > 1.0 && T > 0.0
            fr = saha_ratios(nₑ, T)
            frac_mat[k, :] = fr[1:5]
        end
    end

    p5 = plot(;
        xlabel = "Evolutionary time t̄", ylabel = "Ionization fraction",
        title  = "O Ionization Fractions at Shell Peak (shock-derived σ, b)",
        framestyle = :box, grid = true, gridalpha = 0.3,
        ylims = (0, 1), legend = :right)

    for i in 1:5
        plot!(p5, t_vec, frac_mat[:, i];
            label = ion_labels[i], lw = i == 3 ? 2.5 : 1.5,
            color = ion_colors[i], ls = ion_styles[i])
    end
    if !isnan(t_peak)
        vline!(p5, [t_peak]; label = "Peak t̄", ls = :dash, color = :black, lw = 1, alpha = 0.5)
    end

    p5
end

# ╔═╡ manifold-cell
md"""
## 10 · The Critical Manifold $\mathcal{M}(n_0, T_\mathrm{in}, t) = 0$

We seek the **zero-set** of

$$\mathcal{M}(n_0, T_\mathrm{in}, t) \;\equiv\; \log_{10} L(n_0, T_\mathrm{in}, t) - \log_{10} L^\star = 0$$

where $L^\star$ is the PNLF empirical cutoff luminosity. This defines a
**2-surface** in the 3D parameter space $(n_0, T_\mathrm{in}, t)$: the locus
of all nebulae that can exactly reach M★ at some moment in their evolution.

**Physical interpretation of the three directions:**

- *$t$ direction*: at fixed $(n_0, T_\mathrm{in})$, the zero crossings in $t$
  mark the entry and exit times of the bright window. Their separation
  $\Delta t$ is the duration a nebula spends at $L \geq L^\star$.
- *$n_0$ direction*: higher wind density raises emissivity but also
  deepens quenching. The manifold curves back above $n_\mathrm{crit}$,
  creating a **closed** bright region — no arbitrarily dense nebula can
  exceed M★ indefinitely.
- *$T_\mathrm{in}$ direction*: hotter inner gas shifts the Saha balance
  toward O³⁺, reducing $f_{\mathrm{O}^{2+}}$. There is an optimal
  $T_\mathrm{in}$ where O²⁺ is maximised; the manifold closes on both
  sides in temperature.

The manifold is solved by:
1. Building a grid over $(n_0, T_\mathrm{in})$.
2. At each grid point evaluating $L(t)$ on a coarse $t$-grid to bracket sign changes.
3. Refining each bracket with `Roots.find_zero` (Brent's method).
4. Collecting all solution $t$-values into the surface.

This gives up to *two* solutions per $(n_0, T_\mathrm{in})$ point (entry and
exit), which together bound the bright window and trace the manifold's
two sheets.
"""

# ╔═╡ manifold-fn-cell
"""
    peak_window(n0, T_in, nb_template; t_grid, logL_target)
        → (t_entry, t_exit, t_peak_local, logL_max)

For a nebula with wind density n0 and inner temperature T_in (all other
shock parameters taken from nb_template), find:
  - t_entry : first time L crosses L★ from below  (NaN if never reached)
  - t_exit  : last  time L crosses L★ from above  (NaN if never exits)
  - t_peak_local : time of maximum L
  - logL_max     : maximum log L achieved

Uses Brent's method via Roots.find_zero on each sign-change bracket.
"""
function peak_window(n0::Float64, T_in::Float64, nb_t::NebParams;
                     t_grid::AbstractVector{Float64} = collect(0.1:0.15:7.0),
                     logL_target::Float64 = logL_PNLF_star)

    # Build a NebParams with this (n0, T_in), inheriting shock params
    nb_local = NebParams(n0, T_in, nb_t.T_out, nb_t.Z, nb_t.M_cs,
                         nb_t.sigma0, nb_t.t_thin, nb_t.beta, nb_t.v_IF)

    # Evaluate L on coarse grid; f(t) = logL(t) - logL_target
    f_vals = map(t_grid) do t
        logL = luminosity_at(t, nb_local)
        isnan(logL) ? -Inf : logL - logL_target
    end

    # Peak luminosity on this grid
    logL_vec    = f_vals .+ logL_target
    valid       = isfinite.(logL_vec)
    logL_max    = any(valid) ? maximum(logL_vec[valid]) : NaN
    # argmax over valid entries only — avoids indexing empty vector
    t_peak_loc  = if any(valid)
        idx = argmax(map((l, v) -> v ? l : -Inf, logL_vec, valid))
        t_grid[idx]
    else
        NaN
    end

    # Find all sign-change brackets
    brackets = Tuple{Float64,Float64}[]
    for i in 1:length(t_grid)-1
        if isfinite(f_vals[i]) && isfinite(f_vals[i+1]) &&
           f_vals[i] * f_vals[i+1] < 0.0
            push!(brackets, (t_grid[i], t_grid[i+1]))
        end
    end

    t_entry = NaN
    t_exit  = NaN

    if length(brackets) >= 1
        # Refine first bracket (entry into bright window)
        try
            t_entry = find_zero(
                t -> begin
                    logL = luminosity_at(t, nb_local)
                    isnan(logL) ? -logL_target : logL - logL_target
                end,
                brackets[1], Brent())
        catch
        end
    end

    if length(brackets) >= 2
        # Refine last bracket (exit from bright window)
        try
            t_exit = find_zero(
                t -> begin
                    logL = luminosity_at(t, nb_local)
                    isnan(logL) ? -logL_target : logL - logL_target
                end,
                brackets[end], Brent())
        catch
        end
    end

    return (t_entry=t_entry, t_exit=t_exit,
            t_peak_local=t_peak_loc, logL_max=logL_max)
end

"""
    solve_manifold(nb_template; n0_grid, Tin_grid, logL_target)
        → DataFrame

Solve M(n0, T_in, t) = 0 over a grid of (n0, T_in) values.
Returns a DataFrame with columns:
  log_n0, T_in, t_entry, t_exit, Δt (bright window duration),
  t_peak_local, logL_max, reaches_cutoff.
"""
function solve_manifold(nb_t::NebParams;
                        log_n0_grid::AbstractVector{Float64} = range(10.0, 16.0; length=25),
                        T_in_grid::AbstractVector{Float64}   = range(0.55, 1.30; length=25),
                        logL_target::Float64 = logL_PNLF_star,
                        t_grid::AbstractVector{Float64} = collect(0.1:0.12:7.0))
    rows = []
    for ln0 in log_n0_grid, Tin in T_in_grid
        n0 = 10.0^ln0
        # Guard: T_out must be < T_in
        T_out = min(nb_t.T_out, Tin - 0.05)
        T_out <= 0.3 && (T_out = 0.3)
        pw = peak_window(n0, Tin,
                         NebParams(n0, Tin, T_out, nb_t.Z, nb_t.M_cs,
                                   nb_t.sigma0, nb_t.t_thin, nb_t.beta, nb_t.v_IF);
                         t_grid, logL_target)
        Δt = (!isnan(pw.t_entry) && !isnan(pw.t_exit)) ?
             pw.t_exit - pw.t_entry : NaN
        push!(rows, (
            log_n0         = ln0,
            T_in           = Tin,
            t_entry        = pw.t_entry,
            t_exit         = pw.t_exit,
            Δt             = Δt,
            t_peak_local   = pw.t_peak_local,
            logL_max       = pw.logL_max,
            reaches_cutoff = !isnan(pw.t_entry)
        ))
    end
    return DataFrame(rows)
end

# ╔═╡ manifold-sliders-cell
md"""
### Manifold grid resolution and target

Coarser grids compute faster; refine once you have found the region of interest.
"""

# ╔═╡ sl-manifold-n-grid
@bind manifold_n_grid Slider(10:5:40; default=20, show_value=true)

# ╔═╡ sl-manifold-t-grid
@bind manifold_t_grid Slider(10:5:40; default=20, show_value=true)

# ╔═╡ sl-manifold-delta
@bind manifold_delta Slider(-2.0:0.25:0.0; default=0.0, show_value=true)

# ╔═╡ manifold-compute-cell
begin
    manifold_logL_target = logL_PNLF_star + manifold_delta

    mf_df = solve_manifold(nb;
        log_n0_grid  = range(10.0, 16.0; length=manifold_n_grid),
        T_in_grid    = range(0.55, 1.30; length=manifold_n_grid),
        logL_target  = manifold_logL_target,
        t_grid       = collect(range(0.1, 7.0; length=manifold_t_grid * 3)))

    n_reach    = count(mf_df.reaches_cutoff)
    frac_reach = round(100 * n_reach / nrow(mf_df); digits=1)

    finite_Δt   = filter(isfinite, mf_df.Δt)
    finite_logL = filter(isfinite, mf_df.logL_max)
    mean_Δt_str = isempty(finite_Δt)   ? "—" : string(round(mean(finite_Δt);   digits=2))
    max_logL_str = isempty(finite_logL) ? "—" : string(round(maximum(finite_logL); digits=2))

    md"""
    **Manifold statistics** (target: log L = $(round(manifold_logL_target; digits=2)))

    | | |
    |---|---|
    | Grid points | $(nrow(mf_df)) |
    | Points reaching cutoff | $n_reach ($frac_reach %) |
    | Mean bright-window Δt̄ | $mean_Δt_str |
    | Max log L on grid | $max_logL_str |

    *If 0% reach the cutoff, lower the target with the Δ slider or raise n₀ / M\_cs.*
    """
end

# ╔═╡ plot-manifold-cell
begin
    reach   = filter(:reaches_cutoff => identity, mf_df)
    noreach = filter(:reaches_cutoff => !, mf_df)

    # ── Panel 1: which (n0, T_in) reach the cutoff ───────────────────────────
    p6a = plot(;
        xlabel = "T_in (eV)", ylabel = "log₁₀ n₀ (cm⁻³)",
        title  = "Which nebulae reach the cutoff?",
        framestyle = :box, grid = true, gridalpha = 0.3)

    if !isempty(noreach)
        scatter!(p6a, noreach.T_in, noreach.log_n0;
            label = "Never reaches L★", color = :gray80,
            markersize = 3, markerstrokewidth = 0, alpha = 0.5)
    end
    if !isempty(reach)
        scatter!(p6a, reach.T_in, reach.log_n0;
            label          = "Reaches L★",
            marker_z       = reach.t_peak_local,
            color          = :viridis,
            markersize     = 5,
            markerstrokewidth = 0,
            colorbar_title = "t̄_peak")
    else
        annotate!(p6a, [(mean([0.55, 1.30]), mean([10.0, 16.0]),
            text("No solutions found.\nLower Δ or adjust parameters.", 9, :center, :gray40))])
    end
    hline!(p6a, [log10(ne_crit(T_inner))];
        label = "n_crit(T_in)", ls = :dash, color = :firebrick, lw = 1)

    # ── Panel 2: bright-window duration Δt̄ ──────────────────────────────────
    reach_Δt = filter(r -> isfinite(r.Δt), mf_df)

    p6b = plot(;
        xlabel = "T_in (eV)", ylabel = "log₁₀ n₀ (cm⁻³)",
        title  = "Bright-window duration Δt̄",
        framestyle = :box, grid = true, gridalpha = 0.3)

    if !isempty(reach_Δt)
        scatter!(p6b, reach_Δt.T_in, reach_Δt.log_n0;
            label          = false,
            marker_z       = reach_Δt.Δt,
            color          = :plasma,
            markersize     = 5,
            markerstrokewidth = 0,
            colorbar_title = "Δt̄")
    else
        annotate!(p6b, [(mean([0.55, 1.30]), mean([10.0, 16.0]),
            text("No finite Δt̄ values.", 9, :center, :gray40))])
    end

    # ── Panel 3: manifold slice at median log n₀ ─────────────────────────────
    p6c = plot(;
        xlabel = "T_in (eV)", ylabel = "t̄",
        framestyle = :box, grid = true, gridalpha = 0.3, legend = :topright)

    if !isempty(reach)
        med_ln0    = median(reach.log_n0)
        tol        = max(0.4, step(range(10.0, 16.0; length=manifold_n_grid)))
        slice_rows = sort(filter(r -> abs(r.log_n0 - med_ln0) < tol, reach), :T_in)

        if !isempty(slice_rows)
            plot!(p6c, slice_rows.T_in, slice_rows.t_entry;
                label = "t_entry", lw = 2, color = :steelblue)
            # t_exit may be NaN for points that never exit — filter before plotting
            exit_rows = filter(r -> isfinite(r.t_exit), slice_rows)
            if !isempty(exit_rows)
                plot!(p6c, exit_rows.T_in, exit_rows.t_exit;
                    label = "t_exit", lw = 2, color = :firebrick, ls = :dash)
            end
            plot!(p6c, slice_rows.T_in, slice_rows.t_peak_local;
                label = "t_peak_local", lw = 1.5, color = :seagreen, ls = :dot)
            if !isnan(t_peak)
                hline!(p6c, [t_peak];
                    label = "Global peak t̄", ls = :dashdot,
                    color = :black, lw = 1, alpha = 0.6)
            end
            title!(p6c, "Manifold slice at log n₀ ≈ $(round(med_ln0; digits=1))")
        else
            title!(p6c, "Slice empty — widen tolerance")
        end
    else
        title!(p6c, "No solutions — adjust parameters")
        annotate!(p6c, [(0.9, 3.5, text("No solutions found.", 9, :center, :gray40))])
    end

    # ── Panel 4: peak logL surface ────────────────────────────────────────────
    p6d = plot(;
        xlabel = "T_in (eV)", ylabel = "log₁₀ n₀ (cm⁻³)",
        title  = "Peak luminosity surface",
        framestyle = :box, grid = true, gridalpha = 0.3)

    finite_logL_mask = isfinite.(mf_df.logL_max)
    if any(finite_logL_mask)
        logL_fill = copy(mf_df.logL_max)
        fill_val  = minimum(mf_df.logL_max[finite_logL_mask])
        logL_fill[.!finite_logL_mask] .= fill_val
        scatter!(p6d, mf_df.T_in, mf_df.log_n0;
            label          = false,
            marker_z       = logL_fill,
            color          = :inferno,
            markersize     = 5,
            markerstrokewidth = 0,
            colorbar_title = "max log L")
        hline!(p6d, [log10(ne_crit(T_inner))];
            label = "n_crit", ls = :dash, color = :white, lw = 1, alpha = 0.6)
    else
        annotate!(p6d, [(mean([0.55, 1.30]), mean([10.0, 16.0]),
            text("All logL values are NaN.\nCheck model parameters.", 9, :center, :gray40))])
    end

    plot(p6a, p6b, p6c, p6d;
        layout = (2, 2), size = (820, 680),
        margin = 6Plots.mm, titlefont = font(10))
end

# ╔═╡ diagnostics-cell
md"""
## 11 · Numerical Diagnostics
"""

# ╔═╡ diag-table-cell
begin
    # Rows span from classical PN densities up to the wind-density regime.
    # At n₀ = 10^14 the quenching factor is ~n_crit/n₀ ~ 10^-8:
    # the shell core is essentially dark and emission comes from the flanks.
    test_cases = [
        (1e4,  0.82),    # classical low-density PN
        (1e6,  0.82),    # at n_crit: quench ~ 0.5
        (1e7,  0.82),    # above n_crit: quench ~ 0.07
        (1e10, 0.82),    # dense inner wind
        (1e12, 0.82),    # AGB envelope regime
        (1e14, 0.82),    # wind-anchored n₀: quench ~ 7×10⁻⁹
        (1e14, 1.2),     # same density, hotter
    ]

    rows = []
    for (ne, T) in test_cases
        fr  = saha_ratios(ne, T)
        n_c = ne_crit(T)
        q   = q_eff(T)
        quench = 1.0 / (1.0 + ne / n_c)
        push!(rows, (
            nₑ       = ne,
            T_eV     = T,
            f_O2p    = round(fr[3]; digits=4),
            n_crit   = round(n_c; sigdigits=3),
            quench   = round(quench; digits=3),
            q_eff    = round(q; sigdigits=3),
            sum_fracs = round(sum(fr); digits=6)
        ))
    end

    DataFrame(rows)
end

# ╔═╡ 00000000-0000-0000-0000-000000000002
# Pluto package manager manifest — do not edit
