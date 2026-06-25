### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ fde54e19-d6fe-4134-84fb-c39be3c0eea2
begin
    import Pkg
    # Disables Pluto's isolated environment and uses your terminal project instead
    Pkg.activate(".") 

    using Plots
	using LinearAlgebra
	using DataFrames
	using Statistics
	using QuadGK
	using Interpolations
	using CSV
	using PlutoUI

	# python dependencies
    using PythonCall
    np = pyimport("numpy")
	fac 	= pyimport("pfac.fac")
	ftab  	= pyimport("pfac.table")
	rfac    = pyimport("pfac.rfac")
	sys 	= pyimport("sys")

	fac_dir  	= joinpath(pwd(), "AMDS_data")
end;

# ╔═╡ 06a00479-13de-4fc2-83c7-ba263895ad86
begin
	#debugging statement
	const verbose = true
	
	# block defining physical constants
	const a0 	= 5.291e-9  # bohr radius in cm
	const h 	= 4.135e-15 # eV ⋅ s
	const c 	= 2.99e10  	# cm / s
	const kb 	= 8.62e-5   # eV / Kelvin 
	const α 	= 1 / 137.036 	# fine structure constant
	const emass = 511e3     # eV/c^2

	# conversion factors
	const eV_to_Hartree = 1 / 27.211 # atomic units conversion factor
	const Hartree_to_eV = 27.211     # eV conversion factor
	const velscaling 	= 5.931e7    # 
end;

# ╔═╡ f1a53e3c-fd60-4c8c-8931-66a92e28dc36
begin
	# checks to see if the case B cross sections need to be calculated
	atomicnumber = 8
		
	global minenergy 	= 0.0
	global maxenergy 	= 1000.0
	global slice 		= 1

	#= default values for precomputed table
	minenergy 	= 0.0   eV
	maxenergy 	= 100.0 eV
	slice 		= 0.01
	=#

	# these masks need to be dynamically coded instead of hardcoded eventually
	# they were found by manually inspecting the .en files to find split ground states
	groundstatemasks = [
		[0],
		[0],
		[0],
		[0],
		[0, 1],
		[0, 1, 2],
		[0],
		[0, 1, 2]]
	# notice the excited state mask matches the pattern of the ground state mask
	excitedstatemasks = [
		[4],
		[7],
		[8],
		[46],
		[125],
		[236, 237],
		[272, 273, 274],
		[266]]
end

# ╔═╡ b30d80a6-fb36-4baf-bb75-a17e319566c3
md"""
### ----- functions ----- ###
"""

# ╔═╡ fbb8819e-da92-4cd7-a3f0-811a27aa3ae4
begin
	struct Prr
		p1::Float64
		p2::Float64
		p3::Float64
		p4::Float64
	end
end

# ╔═╡ 1256163d-1be6-4742-a080-baf5136caf83
begin
	function rrparamblockunpacker(pmatrix)
		m, n = size(pmatrix)
		pvec = fill(Prr(-1,-1,-1,-1), m)
		
		for row in 1:m
			p1, p2, p3, p4 = pmatrix[row, :]			
			params = Prr(p1, p2, p3, p4)
			pvec[row] = params
		end
		
		return pvec
	end
end

# ╔═╡ 729ba9a7-bcc2-4386-9348-7cde5d21de27
begin
	function enblockreader(block)
		ilev = pyconvert(Vector{Int64}, block["ILEV"])
		vnl  = pyconvert(Vector{Int64}, block["VNL"])

		return ilev, vnl
	end
end

# ╔═╡ 1c6a1132-bd43-4d5b-a03c-08cf21fb24db
begin
	function rrblockreader(block)
		energy 	= pyconvert(Vector{Float64}, block["Delta E"])
		params 	= pyconvert(Array{Float64}, block["parameters"])
		
		# bound index and angular momentum
		bindex 	= pyconvert(Vector{Float64}, block["bound_index"])
		b2j 	= pyconvert(Vector{Float64}, block["bound_2J"])

		# free index and angular momentum
		findex 	= pyconvert(Vector{Float64}, block["free_index"])
		f2j 	= pyconvert(Vector{Float64}, block["free_2J"])
		
		deltal 	= pyconvert(Vector{Float64}, block["Delta L"])
		return energy, params, bindex, b2j, findex, f2j, deltal
	end
end

# ╔═╡ 9194879b-4d99-4416-bbf4-f1ac4820d40e
begin
	test_block = 1
	test_index = 1
	test_nelectrons = 6
	
	test_ENfile = joinpath(fac_dir, "O0$(test_nelectrons)a.en")
	test_RRfile = joinpath(fac_dir, "O0$(test_nelectrons)a.rr")

	test_ENcontents = rfac.read_en(test_ENfile)
	test_RRcontents = rfac.read_rr(test_RRfile)

	test_ENheader, test_ENdata = test_ENcontents
	test_RRheader, test_RRdata = test_RRcontents

	test_RRblock = test_RRdata[test_block]

	nblocks = pyconvert(Int64, test_RRheader["NBlocks"])
	test_egrid = pyconvert(Vector{Float64}, test_RRblock["EGRID"])
	test_RRxsec   = pyconvert(Array{Float64}, test_RRblock["RR crosssection"])

	qilev = Int64[]
	qvnls = Int64[]

	for ENblock in test_ENdata
		ilevs, vnls = enblockreader(ENblock)
		append!(qilev, ilevs)
		append!(qvnls, vnls)
	end
	
	vnldict = Dict{Int64, Int64}(zip(qilev, qvnls))

	energy, params, bindex, b2j, findex, f2j, deltal = rrblockreader(test_RRblock)
	pblock = rrparamblockunpacker(params)
	
	te = energy[test_index]
	tp = pblock[test_index]
	tgb = b2j[test_index] + 1
	tgf = f2j[test_index] + 1
	tbi = bindex[test_index]

	vnl = get(vnldict, tbi, 0)
	tlb = vnl % 100

	test_energies = collect(minenergy:slice:maxenergy)
end

# ╔═╡ 2822d0de-da6f-4e1a-8df8-29eadb455fcc
print(test_RRblock)

# ╔═╡ 42ab7cc6-c06a-4d84-ba76-34c214526cbe
begin
	function uνlaw(ν, T)
		# blackbody law
		coeff 	= 8 * π * ν ^ 2 / c ^ 3
		exp 	= h * ν / (ℯ ^ (h * ν / (kb * T)) - 1)
		return coeff * exp
	end

	function maxwellianED(E, T)
		# maxwell thermal distribution of velocities
		if E <= 0
			return 0
		else
			coeff = 2 * π * (1 / (π * kb * T))^(3/2)
			exponent = exp(- E / (kb * T))
			return E ^ 0.5 * coeff * exponent
		end
	end

	function dilutionfactor(r, WDradius)
		# standard dilution factor of white-dwarf radiation field
		if r >= WDradius
			radicand = 1 - (WDradius / r)^2
		else 
			radicand = 1
		end

		w = 0.5 * (1 - sqrt(radicand))
		return w
	end

	function opticaldepth(ν)
		# calculates the optical depth at a given radius 
		# dτ / dr = σN(H⁰) + σN(He⁰) + σN(He¹) ...
		# to be implemented later...
		return 0
	end
end

# ╔═╡ 8ca245db-94b1-4f0c-b51d-9267114e9789
function caseB(T, gsm, esm, ENcontent, RRcontent)
	ENheader, ENdata = ENcontent
	RRheader, RRdata = RRcontent

	# energy level extraction
	qilev = Int64[]
	qvnls = Int64[]
		
	# extracting the EN blocks to usable storage
	for ENblock in ENdata
		ilevs, vnls = enblockreader(ENblock)
	
		append!(qilev, ilevs)
		append!(qvnls, vnls)
	end
	
	nblocks = pyconvert(Int64, RRheader["NBlocks"])
	vnldict = Dict{Int64, Int64}(zip(qilev, qvnls))
	verbose && @info("vnl dict length: $(length(vnldict))")

	totalrate = 0.0

	for i in 0:(nblocks-1)
		verbose && @info("Block $i / $(nblocks-1)")

		block = RRdata[i]
		egrid = pyconvert(Vector{Float64}, block["EGRID"])
		RRxsecMATRIX = pyconvert(Array{Float64}, block["RR crosssection"]) #[test_index, :] is the correct access format
		
		energy, _, bindex, b2j, findex, f2j, _ = rrblockreader(block)
		mask = (bindex .∉ Ref(gsm)) .&& (findex .∈ Ref(esm))

		fenergy = energy[mask]
		fRRxsecMATRIX = RRxsecMATRIX[mask, :]

		for j in 1:length(fenergy)
			xsect = fRRxsecMATRIX[j, :]
			intrp = linear_interpolation(egrid, xsect, extrapolation_bc=Line())
			ethrm = fenergy[j]
			emax = maxenergy - ethrm
			
			ratej, _ = quadgk(0.0, emax) do x
				eabsl = ethrm + x
				σ = intrp(x) .* 1e-20
				fE = maxwellianED(eabsl, T)
				v  = sqrt(eabsl)
	        return σ * fE * v * velscaling
    		end

			
			totalrate += ratej
		end
	end
	return totalrate
end

# ╔═╡ 52c18186-8b20-4291-89a9-9e3c363d8be6
caseB(1e4, groundstatemasks[test_nelectrons], excitedstatemasks[test_nelectrons], test_ENcontents, test_RRcontents)

# ╔═╡ 4f81ba64-7410-4a38-90d6-ded67520322d
begin
	"""
	Photoionization and radiative crosssection functions
	"""
	function bfoscstrength(photonenergy::Float64, p::Prr, lb, thermalizingenergy)
		# l 	:= orbital angular momentum of ionized shell
		
		ΔE = photonenergy - thermalizingenergy

		if ΔE <= 0
			return 0
		end
		
		x = abs((ΔE + p.p1) / p.p4)
		y = (1 + p.p3) / (sqrt(x) + p.p3)
		dgf = photonenergy / (ΔE+ p.p4) * p.p1 * (x ^ (-3.5 - lb + 0.5 * p.p2)) * y ^ p.p2

		# dgf is in units of hartree^-1 [1 / energy]
		# function logic to calculate dgf has been verified
		
		if dgf <= 1e-10
			return 0
		else
			return dgf
		end
	end
	
	function photocrosssection(E, p::Prr, gb, gf, lb, lf, thermalenergy)
		# gives the crosssection in cm^2 of the species for an incident photon energy

		if E <= thermalenergy
			return 0
		end
		
		dgf = bfoscstrength(E, p, lb, thermalenergy)

		σ = (2 * π * α / gb) * dgf * a0 ^2
		return σ
	end

	function radiativecrosssection(E, p::Prr, gb, gf, lb, lf, thermalenergy)
		# recombination cross section of radiative recombinations
		# the formula provided in FAC requires units of Hartree
		δ = 1e-7 # factor for numerical stability because of the 1/ε term
		
		ω = E * eV_to_Hartree
		#println("ω: $ω")
		
		ε = ω - thermalenergy * eV_to_Hartree
		#println("ε: $ε")

		if ε <= 0
			return 0
		end
    
	    if ε < δ
	        # At the exact threshold, use the analytically regularized limit 
	        # where the 1/ε singularity cancels out.
	        denom = δ
		else
			denom = (ε * (1 + 0.5 * α ^ 2 * ε))
	    end
		
		coeff = α ^ 2 / 2 * (gb / gf) * ω ^ 2 / denom
		#println("c: $coeff")

		σ = photocrosssection(E, p, gb, gf, lb, lf, thermalenergy)
		# photocrosssection already returns the values in cm^2
		#println("σ: $σ")

		xsection = coeff * σ
		if xsection <= 1e-30
			return 0
		else 
			return xsection
		end
	end
end

# ╔═╡ Cell order:
# ╠═fde54e19-d6fe-4134-84fb-c39be3c0eea2
# ╠═06a00479-13de-4fc2-83c7-ba263895ad86
# ╠═f1a53e3c-fd60-4c8c-8931-66a92e28dc36
# ╠═2822d0de-da6f-4e1a-8df8-29eadb455fcc
# ╠═9194879b-4d99-4416-bbf4-f1ac4820d40e
# ╠═52c18186-8b20-4291-89a9-9e3c363d8be6
# ╠═8ca245db-94b1-4f0c-b51d-9267114e9789
# ╟─b30d80a6-fb36-4baf-bb75-a17e319566c3
# ╠═fbb8819e-da92-4cd7-a3f0-811a27aa3ae4
# ╠═1256163d-1be6-4742-a080-baf5136caf83
# ╠═729ba9a7-bcc2-4386-9348-7cde5d21de27
# ╠═1c6a1132-bd43-4d5b-a03c-08cf21fb24db
# ╠═42ab7cc6-c06a-4d84-ba76-34c214526cbe
# ╠═4f81ba64-7410-4a38-90d6-ded67520322d
