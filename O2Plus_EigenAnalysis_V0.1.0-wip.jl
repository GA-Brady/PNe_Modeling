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
	# block defining physical constants
	const a0 	= 5.291e-9  # bohr radius in cm
	const h 	= 4.135e-15 # eV ⋅ s
	const c 	= 2.99e10  	# cm / s
	const kb 	= 8.62e-5   # eV / Kelvin 
	const α 	= 1 / 137.036 	# fine structure constant

	# conversion factors
	const eV_to_Hartree = 1 / 27.211 # atomic units conversion factor
	const Hartree_to_eV = 27.211     # eV conversion factor
end;

# ╔═╡ 5df90933-6f23-4137-9c04-95008732fb77
begin
    for i in 1:10
        # Your long execution code here
        
        @info("Current Block $i / 10")
        sleep(0.5)
    end
end

# ╔═╡ 8905abb5-93df-4271-8ac2-081b2db56665
begin
	function FACrrunpacker(filepath)
		  if !occursin(".rr", filepath)
			  println("filepath: $filepath does not contain a .rr file")
			  return nothing
		  end
		
		header, datablocks = rfac.read_rr(filepath)

		# since we only care about ground-ground transitions
		print(header)
	end
end

# ╔═╡ 51e01d0a-a289-4ba0-87b0-f0ba3d5e6973
#=╠═╡
plot(testenergies, crosssections ./ 1e-20)
  ╠═╡ =#

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
		if xsection <= 1e-25
			return 0
		else 
			return xsection
		end
	end
end

# ╔═╡ f1a53e3c-fd60-4c8c-8931-66a92e28dc36
begin
	nelectrons = 4

	OxygenENfile = joinpath(fac_dir, "O0$(nelectrons)a.en")
	ENheader, ENdata = rfac.read_en(OxygenENfile)

	O04ilev = []
	O04vnls  = []

	for ENblock in ENdata
		ilevs, vnls = enblockreader(ENblock)

		append!(O04ilev, ilevs)
		append!(O04vnls, vnls)
	end

	# the q+1 ground state is ALWAYS the first entry in the second EN block
	# for all ionization states of oxygen (manually verified)

	# finding the q+1 ground state free-index
	# since ENdata is a python object ∴ zero-indexed
	qplusblock = ENdata[1]
	qplusilev, _ = enblockreader(qplusblock)
	qplusground = qplusilev[1]

	OxygenRRfile = joinpath(fac_dir, "O0$(nelectrons)a.rr")
	RRheader, RRdata = rfac.read_rr(OxygenRRfile)

	nblocks = pyconvert(Int64, RRheader["NBlocks"])
	testenergies = collect(0:.1:150)
	crosssections = zeros(length(testenergies))

	for i in 0:(nblocks-1)
		# extracting vectors from the blocks
		energy, params, bindex, b2j, findex, f2j, deltal = rrblockreader(RRdata[i])
		prrvec = rrparamblockunpacker(params)
	
		# masks to avoid unnecessary computation
		bmask = bindex .!= 0  # only to excited states
		fmask = findex .== qplusground # only from ground state of O03
		
		# total mask of case B recombination
		tmask = bmask .&& fmask
	
		# filtering our values to the ones we really care about
		filtered_energy = energy[tmask]
		filtered_prrvec = prrvec[tmask]
		filtered_bindex = bindex[tmask]
		filtered_b2j    = b2j[tmask]
		filtered_findex = findex[tmask]
		filtered_f2j    = f2j[tmask]
		filtered_deltal = deltal[tmask]
	
		num = length(filtered_energy)
	
		for j in 1:1:num
		# will need to use lookup table to get free lb, lf does not matter.
		# radiativecrosssection(E, p::Prr, gb, gf, lb, lf, thermalenergy)
			for (k, energy) in enumerate(testenergies)
				vnlindex = findfirst(isequal(filtered_bindex[j]), O04ilev)
				lb = O04vnls[vnlindex] % 100
				
				σ = radiativecrosssection(energy, 
							   filtered_prrvec[j],
							   filtered_b2j[j] + 1,
							   filtered_f2j[j] + 1,
							   lb,
							   0,
							   filtered_energy[j]
							  )
				crosssections[k] += σ
			end
		end
		@info("Block $i / $nblocks")
	end

	df = DataFrame(ENERGY = testenergies, XSECTN = crosssections)
	CSV.write("my_output.csv", df)
end

# ╔═╡ Cell order:
# ╠═fde54e19-d6fe-4134-84fb-c39be3c0eea2
# ╠═06a00479-13de-4fc2-83c7-ba263895ad86
# ╠═5df90933-6f23-4137-9c04-95008732fb77
# ╟─8905abb5-93df-4271-8ac2-081b2db56665
# ╠═f1a53e3c-fd60-4c8c-8931-66a92e28dc36
# ╠═51e01d0a-a289-4ba0-87b0-f0ba3d5e6973
# ╠═fbb8819e-da92-4cd7-a3f0-811a27aa3ae4
# ╠═1256163d-1be6-4742-a080-baf5136caf83
# ╠═729ba9a7-bcc2-4386-9348-7cde5d21de27
# ╠═1c6a1132-bd43-4d5b-a03c-08cf21fb24db
# ╠═42ab7cc6-c06a-4d84-ba76-34c214526cbe
# ╠═4f81ba64-7410-4a38-90d6-ded67520322d
