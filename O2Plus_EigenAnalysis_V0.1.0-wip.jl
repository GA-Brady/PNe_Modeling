### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ fde54e19-d6fe-4134-84fb-c39be3c0eea2
begin
    import Pkg
    # Disables Pluto's isolated environment and uses your terminal project instead
    Pkg.activate(".") 
    
    using PythonCall
    np = pyimport("numpy")
end

# ╔═╡ 0a1bb704-6391-11f1-9b23-a91665262d3d
begin
	using Plots
	using LinearAlgebra
	using DataFrames
	using Statistics
	using QuadGK
	
	fac 	= pyimport("pfac.fac")
	ftab  	= pyimport("pfac.table")
	rfac    = pyimport("pfac.rfac")
	sys 	= pyimport("sys")
end

# ╔═╡ 06a00479-13de-4fc2-83c7-ba263895ad86
begin
	const a0 	= 5.291e-9  # bohr radius in cm
	const h 	= 4.135e-15 # eV ⋅ s
	const c 	= 2.99e10  	# cm / s
	const kb 	= 8.62e-5   # eV / Kelvin 
	const α 	= 1 / 137 	# fine structure constant
end

# ╔═╡ f06bfd41-3f5e-483b-b421-756e7ce0fcff
begin
	# collisional excitation stuff
	fac_dir  	= joinpath(pwd(), "AMDS_data")
	O4Pci_file 	= joinpath(fac_dir, "O04a.ci")
	
	data = rfac.read_ci(O4Pci_file)

	# rfac.read_ci returns back a two element list
	# !! these are python objects so they are 0-indexed!!
	# element 0 - header (which appears to be a dict)
	# element 1 - tuple of blocks (each element of the tuple appears to be a dict)
	println(data[0])
	
	bound_2j 	= 0
	free_index 	= 0
	
	println("gridstuff")
	#println(data[1][bound_2j])

	pyEGRID 	= data[1][bound_2j]["EGRID"]

	# We are taking data from the jth block
	pyPARAM 	= data[1][bound_2j]["parameters"][free_index]
	pyCSTRN 	= data[1][bound_2j]["collision strength"][free_index]
	pyΔE     	= data[1][bound_2j]["Delta E"][free_index]

	
	# converting python objects to julia objects
	PARAM 	= pyconvert(Vector{Float64}, pyPARAM)
	EGRID 	= pyconvert(Vector{Float64}, pyEGRID)
	CSTRN 	= pyconvert(Vector{Float64}, pyCSTRN)
	ΔE 		= pyconvert(Float64, pyΔE)
	
	println("PARAM: $(PARAM)")
	println("EGRID: $(EGRID)")
	println("CSTRN: $(CSTRN)")
	println("ΔE   : $(ΔE)")

end

# ╔═╡ 1be147dc-92c7-4078-a4b8-d75f3f58d781


# ╔═╡ ab5b9592-5f39-4fc4-a890-29a82f311816
begin
	function collision_strength(p::Vector{Float64}, E_incident::Float64, E_threshold)
		
		x = E_incident ./ E_threshold
		y = 1 .- 1 ./ x
		Ω = p[1] .* log.(x) .+ p[2] .* y.^2 .+ p[3] ./ x .* y .+ p[4] ./ x.^2 .* y
		
		if Ω .<= 0
			return 0
		else
			return Ω
		end
	end
end

# ╔═╡ 02fda065-185a-4d03-b47a-278c5b2babe8
begin
	E_min = minimum(EGRID)
	E_max = maximum(EGRID)
	
	E_vec = collect(LinRange(E_min, E_max, 100))
	S_vec = [collision_strength(PARAM, E, ΔE) for E in E_vec]
	
	scatter(EGRID .+ ΔE, CSTRN)
	plot!(E_vec, S_vec)
end

# ╔═╡ c2420e2a-34c4-4d51-a72c-19257abf14ed
struct RR
	boundindex::Vector{Float64}
	bound2j::Vector{Float64}
	freeindex::Vector{Float64}
	free2j::Vector{Float64}
	deltaE::Vector{Float64}
	deltal::Vector{Float64}
	param::Array{Float64}
end

# ╔═╡ ede3463c-65f7-4bcb-b873-6190340b11ce
struct Species
	rr::RR
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

# ╔═╡ 8100381f-fe1a-46fd-a039-cd4d77c29d55
begin
	O4P_rr_file = joinpath(fac_dir, "O04a.rr")
	rr_header, rr_data = rfac.read_rr(O4P_rr_file)
	print(rr_data[0])

	pyrr_PARAM 	= rr_data[0]["parameters"][0]
	pyrr_EGRID 	= rr_data[0]["EGRID"]
	pyrr_PIGRD 	= rr_data[0]["PI crosssection"][0]
	pyrr_RRGRD  = rr_data[0]["RR crosssection"][0]
	pyrr_gf 	= rr_data[0]["gf"][0]
	pyrr_ΔE   	= rr_data[0]["Delta E"]

	rr_ΔE    = pyconvert(Vector{Float64}, pyrr_ΔE)[1]
	
	rr_PARAM = pyconvert(Array{Float64}, pyrr_PARAM)
	rr_EGRID = pyconvert(Vector{Float64}, pyrr_EGRID) 
	rr_PIGRD = pyconvert(Vector{Float64}, pyrr_PIGRD)
	rr_RRGRD = pyconvert(Vector{Float64}, pyrr_RRGRD)
	rr_gf 	 = pyconvert(Vector{Float64}, pyrr_gf)
end

# ╔═╡ d79c83b9-f3e0-497a-8ddc-1473f789ac99
FACrrunpacker(O4P_rr_file)

# ╔═╡ a5f0e19b-1067-4156-a7bb-4018ac16b69f
begin
	function bfoscstrength(photonenergy::Float64, p::Vector{Float64}, l, thermalizingenergy)
		# l 	:= orbital angular momentum of ionized shell
		# E_p  	:= incident photon energy
		# E_th 	:= ionization threshold energy
		ΔE = photonenergy - thermalizingenergy
		
		x = abs((ΔE + p[4]) / p[4])
		y = (1 + p[3]) / (sqrt(x) + p[3])
		dgf = photonenergy / (ΔE+ p[4]) * p[1] * (x ^ (-3.5 - l + 0.5 * p[2])) * y ^ p[2]

		# dgf is in units of hartree^-1 [1 / energy]
		# function logic to calculate dgf has been verified
		
		if dgf <= 1e-10
			return 0
		else
			return dgf
		end
	end
end

# ╔═╡ a9ac6d4d-bd42-446b-a08f-cc9f5851b2a4
begin
	rrE_min = rr_ΔE
	rrE_max = rr_ΔE + 1000

	rrE_vec = collect(LinRange(rrE_min, rrE_max, 100))
	bf_strn_vec  = [bfoscstrength(E, rr_PARAM, 0, rr_ΔE) for E in rrE_vec]
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
		if E < 0
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

	#=
	photoionized and radiative cross section functions are malformed. There is a broadcast error between xsection functions and bfoscstrength. Need to be reformatted because of the parameter vector. Broadcasting attempts to connect the ν vector and the parameter vector, which are not necessarily the same length
	=#
	function photocrosssection(ν, p, gi, gf, li, lf, thermalenergy)
		# gives the crosssection in cm^2 of the species for an incident photon energy
		
		photonenergy = h * ν
		dgf = bfoscstrength(photonenergy, p, lf, thermalenergy)

		σ = (2 * π * α / gi) * dgf * a0 ^2
		return σ
	end

	function radiativecrosssection(ν, p, gi, gf, li, lf, thermalenergy)
		# recombination cross section of radiative recombinations
		ω = thermalenergy
		ε =  h * ν - ω
		
		coeff = α^2 / 2 * (gi / gf) * ω ^ 2 / (ε * (1 + 0.5 * α^2 * ε))
		σ = photocrosssection(ν, p, gi, gf, li, lf, thermalenergy)
		return coeff * σ
	end

	function volumephotoionizerate(ν0, species, T)
		# volumetric photoionizing rate 
		# needs to be more numerically stable... big error bars
		σ(ν) = crosssection(ν, species)

		# ionizing budget: flux * cross section * optical depth / photon energy
		integrand(ν, T) = uνlaw(ν, T) * σ(ν) / ( h * ν) * exp( - opticaldepth(ν))
		integral, error = quadgk(x -> integrand(x, T), ν0, Inf, rtol=1e-13)

		return integral, error
	end
end

# ╔═╡ fc950dab-3598-4194-b867-063420f3dee9
begin
	scatter(rr_EGRID, rr_RRGRD)
	νvec = rr_ΔE ./ h
	rvec = radiativecrosssection.(νvec, rr_PARAM, 1, 0, 3, 0, rr_ΔE)
end

# ╔═╡ Cell order:
# ╠═fde54e19-d6fe-4134-84fb-c39be3c0eea2
# ╠═0a1bb704-6391-11f1-9b23-a91665262d3d
# ╠═06a00479-13de-4fc2-83c7-ba263895ad86
# ╠═f06bfd41-3f5e-483b-b421-756e7ce0fcff
# ╠═1be147dc-92c7-4078-a4b8-d75f3f58d781
# ╠═ab5b9592-5f39-4fc4-a890-29a82f311816
# ╠═02fda065-185a-4d03-b47a-278c5b2babe8
# ╠═ede3463c-65f7-4bcb-b873-6190340b11ce
# ╠═c2420e2a-34c4-4d51-a72c-19257abf14ed
# ╠═8905abb5-93df-4271-8ac2-081b2db56665
# ╠═d79c83b9-f3e0-497a-8ddc-1473f789ac99
# ╠═8100381f-fe1a-46fd-a039-cd4d77c29d55
# ╠═a9ac6d4d-bd42-446b-a08f-cc9f5851b2a4
# ╠═fc950dab-3598-4194-b867-063420f3dee9
# ╠═a5f0e19b-1067-4156-a7bb-4018ac16b69f
# ╠═42ab7cc6-c06a-4d84-ba76-34c214526cbe
