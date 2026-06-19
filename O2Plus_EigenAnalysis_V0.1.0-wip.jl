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
	using Interpolations
	
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
	const α 	= 1 / 137.036 	# fine structure constant
	
	const eV_to_Hartree = 1 / 27.211 # atomic units conversion factor
	const Hartree_to_eV = 27.211     # eV conversion factor
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

# ╔═╡ aa1c0378-513c-4f1e-89da-248ed3e11574
rr_RRGRD[1]

# ╔═╡ 006980c0-28ab-4f39-929d-1fa64862e0c7
begin
	rr_PIGRD[1]
end

# ╔═╡ a5f0e19b-1067-4156-a7bb-4018ac16b69f
begin
	function bfoscstrength(photonenergy::Float64, p::Vector{Float64}, lb, thermalizingenergy)
		# l 	:= orbital angular momentum of ionized shell
		# E_p  	:= incident photon energy
		# E_th 	:= ionization threshold energy
		ΔE = photonenergy - thermalizingenergy
		
		x = abs((ΔE + p[4]) / p[4])
		y = (1 + p[3]) / (sqrt(x) + p[3])
		dgf = photonenergy / (ΔE+ p[4]) * p[1] * (x ^ (-3.5 - lb + 0.5 * p[2])) * y ^ p[2]

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

# ╔═╡ f3ed0fa9-c9f8-4040-b1a4-a4dd7122f953
begin
	scatter(rr_EGRID, rr_gf)
	plot!(rrE_vec .- rr_ΔE, bf_strn_vec)
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
	
	function photocrosssection(ν, p, gb, gf, lb, lf, thermalenergy)
		# gives the crosssection in cm^2 of the species for an incident photon energy
		
		photonenergy = h * ν
		dgf = bfoscstrength(photonenergy, p, lb, thermalenergy)

		σ = (2 * π * α / gb) * dgf * a0 ^2
		return σ
	end

	function radiativecrosssection(ν, p, gb, gf, lb, lf, thermalenergy)
		# recombination cross section of radiative recombinations
		# the formula provided in FAC requires units of Hartree
		
		ω = h * ν * eV_to_Hartree
		#println("ω: $ω")
		
		ε = ω - thermalenergy * eV_to_Hartree
		#println("ε: $ε")
		
		coeff = α ^ 2 / 2 * (gb / gf) * ω ^ 2 / (ε * (1 + 0.5 * α ^ 2 * ε))
		#println("c: $coeff")

		σ = photocrosssection(ν, p, gb, gf, lb, lf, thermalenergy)
		# photocrosssection already returns the values in cm^2
		#println("σ: $σ")
		
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

# ╔═╡ 4369932c-7fb5-4493-9c60-824c04b91fc1
begin
	ν0 = (rr_ΔE + rr_EGRID[1]) / h
	radiativecrosssection(ν0, rr_PARAM, 1, 2, 0, 0, rr_ΔE) / 1e-20 
end

# ╔═╡ fc950dab-3598-4194-b867-063420f3dee9
begin
	νvec = rrE_vec ./ h
	pivec  	= [photocrosssection(ν, rr_PARAM, 1, 1, 0, 0, rr_ΔE) for ν in νvec] ./ 1e-20
	xsecvec = [radiativecrosssection(ν, rr_PARAM, 1, 1, 0, 0, rr_ΔE) for ν in νvec] ./ 1e-20
	
end

# ╔═╡ 161a37a3-7a3b-44fb-9dc5-80488b5f375c
begin
	scatter(rr_EGRID, rr_RRGRD)
	plot!(rrE_vec .- rr_ΔE, xsecvec)
end

# ╔═╡ 30ad6c1b-743e-4f8e-ab48-bb0b8f8dc531
begin
	scatter(rr_EGRID, rr_PIGRD)
	plot!(rrE_vec .- rr_ΔE, pivec)
end

# ╔═╡ 9532a371-c3ab-4a0c-8d4f-7f06e01e33bb
pivec[1]

# ╔═╡ Cell order:
# ╠═fde54e19-d6fe-4134-84fb-c39be3c0eea2
# ╠═0a1bb704-6391-11f1-9b23-a91665262d3d
# ╠═06a00479-13de-4fc2-83c7-ba263895ad86
# ╠═f06bfd41-3f5e-483b-b421-756e7ce0fcff
# ╠═ede3463c-65f7-4bcb-b873-6190340b11ce
# ╠═c2420e2a-34c4-4d51-a72c-19257abf14ed
# ╠═8905abb5-93df-4271-8ac2-081b2db56665
# ╠═d79c83b9-f3e0-497a-8ddc-1473f789ac99
# ╠═8100381f-fe1a-46fd-a039-cd4d77c29d55
# ╠═a9ac6d4d-bd42-446b-a08f-cc9f5851b2a4
# ╠═f3ed0fa9-c9f8-4040-b1a4-a4dd7122f953
# ╠═161a37a3-7a3b-44fb-9dc5-80488b5f375c
# ╠═aa1c0378-513c-4f1e-89da-248ed3e11574
# ╠═4369932c-7fb5-4493-9c60-824c04b91fc1
# ╠═fc950dab-3598-4194-b867-063420f3dee9
# ╠═30ad6c1b-743e-4f8e-ab48-bb0b8f8dc531
# ╠═006980c0-28ab-4f39-929d-1fa64862e0c7
# ╠═9532a371-c3ab-4a0c-8d4f-7f06e01e33bb
# ╠═a5f0e19b-1067-4156-a7bb-4018ac16b69f
# ╠═42ab7cc6-c06a-4d84-ba76-34c214526cbe
