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
	const a0 = 5.291e-9 # bohr radius in cm
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
	#println(data)
	
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

# ╔═╡ 8100381f-fe1a-46fd-a039-cd4d77c29d55
begin
	O4P_rr_file = joinpath(fac_dir, "O04a.rr")
	rr_header, rr_data = rfac.read_rr(O4P_rr_file)
	print(rr_data[0])

	pyrr_PARAM 	= rr_data[0]["parameters"][0]
	pyrr_EGRID 	= rr_data[0]["EGRID"]
	pyrr_PIGRD 	= rr_data[0]["PI crosssection"][0]
	pyrr_gf 	= rr_data[0]["gf"][0]
	pyrr_ΔE   	= rr_data[0]["Delta E"][0]

	rr_ΔE    = pyconvert(Float64, pyrr_ΔE)
	
	rr_PARAM = pyconvert(Vector{Float64}, pyrr_PARAM)
	rr_EGRID = pyconvert(Vector{Float64}, pyrr_EGRID) 
	rr_PIGRD = pyconvert(Vector{Float64}, pyrr_PIGRD)
	rr_gf 	 = pyconvert(Vector{Float64}, pyrr_gf)
end

# ╔═╡ a5f0e19b-1067-4156-a7bb-4018ac16b69f
begin
	function bf_osc_strength(E_ph::Float64, p::Vector{Float64}, g, l, E_th)
		# g 	:= statistical weight BEFORE ionization (2J + 1)
		# l 	:= orbital angular momentum of ionized shell
		# E_p  	:= incident photon energy
		# E_th 	:= ionization threshold energy
		E_e = E_ph - E_th
		
		x = abs((E_e + p[4]) / p[4])
		y = (1 + p[3]) / (sqrt(x) + p[3])
		dgf = E_ph / (E_e + p[4]) * p[1] * (x ^ (-3.5 - l + 0.5 * p[2])) * y ^ p[2]

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
	rrE_max = maximum(rr_ΔE + 1000)

	rrE_vec = collect(LinRange(rrE_min, rrE_max, 100))
	bf_strn_vec  = [bf_osc_strength(E, rr_PARAM, 1, 0, rr_ΔE) for E in rrE_vec]

	PI_xsection  = (2 * π / 137 * a0^2 / 1e-20) .* bf_strn_vec
end

# ╔═╡ 612f40cc-0803-4cf5-ab80-960b10ff507e
begin
	scatter(rr_EGRID, rr_gf)
	plot!(rrE_vec .- rr_ΔE, bf_strn_vec)
end

# ╔═╡ b5a3e944-f9cb-4ef5-bf69-f03270e06461
begin
	scatter(rr_EGRID, rr_PIGRD)
	plot!(rrE_vec .- rr_ΔE, PI_xsection)
end

# ╔═╡ 42ab7cc6-c06a-4d84-ba76-34c214526cbe
begin
	function uνlaw(ν, T)
		# blackbody law 
		h 	= 6.63e-27
		c 	= 2.99e10
		kb 	= 1.38e-16 
		
		coeff 	= 8 * π * ν ^ 2 / c ^ 3
		exp 	= h * ν / (ℯ ^ (h * ν / (kb * T)) - 1)
		return coeff * exp
	end

	begin volumephotoionizerate()
		# volumetric photoionizing rate 
		

	end
end

# ╔═╡ Cell order:
# ╠═fde54e19-d6fe-4134-84fb-c39be3c0eea2
# ╠═0a1bb704-6391-11f1-9b23-a91665262d3d
# ╠═06a00479-13de-4fc2-83c7-ba263895ad86
# ╠═f06bfd41-3f5e-483b-b421-756e7ce0fcff
# ╠═ab5b9592-5f39-4fc4-a890-29a82f311816
# ╠═02fda065-185a-4d03-b47a-278c5b2babe8
# ╠═8100381f-fe1a-46fd-a039-cd4d77c29d55
# ╠═a5f0e19b-1067-4156-a7bb-4018ac16b69f
# ╠═a9ac6d4d-bd42-446b-a08f-cc9f5851b2a4
# ╠═612f40cc-0803-4cf5-ab80-960b10ff507e
# ╠═b5a3e944-f9cb-4ef5-bf69-f03270e06461
# ╠═42ab7cc6-c06a-4d84-ba76-34c214526cbe
