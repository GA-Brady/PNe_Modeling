To get the O2Plus... pluto notebook to run correctly, you must have the FAC (FLEXIBLE ATOMIC CODE) code installed in a conda environment. Inside this environment, you will also need numpy. After installing these packages, activate the conda environment.

```user/dir % conda activate (fac-env)```

Open julia inside of your project directory

```(fac-env) user/dir % julia --project=.```

Using ```PreferenceTools.jl```, the environment variables within the ```LocalPreferences.toml``` need updated to your conda environment. My local preferences are structured as follows.

```
julia> ]
(project) pkg> preference add CondaPkg backend=Current
(project) pkg> preference add PythonCall exe=/path/to/conda/bin/python 
```

Now the following can be used to open the Pluto Notebook:

```
julia> using Pluto; Pluto.run()
```

The first cell of this Pluto Notebook overwrites the default pluto environment with the current terminal environment. This ensures that the Pluto Notebook will always copy the current terminal environemnt. Therefore, we can force Pluto to use the ```LocalPreferences.toml``` that we built, instead of its own.
