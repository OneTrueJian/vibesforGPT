"""Keys and names for arrays. Composite keys are concatenated by `_`"""


def _join(*keys):
    """join two or more keys"""
    keys = (k for k in keys if k)
    return "_".join(keys)


# geometry
symprec = "symprec"

# generic suffixes
aux = "aux"
total = "total"
scalar = "scalar"
cumulative = "cumulative"
remapped = "remapped"
flattened = "flattened"
reference = "reference"
cumtrapz = "cumtrapz"

# force constants
fc = "force_constants"
fc_remapped = "force_constants_remapped"
fc_flattened = "force_constants_flattened"

system_name = "system_name"
time_unit = "time_unit"
timestep = "timestep"

reference_atoms = "atoms_reference"
reference_primitive = "atoms_primitive"
reference_supercell = "atoms_supercell"
reference_positions = "positions_reference"
reference_lattice = "lattice_reference"
metadata = "raw_metadata"

cell = "cell"
volume = "volume"
positions = "positions"
velocities = "velocities"
momenta = "momenta"

forces = "forces"
forces_harmonic = "forces_harmonic"
energy = "energy"
energy_kinetic = "energy_kinetic"
energy_potential = "energy_potential"
energy_potential_harmonic = "energy_potential_harmonic"
pressure = "pressure"
pressure_kinetic = "pressure_kinetic"
pressure_potential = "pressure_potential"
stress = "stress"
stress_kinetic = "stress_kinetic"
stress_potential = "stress_potential"
stresses = "stresses"
stresses_potential = "stresses_potential"
virials = "virials"
temperature = "temperature"

heat_flux = "heat_flux"
heat_fluxes = "heat_fluxes"
heat_flux_aux = _join(heat_flux, aux)
heat_fluxes_aux = _join(heat_fluxes, aux)
heat_flux_total = _join(heat_flux, total)

kappa = "kappa"

gk_prefactor = "gk_prefactor"

sigma = "sigma"
sigma_mode = "sigma_mode"
sigma_per_sample = "sigma_per_sample"


# time
time = "time"
omega = "omega"
autocorrelation = "autocorrelation"
fourier_transform = "fourier_transform"
avalanche_data = "avalanche_function"
time_avalanche = "avalanche_time"
time_cutoff = "cutoff_time"
avalanche_function = "avalanche_function"
avalanche_index = "avalanche_index"
power_spectrum = "power_spectrum"

# molecular dynamics
nve = "NVE"
nvt = "NVT"
npt = "NPT"
nsteps = "nsteps"
dt = "dt"

# file management
file = "file"
default_backup_folder = "backups"
cache = "cache.vibes"
workdir = "workdir"

# hash
name = "name"
hash = "hash"
hash_raw = "hash_raw"
trajectory = "trajectory"
st_size = "st_size"

# composite keys
heat_flux_autocorrelation = _join(heat_flux, autocorrelation)
heat_flux_aux_autocorrelation = _join(heat_flux_aux, autocorrelation)
heat_flux_autocorrelation_scalar = _join(heat_flux_autocorrelation, scalar)
kappa_cumulative = _join(heat_flux_autocorrelation, cumtrapz)
kappa_cumulative_scalar = _join(kappa_cumulative, scalar)
heat_flux_power_spectrum = _join(heat_flux, power_spectrum)
heat_flux_total_power_spectrum = _join(heat_flux_total, power_spectrum)
heat_flux_power_spectrum_scalar = _join(heat_flux_power_spectrum, scalar)
heat_flux_aux_power_spectrum = _join(heat_flux_aux_autocorrelation, fourier_transform)
heat_flux_aux_power_spectrum_scalar = _join(heat_flux_aux_power_spectrum, scalar)

# abbreviations
hf = heat_flux
hf_aux = heat_flux_aux
hf_acf = heat_flux_autocorrelation
hf_aux_acf = heat_flux_aux_autocorrelation
hf_acf_scalar = heat_flux_autocorrelation_scalar
hf_power = heat_flux_power_spectrum
hf_aux_power = heat_flux_aux_power_spectrum
k_cum = kappa_cumulative
k_cum_scalar = kappa_cumulative_scalar


# relaxation
relaxation = "relaxation"
relaxation_options = "relaxation_options"
expcellfilter = "ExpCellFilter"

# atoms and calculator
atoms = "atoms"
calculator = "calculator"
parameters = "parameters"
calculator_parameters = ".".join([calculator, parameters])
settings = "settings"

aims_uuid = "aims_uuid"
