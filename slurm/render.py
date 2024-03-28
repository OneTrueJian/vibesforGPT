"""Rendering the actual jobscript."""

from jinja2 import Template


template = r"""#!/bin/bash -l

#SBATCH -J {{ name }}|{{ tag }}
#SBATCH -o {{ logfile }}.%j
#SBATCH -e {{ logfile}}.%j
#SBATCH -D ./
#SBATCH --mail-type={{ mail_type }}
#SBATCH --mail-user={{ mail_address }}
#SBATCH --nodes={{ nodes }}
#SBATCH --ntasks-per-node={{ cores }}
#SBATCH --ntasks-per-core=1
#SBATCH -t {{ h }}:{{ min }}:00
#SBATCH --partition={{ queue }}

{{ pre_command }}
{{ command }}

"""


def render(arguments):
    """Render the jobscript."""
    return Template(template).render(arguments)
