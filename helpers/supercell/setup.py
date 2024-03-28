from numpy.distutils.core import Extension


ext = Extension(name="supercell", sources=["linalg.f90", "supercell.f90"])

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(
        name="supercell",
        description="find cubic supercells",
        author="Florian Knoop",
        ext_modules=[ext],
    )
