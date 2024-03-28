"""Phono3py workflow context managing"""

from vibes.phonopy.context import PhonopyContext

from . import _defaults as defaults
from . import postprocess, wrapper


class Phono3pyContext(PhonopyContext):
    """PhonopyContext with changed name"""

    def __init__(self, *args, **kwargs):
        kw = {"name": defaults.name, "template_dict": defaults.settings_dict}
        kwargs.update(kw)
        super().__init__(*args, **kwargs)

        self.backend = wrapper
        self.postprocess = postprocess
