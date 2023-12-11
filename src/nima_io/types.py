"""Type definitions used throughout the nima_io package."""

from typing import Union  # py3.9 still requires this

Kwargs = dict[str, Union[str, int, float, bool, None]]
