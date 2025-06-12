"""Define settings for the application."""

# %% IMPORTS

import pydantic as pdt
import pydantic_settings as pdts

from qto_categorizer_ml import jobs

# %% SETTINGS


class Settings(pdts.BaseSettings, strict=True, frozen=True, extra="forbid"):
    """Base class for application settings.

    Use settings to provide high-level preferences.
    i.e., to separate settings from initialization (e.g., CLI).
    """


class MainSettings(Settings):
    """Main settings for the application.

    Parameters:
        job (jobs.JobKind): job to run.
    """

    job: jobs.JobKind = pdt.Field(..., discriminator="KIND")
