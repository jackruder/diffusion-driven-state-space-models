"""Vendored ermongroup CSDI (github.com/ermongroup/CSDI, MIT).

Verbatim copies of ``diff_models.py`` (the ``diff_CSDI`` denoiser) and
``main_model.py`` (``CSDI_base`` / ``CSDI_Forecasting`` — DDPM ε-MSE loss,
ancestral sampler, side-info builder), with the single edit that
``main_model.py``'s top-level ``from diff_models import diff_CSDI`` is made a
package-relative import. Kept byte-for-byte otherwise so that
:class:`~ddssm.model.transitions.csdi_transition.CSDITransition` exercises the
*literal* CSDI code path — a clean indictment/exoneration of our own transition
code. Vendored (not sys.path-hacked to /tmp) so checkpoints stay reloadable.
"""
