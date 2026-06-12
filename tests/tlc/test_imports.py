"""Import test for the 3LC integration.

Importing the integration package runs the 3lc / 3lc-ultralytics version guard and
imports every integration module, so this catches API breaks against the installed
3lc packages without needing an API key or any data.
"""


def test_import_integration() -> None:
    import utils.loggers.tlc

    assert utils.loggers.tlc.create_dataloader is not None
