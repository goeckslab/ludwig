import pytest

from ludwig.utils import trainer_utils


def test_get_terminal_steps_per_checkpoint():
    # steps_per_checkpoint and checkpoints_per_epoch cannot both be specified.
    with pytest.raises(Exception):
        trainer_utils.get_terminal_steps_per_checkpoint(
            steps_per_epoch=1024,
            steps_per_checkpoint=1,
            checkpoints_per_epoch=1,
        )

    assert trainer_utils.get_terminal_steps_per_checkpoint(steps_per_epoch=1024, steps_per_checkpoint=100) == 100
    assert trainer_utils.get_terminal_steps_per_checkpoint(steps_per_epoch=1024, steps_per_checkpoint=2048) == 1024
    assert trainer_utils.get_terminal_steps_per_checkpoint(steps_per_epoch=1024, checkpoints_per_epoch=2) == 512
    assert trainer_utils.get_terminal_steps_per_checkpoint(steps_per_epoch=1024, checkpoints_per_epoch=2.5) == 409
    assert trainer_utils.get_terminal_steps_per_checkpoint(steps_per_epoch=1024, checkpoints_per_epoch=0.5) == 1024
    assert trainer_utils.get_terminal_steps_per_checkpoint(steps_per_epoch=1024) == 1024
    assert (
        trainer_utils.get_terminal_steps_per_checkpoint(
            steps_per_epoch=1024, steps_per_checkpoint=0, checkpoints_per_epoch=0
        )
        == 1024
    )
