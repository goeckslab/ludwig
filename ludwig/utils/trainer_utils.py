import logging


def get_terminal_steps_per_checkpoint(
    steps_per_epoch: int, steps_per_checkpoint: int = 0, checkpoints_per_epoch: float = 0, should_log: bool = False
):
    """Returns the terminal steps per checkpoint to use for the training loop, given user+default inputs."""

    if steps_per_checkpoint != 0 and checkpoints_per_epoch != 0:
        raise ValueError(
            "It is invalid to specify both checkpoints_per_epoch AND steps_per_checkpoint. Please specify one or the "
            "other, or specify neither to checkpoint/eval the model every epoch."
        )

    # Set steps_per_checkpoint based on the checkpoints_per_epoch, if it was specified.
    if checkpoints_per_epoch != 0:
        steps_per_checkpoint = int(steps_per_epoch / checkpoints_per_epoch)

    # Check steps_per_checkpoint and cap it at steps_per_epoch.
    if steps_per_checkpoint == 0 or steps_per_checkpoint > steps_per_epoch:
        steps_per_checkpoint = steps_per_epoch
        if should_log:
            logging.info(
                f"Note: steps_per_checkpoint (was {steps_per_checkpoint}) is now set to the number of "
                f"steps per epoch: {steps_per_epoch}.\n"
            )

    return steps_per_checkpoint
