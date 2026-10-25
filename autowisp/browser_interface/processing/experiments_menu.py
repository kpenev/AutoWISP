"""Decide what the progress page's experiments menu offers for each one."""


def plan_experiments_menu(described, selected=None):
    """
    Return the menu's rows, and whether an experiment needs attention.

    Each row says which actions the menu offers for one experiment, decided
    from its record alone, by the same rules the engine applies:

    * **Selecting** it is offered only while no operation (create, reset,
      merge, restore or delete) is working on it, and none was interrupted
      part way. ``describe_experiments`` then gives it no operation.
    * **Deleting** it is offered unless it is being merged or its merge was
      interrupted, which the engine refuses, or its operation is running on
      another machine, which cannot be stopped from here. An operation
      running on this machine is stopped by the delete.
    * **Restoring** the original is offered when a merge was interrupted, or
      the restore of one was: the experiment is then ``merging`` with
      nothing running.

    An experiment needs attention when it is stale, or an operation on it
    was interrupted: either leaves something for the user to do.

    Args:
        described([dict]):    The experiments, as
            :meth:`.ExperimentRecords.describe_experiments` returns them.

        selected(str):    The name of the experiment selected, whose own
            actions the menu shows separately; left out of the rows. None in
            the original.

    Returns:
        [dict]:    Each experiment but the selected one, as described, with
        ``selectable``, ``offer_delete`` and ``offer_restore`` added.

        bool:    Whether any experiment, the selected one included, needs
        attention.
    """

    rows = []
    for experiment in described:
        if experiment["name"] == selected:
            continue
        operation = experiment["operation"]
        running = None if operation is None else operation["running"]
        merging = experiment["state"] == "merging"
        rows.append(
            {
                **experiment,
                "selectable": operation is None,
                "offer_delete": not merging and running in (None, "here"),
                "offer_restore": merging and running is None,
            }
        )
    needs_attention = any(
        experiment["state"] == "stale"
        or (
            experiment["operation"] is not None
            and experiment["operation"]["running"] is None
        )
        for experiment in described
    )
    return rows, needs_attention
