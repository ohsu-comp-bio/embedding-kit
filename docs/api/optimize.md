# Optimize API

Training loop functions for VAE and general models.

## fit_vae

::: embkit.optimize.fit_vae
    options:
      show_source: false

## fit

::: embkit.optimize.fit
    options:
      show_source: false

## Multitask training

Weighted and interleaved multitask training over a list of `LearningTask`
objects. Each task supplies its own model, dataset, batch size, and loss, and
the two helpers coordinate the shared optimizer across them. `LearningTask`
describes one task; the two train functions share the `pairing_mode`,
`task_schedule`, gradient-clipping, and learning-rate-schedule options.

### LearningTask

::: embkit.optimize.multitask.LearningTask
    options:
      show_source: false
      members:
        - __init__
      merge_init_into_class: true

### multi_task_train_sync

Weighted, synchronized multitask training. Each epoch draws one batch from every
task per step and backpropagates the summed, weighted loss.

::: embkit.optimize.multitask.multi_task_train_sync
    options:
      show_source: false

`multi_task_train_weighted_sync` is an alias for `multi_task_train_sync`.

### multi_task_train_interleaved

Interleaved multitask training that cycles through tasks one at a time, using a
`task_schedule` to control task order.

::: embkit.optimize.multitask.multi_task_train_interleaved
    options:
      show_source: false
