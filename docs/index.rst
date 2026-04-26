SpectraX
========

A JAX-native neural-network library with a **PyTorch-shaped eager
surface** and an **explicit graph/state seam** underneath.
Subclass :class:`~spectrax.Module`, override ``forward``, call ``model(x)``.
Modules are **JAX pytrees** — :func:`jax.jit`, :func:`jax.tree.map`,
:func:`jax.value_and_grad` accept them directly — and when you want
fine-grained control, :func:`~spectrax.export` still returns the
``(GraphDef, State)`` pair used under the hood.

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 4 4 0 0

   .. grid-item-card:: 🚀 Quickstart
      :link: quickstart
      :link-type: doc

      Install, build a module, run a forward/backward/optimizer step.

   .. grid-item-card:: 🧩 Modules
      :link: guides/modules
      :link-type: doc

      The eager surface: classes, containers, variables, the graph/state seam.

   .. grid-item-card:: ⚡ Transforms
      :link: guides/transforms
      :link-type: doc

      Module-aware ``eval_shape`` / ``jit`` / ``grad`` / ``vmap`` / ``scan`` / ``remat``.

   .. grid-item-card:: 🔎 Selectors
      :link: guides/selectors
      :link-type: doc

      One predicate DSL for every "subset of the model" API.

   .. grid-item-card:: 📡 Dynamic scope
      :link: guides/scope
      :link-type: doc

      Thread context values without wiring every layer signature.

   .. grid-item-card:: 🎯 LoRA fine-tuning
      :link: guides/lora
      :link-type: doc

      Low-rank adapters over the collection system.

   .. grid-item-card:: 🔥 FP8 training
      :link: guides/fp8
      :link-type: doc

      Delayed-scaling E4M3/E5M2 with rolling amax history.

   .. grid-item-card:: 🌐 Sharding
      :link: guides/sharding
      :link-type: doc

      SPMD over ``jax.sharding.Mesh`` via logical axis names.

   .. grid-item-card:: 🔗 Pipeline parallelism
      :link: guides/pipeline
      :link-type: doc

      Stage-parallel training with GPipe / 1F1B / ZB-H1 / Interleaved.

   .. grid-item-card:: 🧠 Design
      :link: design
      :link-type: doc

      Why SpectraX is shaped the way it is.

   .. grid-item-card:: 📈 Performance
      :link: performance
      :link-type: doc

      Dispatch-path optimizations, benchmarks, trade-offs.

   .. grid-item-card:: 📖 API reference
      :link: api_docs/index
      :link-type: doc

      Every public symbol, auto-generated from source docstrings.

   .. grid-item-card:: 📝 Changelog
      :link: changelog
      :link-type: doc

      Release notes.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: User guide

   quickstart
   guides/modules
   guides/transforms
   guides/selectors
   guides/scope
   guides/lora
   guides/fp8
   guides/sharding
   guides/pipeline


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   api_docs/index
   design
   performance
   changelog
