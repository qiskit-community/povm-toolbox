============
Introduction
============

------------------------
Generalized measurements
------------------------
..
   Define non-breaking space
.. |_| unicode:: 0xA0 
   :trim:

The most general class of measurements in quantum mechanics are
described by the POVM formalism. An :math:`n`-outcome POVM is a set of
:math:`n` positive semi-definite Hermitian operators
:math:`\mathbf{M} = \{M_k\}_{k \in \{1, \dots, n \}}` that sum to the
identity. Mathematically, this means the set of operators satisfies the following proeperties:

   #. :math:`M_k^\dagger = M_k` for all :math:`k \in \{1,2 \dots, n\}`,
   #. :math:`\langle \psi | M_k | \psi \rangle \geq 0` for all :math:`k \in \{1,2 \dots, n\}` and all states :math:`|\psi \rangle`,
   #. :math:`\sum_{k=1}^n M_k = \mathbb{I}`.

Given a
:math:`d`-dimensional state :math:`\rho`, the probability of observing
outcome :math:`k` is given by Born’s rule as
:math:`p_k = \mathrm{Tr}[\rho M_k]`. Standard projective measurements (PMs) are
a special case of POVMs, where each POVM operator is a projector such
that :math:`M_k = \ketbra{\phi_k}{\phi_k}` for some pure states
:math:`\phi_k`. A POVM is said to be *informationally complete* (IC) if
it spans the space of Hermitian
operators |_| [d2004informationally]_. Then, for any
observable :math:`\mathcal{O}`, there exist :math:`\omega_k \in \mathbb{R}` such that

.. math::

   \label{eqn:observable_povm_decomp}
  \mathcal{O}= \sum_{k=1}^{n} \omega_k M_k .

Given such a decomposition of :math:`\mathcal{O}`, the expectation value
:math:`{\langle\mathcal{O}\rangle}_\rho` can be written as

.. math::

   \label{eqn:expectation_value_decomp}
   {\langle\mathcal{O}\rangle}_\rho = \mathrm{Tr}[\rho O] = \sum_k \omega_k \mathrm{Tr}[\rho M_k] = \mathbb{E}_{k \sim \{p_k\}}[\omega_k].

In other words, :math:`{\langle\mathcal{O}\rangle}_\rho` can be expressed as the mean
value of the random variable :math:`\omega_k` over the probability
distribution :math:`\{p_k\}`. Given a sample of :math:`S` measurement
outcomes :math:`\{ k^{(1)}, \dots, k^{(S)} \}`, we can thus construct an
unbiased Monte-Carlo estimator of :math:`{\langle\mathcal{O}\rangle}_\rho` as

.. math::

   \label{eqn:canonical_estimator}
       \hat{o} : \{k^{(1)},\dots, k^{(S)}\} \mapsto \frac{1}{S} \sum_{s=1}^{S} \omega_{k^{(s)}}.

.. _`sec:PM-simulabel_POVMs`:

------------------
PM-simulable POVMs
------------------

Digital quantum computers typically only give access to projective
measurements (PMs) in a specified computational basis. More general
POVMs can be implemented through additional quantum resources, e.g., by
coupling to a higher-dimensional space in a Naimark
dilation |_| [gelfand1943imbedding]_ with ancilla
qubits |_| [chen2007ancilla]_ or
qudits |_| [fischer_ancilla_free_2022]_, |_| [stricker2022experimental]_
or through controlled operations with mid-circuit measurements and
classical feed-forward |_| [ivashkov2023highfidelity]_.
While these techniques have been demonstrated in proof-of-principle
experiments, their full-scale high-fidelity implementation remains a
challenge for current quantum
devices |_| [fischer_ancilla_free_2022]_. Of particular
interest are thus POVMs that can be implemented without additional
quantum resources, i.e., only through projective measurements in
available measurement bases.

More complex POVMs can be built from available projective measurements
through convex combinations of POVMs: For two :math:`n`-outcome POVMs
:math:`\mathbf{M}_1` and :math:`\mathbf{M}_2` acting on the same space, their
convex combination with elements :math:`M_k = p M_{1,k} + (1-p) M_{2,k}`
for some :math:`p \in [0,1]` is also a valid POVM. This can be achieved
in practice by a *randomization of measurements* procedure, which simply
consists of the following two steps for each measurement shot. First,
randomly pick :math:`\mathbf{M}_1` or :math:`\mathbf{M}_2` with probability
:math:`p` or :math:`1-p`, respectively, then perform the measurement
associated with the chosen POVM. We call POVMs that can be achieved by
randomizations of projective measurements *PM-simulable*. On digital
quantum computers the easiest basis transformations are single-qubit
transformations of the computational basis. POVMs that consist of
single-qubit PM-simulable POVMs are thus the most readily accessible
class of generalized measurements and have found widespread application.
These include classical shadows and most of their derivatives, see
Appendix |_| .

Importantly, PM-simulable informationally-complete POVMs are
overcomplete |_| [dariano_classical_2005]_. The
decomposition of observables from
Eq. |_| blabla is
thus not unique. In this work, we leverage these additional degrees of
freedom to build better observable estimators, see
Fig.

.. figure:: overview_schematic.*
   :width: 50.0%

   Schematic of dual frame optimization. Generalized measurements are
   performed on the quantum system. Upon obtaining outcome :math:`k`,
   the corresponding canonical dual operator :math:`D_k` – also known as
   *classical shadow* – can be efficiently computed and stored on a
   classical computer. The expectation value of any observable :math:`\mathcal{O}`
   can be estimated from a sample of dual operators. Leveraging
   additional degrees of freedom, we can optimize these dual operators
   through classical post-processing, effectively reducing the
   estimation variance.



.. [d2004informationally] d2004informationally.
.. [gelfand1943imbedding] gelfand1943imbedding.
.. [chen2007ancilla] chen2007ancilla.
.. [fischer_ancilla_free_2022] fischer_ancilla_free_2022.
.. [stricker2022experimental] stricker2022experimental.
.. [ivashkov2023highfidelity] ivashkov2023highfidelity.
.. [dariano_classical_2005] dariano_classical_2005.