============
Introduction
============

.. seealso::

   This content is adapted from the work of 
   Laurin E. Fischer, Timothée Dao, Ivano Tavernelli, and Francesco Tacchino;
   "*Dual-frame optimization for informationally complete quantum measurements*";
   Phys. Rev. A 109, 062415;
   DOI: https://doi.org/10.1103/PhysRevA.109.062415

------------------------
Generalized measurements
------------------------

The most general class of measurements in quantum mechanics are
described by the POVM formalism. An :math:`n`-outcome POVM is a set of
:math:`n` positive semi-definite Hermitian operators
:math:`\mathbf{M} = \{M_k\}_{k \in \{1, \dots, n \}}` that sum to the
identity. Mathematically, this means the set of operators satisfies the following properties:

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
operators [#d2004informationally]_. Then, for any
observable :math:`\mathcal{O}`, there exist :math:`\omega_k \in \mathbb{R}` such that

.. math::
   :label: observable_povm_decomp

   \mathcal{O}= \sum_{k=1}^{n} \omega_k M_k .

Given such a decomposition of :math:`\mathcal{O}`, the expectation value
:math:`{\langle\mathcal{O}\rangle}_\rho` can be written as

.. math::
   :label: expectation_value_decomp

   {\langle\mathcal{O}\rangle}_\rho = \mathrm{Tr}[\rho O] = \sum_k \omega_k \mathrm{Tr}[\rho M_k] = \mathbb{E}_{k \sim \{p_k\}}[\omega_k].

In other words, :math:`{\langle\mathcal{O}\rangle}_\rho` can be expressed as the mean
value of the random variable :math:`\omega_k` over the probability
distribution :math:`\{p_k\}`. Given a sample of :math:`S` measurement
outcomes :math:`\{ k^{(1)}, \dots, k^{(S)} \}`, we can thus construct an
unbiased Monte-Carlo estimator of :math:`{\langle\mathcal{O}\rangle}_\rho` as

.. math::
   :label: canonical_estimator

   \hat{o} : \{k^{(1)},\dots, k^{(S)}\} \mapsto \frac{1}{S} \sum_{s=1}^{S} \omega_{k^{(s)}}.

------------------
PM-simulable POVMs
------------------

Digital quantum computers typically only give access to projective
measurements (PMs) in a specified computational basis. More general
POVMs can be implemented through additional quantum resources, e.g., by
coupling to a higher-dimensional space in a Naimark
dilation [#gelfand1943imbedding]_ with ancilla
qubits [#chen2007ancilla]_ or
qudits [#fischer_ancilla_free_2022]_ [#stricker2022experimental]_
or through controlled operations with mid-circuit measurements and
classical feed-forward [#ivashkov2023highfidelity]_.
While these techniques have been demonstrated in proof-of-principle
experiments, their full-scale high-fidelity implementation remains a
challenge for current quantum
devices [#fischer_ancilla_free_2022]_. Of particular
interest are thus POVMs that can be implemented without additional
quantum resources, i.e., only through :ref:`projective
measurements in available measurement bases <projective-measurements>`.

More complex POVMs can be built from available projective measurements
through convex combinations of POVMs: For two :math:`n`-outcome POVMs
:math:`\mathbf{M}_1` and :math:`\mathbf{M}_2` acting on the same space, their
convex combination with elements :math:`M_k = p M_{1,k} + (1-p) M_{2,k}`
for some :math:`p \in [0,1]` is also a valid POVM. This can be achieved
in practice by a :ref:`randomization of measurements procedure <randomization>`, which simply
consists of the following two steps for each measurement shot. First,
randomly pick :math:`\mathbf{M}_1` or :math:`\mathbf{M}_2` with probability
:math:`p` or :math:`1-p`, respectively, then perform the measurement
associated with the chosen POVM. We call POVMs that can be achieved by
randomization of projective measurements *PM-simulable*. On digital
quantum computers the easiest basis transformations are single-qubit
transformations of the computational basis. POVMs that consist of
single-qubit PM-simulable POVMs are thus the most readily accessible
class of generalized measurements and have found widespread application.
These include classical shadows and most of their derivatives.

Importantly, PM-simulable informationally-complete POVMs are
overcomplete [#dariano_classical_2005]_. The
decomposition of observables from
Eq. :eq:`expectation_value_decomp` is
thus not unique. In this toolbox, we leverage these additional degrees of
freedom with :ref:`frame theory <frame-theory>`.


.. rubric:: References

.. [#d2004informationally] G. M. d'Ariano, P. Perinotti, M. Sacchi, Journal of
   Optics B: Quantum and Semiclassical Optics 6, S487 (2004).
.. [#gelfand1943imbedding] I. Gelfand, M. Neumark, Matematicheskii Sbornik 12,
   197 (1943).
.. [#chen2007ancilla] P.-X. Chen, J. A. Bergou, S.-Y. Zhu, G.-C. Guo, Physical
   Review A 76, 060303 (2007).
.. [#fischer_ancilla_free_2022] L. E. Fischer, D. Miller, F. Tacchino,, P. K.
   Barkoutsos, D. J. Egger, I. Tavernelli, Phys. Rev. Res. 4, 033027 (2022).
.. [#stricker2022experimental] R. Stricker, M. Meth, L. Postler, C. Edmunds, C.
   Ferrie, R. Blatt, P. Schindler, T. Monz, R. Kueng, M. Ringbauer, PRX Quantum
   3, 040310 (2022).
.. [#ivashkov2023highfidelity] P. Ivashkov, G. Uchehara, L. Jiang, D. S. Wang, A.
   Seif (2023), arXiv:2312.14087.
.. [#dariano_classical_2005] G. M. d'Ariano, P. L. Presti, P. Perinotti, Journal
   of Physics A: Mathematical and General 38, 5979 (2005).
