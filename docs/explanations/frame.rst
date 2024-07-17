.. _frame-theory:

===========================
Frame theory and dual space
===========================

.. seealso::

   This content is adapted from the work of L. E. Fischer, T. Dao, I. Tavernelli,
   and F. Tacchino, `Dual-frame optimization for informationally complete
   quantum measurements <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.109.062415>`_, Phys. Rev. A 109 (2024).

---------------
POVMs as frames
---------------

As a reminder, given an IC-POVM :math:`\mathbf{M} = \{M_k\}_{k \in \{1, \dots, n \}}` and an
observable :math:`\mathcal{O}`, there exist :math:`\omega_k \in \mathbb{R}` such that

.. math::
   :label: observable_povm_decomposition

   \mathcal{O} = \sum_{k=1}^{n} \omega_k M_k .

We will now outline a formal approach to obtain the coefficients
:math:`\omega_k` in Eq. :eq:`observable_povm_decomposition` for
a given observable :math:`\mathcal{O}`. First, we note that the minimal number of
linearly independent POVM elements for an IC-POVM is :math:`n = d^2`. We
call such POVMs *minimally informationally complete*.
In that case, the coefficients :math:`\omega_k` are unique. However, for
POVMs with :math:`n > d^2`, such as those that arise from IC PM-simulable
POVMs, the decomposition in
Eq. :eq:`observable_povm_decomposition` is not unique. This
redundancy is described by frame theory, as outlined in
Ref. [#innocenti2023shadow]_. 

Simply speaking, a *frame* is a generalization
of the notion of the basis of a vector space, where the basis elements may
be linearly dependent. The set of POVM operators
:math:`\mathbf{M} = \{M_k\}_{k \in \{1, \dots, n \}}` forms a frame for the
space of Hermitian operators if and only if it is IC. For any frame,
there exists at least one dual frame
:math:`\mathbf{D} = \{D_k\}_{k \in \{1, \dots, n \}}`, such that

.. math::
   :label: eqn:definition_duals  
   
   \mathcal{O} = \sum_{k=1}^n \mathrm{Tr}[\mathcal{O} D_k] M_k

for any Hermitian operator :math:`\mathcal{O}`. Therefore, the
coefficients :math:`\omega_k` can simply be obtained from the duals
:math:`\mathbf{D}` as 

.. math::
   :label: coeffs_from_duals

   \omega_k = \mathrm{Tr}[\mathcal{O} D_k].

Notably, dual operators generalize the concept of
classical shadows of a quantum
state [#huang_predicting_2020]_ (see below for details), thus providing a direct
connection to the popular randomized measurement
toolbox [#elben2022randomized]_.

------------------------
Constructing dual frames
------------------------

For a minimally IC POVM, only one dual frame exists. It can be
constructed from the POVM elements as

.. math::
   :label: def_canonical_duals

   \left| D_k \right\rangle\kern-3mu\rangle = \mathcal{F}^{-1} \left| M_k \right\rangle\kern-3mu\rangle \, , \quad k =1,2,\dots,n

with the *canonical frame superoperator*

.. math::
   :label: def_frame_superop

   \mathcal{F} = \sum_{k=1}^n \left| M_k \right\rangle\kern-3mu\rangle\kern-5mu\left\langle\kern-3mu\langle M_k \right|,

where we have used the widespread vectorized
'double-ket' notation. Thus, the frame
superoperator can be used to transform between the POVM space and the
dual space.

For an overcomplete POVM, the canonical frame superoperator creates
one of infinitely many possible dual frames. Other valid dual frames can
be obtained through a parametrized frame superoperator as follows:

.. math::
   :label: def_alpha_duals

   \left| D_k \right\rangle\kern-3mu\rangle = \alpha_k \mathcal{F}^{-1}_{\alpha} \left| M_k \right\rangle\kern-3mu\rangle \, ,
   \quad \quad \text{with } \mathcal{F}_{\alpha} = \sum_{k=1}^n \alpha_k \left| M_k \right\rangle\kern-3mu\rangle\kern-5mu\left\langle\kern-3mu\langle M_k \right|,

for real parameters :math:`\{\alpha_k\}_k \subset \mathbb{R}` such that :math:`\mathcal{F}_{\alpha}` in invertible [#fischer_dual_frame_2023]_.

-----------------------------
Relation to classical shadows
-----------------------------

TODO

.. rubric:: References

.. [#innocenti2023shadow] L. Innocenti, S. Lorenzo, I. Palmisano, F. Albarelli,
   A. Ferraro, M. Paternostro, and G. M. Palma, PRX
   Quantum 4, 040328 (2023).
.. [#huang_predicting_2020] H.-Y. Huang, R. Kueng, and J. Preskill, Nature Physics
   16, 1050 (2020).
.. [#elben2022randomized] A. Elben, S. T. Flammia, H.-Y. Huang, R. Kueng,
   J. Preskill, B. Vermersch, and P. Zoller, Nature Reviews
   Physics 5, 9 (2022).
.. [#fischer_dual_frame_2023] L. E. Fischer, T. Dao, I. Tavernelli,
   and F. Tacchino, *Dual-frame optimization for informationally complete
   quantum measurements*, Phys. Rev. A 109 (2024).