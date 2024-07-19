.. _frame-theory:

===========================
Frame theory and dual space
===========================

.. seealso::

   This content is adapted from the work of 
   Laurin E. Fischer, Timothée Dao, Ivano Tavernelli, and Francesco Tacchino;
   "*Dual-frame optimization for informationally complete quantum measurements*";
   Phys. Rev. A 109, 062415;
   DOI: https://doi.org/10.1103/PhysRevA.109.062415

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
state [#huang_predicting_2020]_ (see :ref:`section <classical-shadows>` below for details), thus providing a direct
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


.. _classical-shadows:

-----------------------------
Relation to classical shadows
-----------------------------



We now show the explicit connection to the technique of classical shadows 
[#huang_predicting_2020]_. The technique consists of rotating the state :math:`\rho` by a
unitary :math:`U_i`, sampled from a set :math:`\mathcal{U}`, and then
performing a measurement in the computational basis. We show in the section :ref:`pm-simulable` that
this protocol is equivalent to performing the PM-simulable POVM
:math:`\mathbf{M} = \biguplus_i q_i \mathbf{P}_i = \{q_i P_{i,k}\}_{(i,k)}`,
where :math:`P_{i,k} = U_i^\dagger \ketbra{k} U_i` and the outcomes are labeled by
:math:`(i,k)`. It now appears that the measurement channel

.. math::

    \mathcal{M} : \rho \mapsto \mathbb{E}_{i \sim \{q_i\}} \sum_k \mathrm{Tr}[\rho P_{i,k}] P_{i,k}
    = \sum_{i,k} \frac{\mathrm{Tr}[\rho M_{i,k}]}{\mathrm{Tr}[M_{i,k}]} M_{i,k}  \, ,
    \qquad  M_{i,k} = q_i P_{i,k} \, , 

is actually an :math:`\alpha`-frame superoperator :math:`\mathcal{F}_{\alpha}`
associated with the POVM :math:`\mathbf{M}`, where the coefficients are
given by :math:`\alpha_{i,k} = 1/\mathrm{Tr}[M_{i,k}] = 1/q_i` for all
:math:`i,k`. Most importantly, the elements of the dual frame given by
this :math:`\alpha`-parametrization are the classical shadows:

.. math::

   \hat{\rho}_{i,k} = \mathcal{M}^{-1}(P_{i,k}) = \frac{1}{q_i} \mathcal{M}^{-1}(M_{i,k}) = \alpha_{i,k} \mathcal{F}_{\alpha}^{-1}(M_{i,k}) = D_{i,k}  \, . 

In other words, the classical shadows technique
consists of performing a PM-simulable POVM and choosing a specific
dual frame. However, nothing prevents us from choosing another dual
frame. Any dual frame defines an unbiased estimator
of the state. More precisely, for any dual frame :math:`\mathbf{D} = \{D_{i,k}\}` and any state
:math:`\rho`, we have

.. math::

   \rho = \sum_{i,k} \mathrm{Tr}[ \rho M_{i,k}] D_{i,k} = \mathbb{E}_{i,k}[D_{i,k}] \, ,

which follows from the reciprocity of duality. That is, if :math:`\mathbf{D}` is a dual frame
to :math:`\mathbf{M}`, then :math:`\mathbf{M}` is a dual frame to :math:`\mathbf{D}` 
[#casazza2013finite_frame]_.


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
.. [#casazza2013finite_frame] P. G. Casazza, G Kutyniok and F Philipp, 
   *Finite frames: theory and applications*, Birkhäuser, Boston, (2013).
