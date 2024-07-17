.. _projective-measurements:

=======================
Projective measurements
=======================

.. seealso::

   This content is adapted from the work of Timothée Dao; "*Informationally
   Complete Generalized Measurements for Estimating Quantum Expectation Values*"
   [Unpublished master's thesis]; ETH Zürich (2023).

---------------------------------
Measurement in an arbitrary basis
---------------------------------

Consider a :math:`d`-dimensional quantum system with a computational basis
:math:`\{ \ket{k}\}_{k=1}^d`. The corresponding projective measurement
in this basis is described by :math:`\mathbf{P} = \{\ketbra{k}{k}\}_{k=1}^d`.
Suppose we want to perform a projective measurement in another orthonormal basis
:math:`\{ \ket{\psi_k}\}_{k=1}^d`. As a unitary transformation is
equivalent to a change of basis, there exists a unitary :math:`U` such
that :math:`\ket{\psi_k} = U \ket{k}` for all :math:`k=1,2,\dots,d`.
This implies that the new projective measurement is described by the PVM
:math:`\{\ketbra{\psi_k}{\psi_k}\}_{k=1}^d = \{U \ketbra{k}{k} U^\dagger\}_{k=1}^d`.
The probability of obtaining the outcome :math:`k` is then given by

.. math::
   :label: probability

   p_k = \mathrm{Tr}[\ketbra{\psi_k}{\psi_k} \rho] = \mathrm{Tr}[U \ketbra{k}{k}
   U^\dagger \rho] = \mathrm{Tr}[\ketbra{k}{k} U^\dagger \rho U] \, ,

where we used the invariance of the trace under cyclic permutations in
the last equality. It becomes now clear that the two procedures
described below are equivalent:

.. container:: center

   +----------------------------------+-------------------------------+
   | Procedure 1A                     | Procedure 1B                  |
   +==================================+===============================+
   | 1. Prepare state :math:`\rho`    | 1. Prepare state :math:`\rho` |
   +----------------------------------+-------------------------------+
   | 2. Measure in the basis          | 2. Let the state evolve as    |
   | :math:`\{ \ket{\psi_k}\}_{k}     | :math:`\rho                   |
   | =\{ U \ket{k}\}_{k}`             | \mapsto U^\dagger \rho U`     |
   +----------------------------------+-------------------------------+
   |                                  | 3. Measure in the             |
   |                                  | computational basis           |
   +----------------------------------+-------------------------------+


This equivalence is relevant in many practical situations. In
experiments, one can often only perform measurements in a single, fixed
(computational) basis but can apply various unitary transformations to
the state before the measurement. Therefore, through this equivalence,
one can emulate other projective measurements.


**Example**:
   Consider a qubit system and suppose we only have an apparatus
   performing measurements in the computational basis,
   :math:`\mathbf{M}_Z = \{Z_+ , Z_-\} = \{ \ketbra{0}{0} , \ketbra{1}{1}\}`. We
   can still perform an :math:`X` measurement,
   :math:`\mathbf{M}_X = \{X_+ , X_-\} = \{\ketbra{+}{+} , \ketbra{-}{-}\}`, by
   applying the Hadamard transformation

   .. math::
      :label: hadamard

      H = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}

   to the state and then performing a measurement in the computational
   basis. Indeed, we have :math:`\ket{+} = H \ket{0}` and
   :math:`\ket{-} = H \ket{1}`.

.. _pm-simulable:

-------------------------
PM-simulable measurements
-------------------------

We can extend the Procedures 1A and 1B to PM-simulable POVMs, which can
always be achieved by a :ref:`randomization technique <randomization>`. Suppose we want to perform the
measurement associated with the POVM
:math:`\mathbf{M} = \biguplus_i q_i \mathbf{P}_i = \{q_i \ketbra{\psi^{i}_k}{\psi^{i}_k} \}_{(i,k)}`,
where :math:`\{q_i\}_i` is a probability distribution and
:math:`\mathbf{P}_i = \{\ketbra{\psi^{i}_k}{\psi^{i}_k}\}_k` are rank-1 PVMs. The
outcomes are labeled by the pair :math:`(i,k)`. Let :math:`\{U_i\}_i` be
the set of unitary operators such that
:math:`\ket{\psi^{i}_k} = U_i \ket{k}` for all :math:`k,i`. Then, the
two procedures described below are equivalent:

.. container:: center

   +----------------------------------+-------------------------------+
   | Procedure 2A                     | Procedure 2B                  |
   +==================================+===============================+
   | 1. Prepare state                 | 1. Prepare state :math:`\rho` |
   | :math:`\rho`                     |                               |
   +----------------------------------+-------------------------------+
   | 2. Randomly pick :math:`i`       | 2. Randomly pick :math:`i`    |
   | with probability :math:`q_i`     | with probability :math:`q_i`  |
   +----------------------------------+-------------------------------+
   | 3. Measure in the basis          | 3. Let the state evolve as    |
   | :math:`\{                        | :math:`\rho \mapsto           |
   | \ket{\psi^{i}_k}\}_{k} =\{       | U_{i}^\dagger \rho U_{i}`     |
   | U_{i} \ket{k}\}_{k}`             |                               |
   +----------------------------------+-------------------------------+
   |                                  | 4. Measure in the             |
   |                                  | computational basis           |
   +----------------------------------+-------------------------------+

Note that usually, not all unitary operations are achievable in
practice. Therefore, instead of starting from the POVM we would ideally
like to perform, we usually first define the set of achievable unitary
operations :math:`\mathcal{U}=\{U_i\}_i`. We then determine the
corresponding set of achievable PVMs :math:`\mathcal{S}=\{\mathbf{P}_i\}_i`,
where
:math:`\mathbf{P}_i = \{U_i \ket{k} \bra{k} U_i^\dagger\}_k, \forall i`.
Finally, we choose the POVM to be performed from the convex hull
:math:`\mathcal{S}^\mathrm{conv}`.
