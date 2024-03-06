# Classical Shadow samplers

The Classical Shadow samplers define a first use case of the new POVM samplers.

## Getting an estimate

By default, the estimator uses canonnical dual operators to estimate the expecatation value.

```python
povm_implementation = ClassicalShadowImplementation(n_qubits=n_qubits)

sampler = POVMSampler(backend, shot, seed)
job = sampler.run(povm_implementation: POVMImplementation,
                  circuits: Sequence[QuantumCircuit])
result = job.result()

estimator = POVMPostProcessor(results)
estimate = estimator.get_expectation_value(observable, method="canonical")
```

The user might want to use different dual operators for the estimation, such as the optimization-free "emprirical frequency dual operators".

```python
estimate = estimator.get_expectation_value(observable, method="empirical_frequencies")
```

# PM-simulable POVM samplers

  

The `PVMSimulablePOVM` samplers define a first use case of the new POVM samplers.

  

## Getting an estimate

  

By default, the estimator uses canonnical dual operators to estimate the expecatation value.

```python
n_qubit = n_qubit
n_PVM = n_PVM
parameters = parameters_POVM
```

How does the user specify the POVM ?
- By a parametrization with minimal but abstract parameters ? e.g., an `np.ndarray` of shape `(n_qubits, 3*n_PVM-1)`
- By the a set of PVMs and some associated weights ? But then, how does the user specify the PVMs ?
- By qubit rotation gates ? I.e. quantum circuit

```python
povm_implementation = PMSimImplementation(parameters = parameters, n_qubit=n_qubit)
sampler = POVMSampler(backend, shot, seed)
job = sampler.run(povm_implementation, circuits)
result = job.result()

estimator = POVMPostProcessor(results)
estimate = estimator.get_expectation_value(observable, method="optimized")
```


## How do the classes look like 

```python
class POVMImplementation:
	
	def __init__(
		self,
		n_qubit : int 
	) -> None :
		
		self.n_qubit = n_qubit

	def measurement_qc(self) -> QuantumCircuit :
		raise NotImplementedError("The subclass of POVMImplementation must implement `_build_from_param` method.")

	def get_parameterValues_and_shot(self, shot:int) -> QuantumCircuit :
		raise NotImplementedError("The subclass of POVMImplementation must implement `get_parameter_and_shot` method.")


```


```python
class PVMSimPOVMImplementation(POVMImplementation):
	
	def __init__(
		self,
		n_qubit : int,
		parameters : np.ndarray | None = None,
	) -> None :
		
		super().__init__(n_qubit)
		self.set_parameters(parameters)


	def measurement_qc(self) -> QuantumCircuit :
		
		theta = ParameterVector('theta',length=n_qubit)
		phi = ParameterVector('phi', length=n_qubit)

		qc = QuantumCircuit(self.n_qubit)
		for i in range(self.n_qubit):
			qc.u(theta=theta[i], phi=phi[i], lam=0 , qubit=i)

		return qc	


	def get_parameterValues_and_shot(self, shot:int) -> QuantumCircuit :
        """
        Returns a list with concrete parameter values and associated number of shots.
        """

		PVM_idx = np.zeros((shot,n_qubit), dtype=int)

		for i in range(n_qubit):
			PVM_idx[:,i] = np.random.choice(self.n_PVM, size=int(shot), replace=True, p=self.PVM_distribution[i])
		counts = Counter(tuple(x) for x in PVM_idx)

		return [tuple(([self.PVM_angles[i,c[i]] for i in range(n_qubit)], counts[c])) for c in counts]

	def set_parameters(self, parameters: np.ndarray):
		"""Set parameters"""
		self.PVM_angles = ...
		self.PVM_distribution = ... 
```




wait... how to include to randomization part ? It's not always the same measurement circuit that has to be performed...
We could use a fixed schedule but not 100% sure it works...




```python
class POVMSampler(BaseSampler):
	
	def __init__(
		self,
		povm_implementation : POVMImplementation | None = None,
		backend : Backend,
		shot : int,
		seed : int | None = None 
	) -> None :

		super().__init__(backend=backend, shot=shot, seed=seed)
		self.povm_implementation = povm_implementation



	def run(
		self,
		circuits: QuantumCircuit | Sequence[QuantumCircuit],
		parameter_values: Sequence[float] | Sequence[Sequence[float]] | None = None,
		**run_options: Any,
	) -> PovmSamplerJob :

		circuits.append(self.povm_implementation.parametrized_qc)

		job_list = []
		
		for msmt_parameter, msmt_shot in self.povm_implementation.get_parameter_and_shot(self.shot):
			job = super().run(cicuits, concatenate(parameter_values,msmt_parameter), msmt_shot)
			job_list.append((job, PVM_idx))
		
		return PovmSamplerJob(job_list, self.povm_implementation)
```


```python
class POVMSamplerJob:
	
	def __init__(
		self,
		job_list : Sequence[tuple(BaseJob, int)],
		povm_implementation : POVMImplementation
	) -> None :
		self.job_list = job_list
		self.povm_implementation = povm_implementation

	def result(self)  :
		"""Transform the tuple (outcome, PVM_used) into an outcome k in [0,1,2,...,n-1] with n = 2*n_PVM"""
		return something
```


```python
class POVMPostProcessor:

	def __init__(
		self,
		povm_sample: POVMResult
	) -> None :
		self.count = povm_result.result.get_count()
		self.povm = BasePOVM.from_implementation(povm_result.povm_implementation)

```

## Optimizing the dual operators

The user might want to optimize the dual operators with respect to given observable or set of observables, where the cost function is evaluated using the data sampled.

```python
estimate1 = estimator.get_expectation_value(observable1, method="optimized") # optimization is performed w.r.t. observable1, optimized duals not necessarily stored
estimate2 = estimator.get_expectation_value(observable2, method="optimized") # optimization is performed w.r.t. observable2
```

Would it be better to have a syntax like this :

```python
estimator.optimize_duals(observables:List[Operator] = [obseravble1, observable2]) # where the optimization is performed and optimized duals are the stored
estimate1 = estimator.get_expectation_value(observable1, method="optimized") # uses the (already computed) opitmized duals
estimate2 = estimator.get_expectation_value(observable2, method="optimized") # uses the (already computed) opitmized duals
```

For check/comparison purposes, would it be interesting to offer the possibility to feed in the true state to compute the optimal dual operators ?

```python
estimator.set_optimal_duals(state) # where the optimization is performed and optimized duals are the stored
optimal_estimate1 = estimator.get_expectation_value(observable1, method="optimal") # uses the (already computed) opitmized duals
optimal_estimate2 = estimator.get_expectation_value(observable2, method="optimal") # uses the (already computed) opitmized duals
```
