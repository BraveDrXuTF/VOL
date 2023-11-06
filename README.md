# Variational operator learning: A unified paradigm marrying training neural operators and solving partial differential equations

This repository contains the code for the paper:
- [(VOL) Variational operator learning: A unified paradigm marrying training neural operators and solving partial differential equations](https://arxiv.org/abs/2023.04234)

## HighLights

![cases](./materials/cases.png)
 - Backpropagation-free residual: Different from physics-informed approaches, VOL does not need to conduct any additional backpropagation operation to calculate residual.
 - Matrix-free manner ($\mathcal{O}(N)$ time complexity for residual calculation & $\mathcal{O}(1)$ space for filters): VOL also does not need to calculate or assemble stiffness matrix like what we do in traditional FEMs pipeline.
 - For all experiments, VOL uses a label-free training set and a 5-label-only shift and follows a power law.
 - Strictly satisfied Dirichlet boundary condition.
 - VOL can be applied to any field-type neural operator in principle.


## Guide to Reproduce the experiment
If you want to reproduce experiments in the paper, please follow the steps below:

 - Comment out the code block under `if __name__ == '__main__':` of the corresponding scripts.
 - Paste specific settings of experiments you want to reproduce. Note you need to specify all file paths again in the code, because these file paths in my code may not exist on your computers. 
 - You need to make sure `trainid` variable matches the physics you want to simulate according to `DatasetL1` class; you need to make sure `Resolution` variable matches the resolution of the data you want to simulate; you need to make sure the trial kernel and the test kernel used in the experiment match the resolution and the physics.
 - Run the script.
 

## My goal

In the long term, I want to design **parallel, real-time, low-carbon, reliable, data-efficient neural solvers** for the next generation simulation and CAE software. 


## Citations

```
@misc{xu2023variational,
      title={Variational operator learning: A unified paradigm marrying training neural operators and solving partial differential equations}, 
      author={Tengfei Xu and Dachuan Liu and Peng Hao and Bo Wang},
      year={2023},
      eprint={2304.04234},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements

The implementation of VOL is benefited greatly from remarkable projects from the community. We would like to sincerely thank [FNO](https://github.com/neuraloperator/neuraloperator/tree/master), [F-FNO](https://github.com/alasdairtran/fourierflow), and [PyTorch](https://github.com/pytorch/pytorch) for their awesome open source.
