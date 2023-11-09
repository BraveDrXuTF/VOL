# Variational operator learning: A unified paradigm marrying training neural operators and solving partial differential equations

This repository contains the code for the paper:
- [(VOL) Variational operator learning: A unified paradigm marrying training neural operators and solving partial differential equations](https://arxiv.org/abs/2023.04234)

🔥 The bitmaps in results with improved quailty have just been updated in the latest preprint of our paper. You can also check the paper in our repository.
## HighLights


 - Backpropagation-free residual: Different from physics-informed approaches, VOL does not need to conduct any additional backpropagation operation to calculate residual.
 - Matrix-free manner ($\mathcal{O}(N)$ time complexity for residual calculation & $\mathcal{O}(1)$ space for filters): VOL also does not need to calculate or assemble stiffness matrix like what we do in traditional FEMs pipeline.
 - For all experiments, VOL uses a label-free training set and a 5-label-only shift and follows a power scaling law.
 - Strictly satisfied Dirichlet boundary condition.
 - VOL can be applied to *any field-type neural operator* in principle.
## Cases
![cases](./materials/cases.png)

## Guide to Reproduce the experiment

If you want to reproduce experiments in the paper, please follow the steps below:

 - Comment out the code block under `if __name__ == '__main__':` of the corresponding scripts.
 - Paste specific settings of experiments you want to reproduce. Note you need to specify all file paths again in the code, because these file paths in my code may not exist on your computers. 
 - You need to make sure `trainid` variable matches the physics you want to simulate according to `DatasetL1` class; you need to make sure `Resolution` variable matches the resolution of the data you want to simulate; you need to make sure the trial kernel and the test kernel used in the experiment match the resolution and the physics.
 - Run the script.
 
## Settings of scaling experiments


### Heat transfer with variable heat source 

```python

trainnum_list = [100,200,500,1000,2000,5000,10000]
num_step_list = [2,]
lr_list=[0.01,]
withshift_list = [True,]
num_cycles_list = [0.5,]
num_epoches_list = [50,]
bs_list = [16,] 


experiments_doing_num=1
current_exnum = 0 # 之前已经完成了0个实验
for i in range(20230821,20230821+5):
    for tn in trainnum_list:
        for num_step in num_step_list:
            for lr in lr_list:
                for sft in withshift_list:
                    for num_cycles in num_cycles_list:
                        for bs in bs_list:
                            for num_epoches in num_epoches_list:
                                if experiments_doing_num<=current_exnum:
                                    experiments_doing_num=experiments_doing_num+1
                                    continue
                                    
                                else:
                                    print('experiment doing num: {}'.format(experiments_doing_num))
                                    batch_size = bs
                                    trainnum=tn
                                    testnum=2000
                                    num_training_steps = int(math.ceil(trainnum/batch_size)*num_epoches)
                                    learningrate=lr
                                    weight_decay=1e-3
                                    # num_cycles=0.3
                                    configlist = [
                                        {
                                        "momentum": 0,
                                        "lr": learningrate,
                                        "optimizer": "adamw",
                                        "scheduler": 'cosine_with_warm',
                                        "scheduler_config": {"warm_steps":num_training_steps//10,
                                        "num_training_steps": num_training_steps,
                                        "num_cycles":num_cycles},
                                        "multimodels":True,
                                        "num_epoches":num_epoches,
                                        "residual_loss_type_list":['gongEeTiDuWeak']*num_epoches,
                                        "batch_size": bs,
                                        "traindata": trainnum,
                                        "testdata": testnum,
                                        "trainseed": i,
                                        "regularL2":False,
                                        "test_function": True,
                                        "weight_decay": weight_decay,
                                        # "val_mode": True,
                                        "shift": sft,
                                        "num_step": num_step,
                                        "experiment_name": "ParaQPARALLELS_withgrid/bs{0}train{1}test{2}seed{3}lr{4}ep{5}warm5percentdecay{6}cycle{7}datadype{8}_numstep{9}_shift{10}_withgrid".format(bs,trainnum,testnum,i,learningrate,num_epoches,weight_decay,num_cycles,datatype,num_step,sft),
                                        "ifgrid":True
                                        } 
                                        
                                        ]
                                    print('Starting: ')
                                    print(configlist[0])
                                    trainOneParameter(config=configlist[0])
                                    experiments_doing_num=experiments_doing_num+1

```

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
