# DEPLM
Code for "Random resistive memory-based extreme point learning machine for unified visual processing."

## To release soon

## Abstract

Visual sensors, including 3D LiDAR, neuromorphic DVS sensors, and conventional frame cameras, are increasingly integrated
into edge-side intelligent machines. Realizing intensive multi-sensory data analysis directly on edge intelligent machines
is crucial for numerous emerging edge applications, such as augmented and virtual reality and unmanned aerial vehicles,
which necessitates unified data representation, unprecedented hardware energy efficiency and rapid model training. However,
multi-sensory data are intrinsically heterogeneous, causing significant complexity in the system development for edge-side
intelligent machines. In addition, the performance of conventional digital hardware is limited by the physically separated
processing and memory units, known as the von Neumann bottleneck, and the physical limit of transistor scaling, which
contributes to the slowdown of Moore’s law. These limitations are further intensified by the tedious training of models with
ever-increasing sizes. In this study, we propose a novel hardware-software co-design, random resistive memory-based deep
extreme point learning machine (DEPLM), that offers efficient unified point set analysis. Data-wise, the multi-sensory data are
unified as point sets and can be processed universally. Software-wise, most weights of deep extreme point learning machines
are exempted from training, which significantly reduce training complexity. Hardware-wise, nanoscale resistive memory not
only enables collocation of memory and processing, mitigating the von Neumann bottleneck and the slowdown of Moore’s law,
but also leverages the inherent programming stochasticity for generating the random and sparse weights of the DEPLM, which
also lessens the impact of read noise. We demonstrate the system’s versatility across various data modalities and two different
learning tasks. Compared to a conventional digital hardware-based system, our co-design system achieves 5.90×, 21.04×, and
15.79× energy efficiency improvements on ShapeNet 3D segmentation, DVS128 Gesture event-based gesture recognition, and
Fashion-MNIST image classification tasks, respectively, while achieving 70.12%, 89.46%, and 85.61% training cost reduction
when compared to conventional systems. Our random resistive memory-based deep extreme point learning machine may pave
the way for energy-efficient and training-friendly edge AI across various data modalities and tasks.
