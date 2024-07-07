# MixUp-and-CutOut-Data-Augumentation-Techniques

<img src="https://github.com/leonard-sanya/MixUp-and-CutOut-Data-Augumentation-Techniques/blob/main/data_augmementation.png" width="720" height="480"/>

## Description

We implemented three additional data augmentation techniques: CutOut, MixUp, and CutMix to assess their impact on model classification accuracy.

- **CutOut:** This CNN regularization method involves randomly masking out regions of input images, promoting robust feature learning by the network.

- **MixUp:** Generates virtual training examples by linearly interpolating between pairs of examples and their labels. Mathematically, this can be represented as:
  
$$ \hat{x} = \lambda x_i + (1-\lambda)x_j$$

   $$ \hat{y} = \lambda y_i + (1-\lambda)y_j$$

where $x_i,x_j$ are raw inputs vectors $y_i,y_j $ are one-hot label encodings and $(x_i,y_i),(x_j,y_j)$ are two randomly sampled examples from the training set. $\lambda$ is a hyperameter between 0 and 1.

- **CutMix:** Combines pairs of training samples by cutting and pasting patches, with ground truth labels mixed proportionally to the area of patches. Given samples $(x_a, y_a)$ and $(x_b, y_b)$, CutMix generates a new training sample $(\hat{x}, \hat{y})$, which is utilized for model training. The CutMix operation is defined as


   $$\hat{x} = M \odot x_a + (1- M)\odot x_b$$
  
    $$\hat{y} = \lambda y_a + (1-\lambda)y_b$$

## Installation
- Inorder to run this implementation, clone the repository https:

       https://github.com/leonard-sanya/MixUp-and-CutOut-Data-Augumentation-Techniques.git   
      
- Run the Mixup and Cutout data augmentation techniques on CIFAR 10 and CIFAR100 using the AlexNet backbone by using the following command:

      python main.py

## Results
- The suummary table below shows model evaluation when trained using the Cutout, Cutmix and Mixup, we notice a significance change in the Top-1 error rate when using the Mixup data augmentation.

<img src="https://github.com/leonard-sanya/MixUp-and-CutOut-Data-Augumentation-Techniques/blob/main/results.png" width="1024" height="150"/>
	
## License

This project is licensed under the [MIT License](LICENSE.md). Please read the License file for more information.

Feel free to explore each lab folder for detailed implementations, code examples, and any additional resources provided. Reach out to me via [email](lsanya@aimsammi.org) in case of any question or sharing of ideas and opportunities

