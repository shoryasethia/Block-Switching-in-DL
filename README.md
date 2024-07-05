# Block-Switching-in-DL
## **Paper : [2002.07920v1](https://github.com/shoryasethia/Block-Switching-in-DL/blob/main/2002.07920v1.pdf)**
Switching block in consists of multiple channels. Each regular model is split into a lower part, containing all convolutional layer. lower parts are again combined to form single output providing parallel channels of block switching while the other parts are discarded. These models tend to have similar characteristics in terms of classification accuracy and robustness, yet different model parameters due to random initialization and stochasticity in the training process.
## Architecture
![Architecture](https://github.com/shoryasethia/Block-Switching-in-DL/blob/main/BlockSwitching.png)

## Phase 1: Training Sub-Models

### Initial Training
Started by training `4` instances of the `modified VGG16` separately. Each instance begins with different random weight initializations. These models are based on VGG16 architecture but have been adapted for CIFAR-10 by adding some more layers.

- Modified VGG16 with customized layers for CIFAR-10.
- Each model instance trained independently with variations due to random initialization and stochastic training.

## Phase 2: Creating the Switching Block

### Splitting the Models
After training, each modified VGG16 model is split into two parts:
- **Lower Part**: Includes the initial convolutional layers and feature extraction components.
- **Upper Part**: Comprises the fully connected layers and classification head.
- **Discard Upper Parts**: Remove the upper parts of all trained modified VGG16 models.

### Forming Parallel Channels
- **Grouping Lower Parts**: The lower parts (initial convolutional layers) of these trained models are grouped together to form parallel channels.
- **Base of Switching Block**: These parallel channels serve as the base of the switching block.

### Connecting the Switching Block
- **Adding Common Upper Model**: Introduce a new, randomly initialized common upper model.
- **Switching Mechanism**: Connect the parallel channels (lower parts) to the common upper model. At runtime, only one parallel channel is active for processing any given input, introducing stochastic behavior.

## Phase 3: Fine-Tuning the Combined Model
### Retraining
- **Combined Model Setup**: The switching block (parallel channels + common upper model) is retrained on the CIFAR-10 dataset.
- **Accelerated Training**: Retraining is faster since the lower parts (parallel channels) are pre-trained.
- **Adaptation Learning**: The common upper model learns to adapt to inputs from various parallel channels, ensuring robust classification regardless of the active channel.

# Results
| Model | Accuracy on 10K test images of CIFAR10 | Saved Model |
|-------|----------------------------------------|-------------|
|`vgg16_cifar10_sub_model_0`|0.7327|[vgg16_cifar10_sub_model_0.h5](https://drive.google.com/file/d/1uUH6m9EQnhNMII6QyvZLR1ygPFywKn2T/view?usp=sharing)|
|`vgg16_cifar10_sub_model_1`|0.6914|[vgg16_cifar10_sub_model_1.h5](https://drive.google.com/file/d/1NJloU6zHerW-e6Gyh5D4CnCatJh0cZpf/view?usp=drive_link)|
|`vgg16_cifar10_sub_model_2`|0.7165|[vgg16_cifar10_sub_model_2.h5](https://drive.google.com/file/d/1yp3VGRutPGw9HaR1oiNOmc6QuGoXwWRO/view?usp=drive_link)|
|`vgg16_cifar10_sub_model_3`|0.7371|[vgg16_cifar10_sub_model_3.h5](https://drive.google.com/file/d/1aH67mx3vkaGhtGDZN-rfLMqcyfXWz_Vm/view?usp=drive_link)|
|`block_switching_model`|**0.7704**|[cifar10_block_switching_model.h5](https://drive.google.com/file/d/1qUxwGrChvm6wlhXheItyWwSOKabNomDg/view?usp=drive_link)|
> Despite all submodels have same architecture learned weights are different, since weight initialization was random and hence their gradients were also in different directions. This feature makes model robust and some what powers it to defend against adversarial attacks.
# Conclusion
The parent/final Block Switching model performs better than the children/sub-models. 

## Future Work
I believe that adding more switch blocks will increase the validation accuracy, as it make the overall process more stochastic and robust. Plus make the model more defensive against any adversarial attacks. 
> **Check out [this](https://github.com/shoryasethia/Adversarial-Attack-Defence) repo, where I used Denoising AutoEncoder + Same block switch model to defend against an FGSM adversarial attack.**
> 
### Author : [@shoryasethia](https://github.com/shoryasethia)
> If you liked anything, do give this repo a star.

