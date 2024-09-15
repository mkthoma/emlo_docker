# MNIST Training using Docker

This repository contains a Docker setup for training an MNIST model using PyTorch. It includes instructions for building the Docker image and running the container without caching.

## Building the docker image
Clone the repository. To build the Docker image with no cache, use the following command:

```
docker build --no-cache -t mnist-train .
```
* ***--no-cache***: Ensures that Docker does not use cached layers from previous builds, which helps to avoid potential issues with outdated dependencies.
* ***-t mnist-train***: Tags the resulting image with the name mnist-train which can be customized.

## Running the Docker Container
To run the Docker container based on the built image, use the following command:

```
docker run mnist-train
```
* ***mnist-train***: The name of the Docker image to use.

### Parameters Used
1. **--batch-size**
    * Description: Specifies the number of samples per batch to load during training.
    * Default Value: 64
    * Usage Example: ```--batch-size 128```

2. **--test-batch-size**
    * Description: Specifies the number of samples per batch during testing.
    * Default Value: 1000
    * Usage Example: ``` --test-batch-size 500 ```

3. **--epochs**
    * Description: The number of times the entire training dataset is passed through the model.
    * Default Value: 10
    * Usage Example: ```--epochs 20```

4. **--lr**
    * Description: The learning rate for the optimizer.
    * Default Value: 0.01
    * Usage Example: ```--lr 0.001```

5. **--momentum**
    * Description: Momentum factor for the Stochastic Gradient Descent (SGD) optimizer.
    * Default Value: 0.5
    * Usage Example: ```--momentum 0.9```

6. **--seed**
    * Description: Random seed for initializing the random number generator.
    * Default Value: 1
    * Usage Example: ```--seed 42```

7. **--log-interval**
    * Description: Interval (in number of batches) at which to log training status.
    * Default Value: 10
    * Usage Example: ```--log-interval 20```

8. **--cuda**
    * Description: Enables CUDA (GPU) training if available.
    * Default Value: False
    * Usage Example: ```--cuda```

9. **--dry-run**
    * Description: Quickly checks a single pass through the training process.
    * Default Value: False
    * Usage Example: ```--dry-run```

10. **--save-model**
    * Description: Saves the model checkpoint after training if specified. Model gets saved to the path model_checkpoint.pth
    * Default Value: True
    * Usage Example: ```--save-model```

11. **--resume**
    * Description: Whether to resume training from the saved model checkpoint (model_checkpoint.pth).
    * Default Value: False
    * Usage Example: ```--resume```

<br/>

To use these parameters when running train.py, include them in the command line:

```
python train.py --batch-size 128 --epochs 20 --lr 0.001 --momentum 0.9 --seed 42 --log-interval 20 --cuda --save-model
```

This command will:
* Use a batch size of 128.
* Train for 20 epochs.
* Use a learning rate of 0.001 and a momentum of 0.9.
* Set the random seed to 42.
* Log training status every 20 batches.
* Enable GPU training (if available).
* Save the model checkpoint after training.

These parameters provide fine-grained control over the training process and can be adjusted according to your needs and system capabilities.

While running the docker image you have just built with these images you can use 
```
docker run mnist-train --batch-size 128 --epochs 20 --lr 0.001 --momentum 0.9 --seed 42 --log-interval 20 --cuda
```
For saving and loading model checkpoints, you will need to mount a volume to the docker image workspace before running the command as the checkpoints will be saved inside the container and wont be accessible after it runs. You should mount the same volume or directory to the same path in the container where the checkpoint was saved. 

To save the model (model_checkpoint.pth) follow the command below:

```
docker run -it --rm -v /path/on/host:/workspace/output mnist-train --save-model
```
<br/>

Same goes for loading the model checkpoint (model_checkpoint.pth):
```
docker run -it --rm -v /path/on/host:/workspace/output mnist-train --resume
```

## Dockerfile

The Dockerfile provided in this repository uses a minimal base image (python:3.9-slim) and 
installs the required Python packages only for CPU and not GPU (torch and torchvision) without using cache. This approach helps in keeping the Docker image as small as possible.

## Additional Notes

* Ensure that the **train.py** script and any other necessary files are present in the same directory as the Dockerfile before building the image.
* If you need GPU support, you can modify the **requirements.txt** accordingly, but this will typically require a different base image and setup.
