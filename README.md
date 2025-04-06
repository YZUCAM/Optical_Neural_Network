<div align="center">
<table>
  <tr>
    <td rowspan="2">
      <img src="https://raw.githubusercontent.com/YZUCAM/Optical_Neural_Network/main/docsrc/phase_plate2.png" width="600"/>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/YZUCAM/Optical_Neural_Network/main/docsrc/convolved_digits.png" width="300"/>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/YZUCAM/Optical_Neural_Network/main/docsrc/confusion_matrix.png" width="300"/>
    </td>
  </tr>
</table>
</div>

# Optical_Neural_Network
Optical convolutional neural network, including optimizing optical phase plate and training digital countpart. Network structure is a 2 layer network: Optical convolutional layer + small scale fully connectional layer.


## Dependencies
pytorch<br>
numpy<br>
Scipy<br>

## Training
Firstly, use train_digital_cnn.py to generate the digital convolutional kernels. The dataset can be any image dataset, such as MNIST, FASHION_MNIST, Cifar 10, and etc.<br>
After getting the digital convolutional kernels, use train_optical_cnn.py to generate the hogogram phase mask for spatial light modulator (SLM).<br>
In optical neural network, the kernel weight matrix is represented by phase mask displayed on SLM.<br>
Final step, use fine_tune_fc.py to fine tuning the last small scale fully connection layer parameters.

## Evaluating 
In notebook folder, jupyternote book is used to retrieve model parameter and evaluate each step parameters.<br>
1. digital_cnn_retrieve.ipynb used for getting digital convolutional model, checking kernel and testing model accuracy.
2. optical_cnn_retrieve.ipynb used for getting optical convolutional model, checking phase plate shape and testing model accuracy.
3. fine_tuned_model_retrieve.ipynb combining optical convolutional layer and digital fully connection layer, use it can check final results and final model accuracy.
