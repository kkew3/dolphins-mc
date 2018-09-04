# Saliency CAE

Append a sigmoid function to the end of a convolutional autoencoder to output
visual saliency. The autoencoder is trained using `MSELoss` and Lasso
regularization to encourage focusing attention to interesting regions.

## Issue

Out of Memory. Try either

- downsampling the frame; or
- shrinking the network
