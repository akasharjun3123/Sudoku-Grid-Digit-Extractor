# Sudoku-Grid-Digit-Extractor

Aim: To extract the digits out of an image of unsolved sudoku puzzle for further processing and solving the puzzle. 

Tools & Technologies: TensorFlow, Keras, OpenCV, Deep Learning, Convolutional Neural Networks, Python, Pickle, Numpy, Matplotlib, etc.

Theory:

    1.Preprocessing the Image:

          1. Prepares the image by resizing.
          2. Preprocesses the image by converting it to grayscale, applying Gaussian blur, and thresholding.
          3. Finds contours in the preprocessed image and identifies the largest contour.
          4. Reorders the points of the largest contour and warps the perspective to obtain a bird's-eye view of the Sudoku grid.
          5. Final images will go through futher processing and predictions

![preProcessedImage3](https://github.com/akasharjun3123/Sudoku-Grid-Digit-Extractor/assets/139098586/51128312-061d-4ba8-bc3b-2edef6998071)
   

    2. Training and Saving the model 

            1. Loading MNIST dataset of handwritten digit images and labels. 
            2. Normalizing pixel values to a range of [0, 1]. 
            3. Calculate class frequencies of each digit class. 
            4. Adjusting the size of image dimensions for CNN compatibility (28*28). 
            5. Augmenting the training data which applies transformations to increase dataset diversity. 
            6. Constructing a CNN model with convolutional and pooling layers. 
            7. Compiling the model with Specifies optimizer, loss function, and metrics. 
            8. Trains the model with augmented data for 5 or user selected epochs. 
            9. Saves the trained model to a file. 
            10. Calculates test accuracy and making predictions 
![TestingImages1](https://github.com/akasharjun3123/Sudoku-Grid-Digit-Extractor/assets/139098586/b70c73fc-0587-4a0b-88c1-5fc6a110239b)

    3. Processing the Extracted Sudoku Grid
            1. Splits the Sudoku grid into individual digit boxes.
            2. Processes the digit boxes by cropping borders and reshaping them.
            3. Applies filters on the digit boxes for prediction using a trained model.
            4. Loads a pre-trained model for digit recognition.
            5. Makes predictions using the trained model.
            6. Processes the predicted digits and displays them visually.
            7. Prints and stores the predicted digits and their corresponding prediction accuracies.

![extractedDigits3](https://github.com/akasharjun3123/Sudoku-Grid-Digit-Extractor/assets/139098586/8f15b067-65c2-4dc6-a5ae-8d15fe8146ef)
![predictedDigits3](https://github.com/akasharjun3123/Sudoku-Grid-Digit-Extractor/assets/139098586/e2b7bcc9-9db2-4f7f-9f79-1ffe39021c3e)












