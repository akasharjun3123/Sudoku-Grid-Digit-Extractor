# Sudoku-Grid-Digit-Extractor

Aim: To extract the digits out of an image of unsolved sudoku puzzle for further processing and solving the puzzle. 

Tools & Technologies: TensorFlow, Keras, OpenCV, Deep Learning, Convolutional Neural Networks, Python, Pickle, Numpy, Matplotlib, etc.

Theory:

    1.Preprocessing the Image:

          1. Prepares the image by resizing.
          2. Preprocesses the image by converting it to grayscale, applying Gaussian blur, and thresholding.
          3. Finds contours in the preprocessed image and identifies the largest contour.
          4. Reorders the points of the largest contour and warps the perspective to obtain a bird's-eye view of the Sudoku grid.
          5. Splits the Sudoku grid into individual digit boxes.
          6. Processes the digit boxes by cropping borders and reshaping them.
          7. Applies filters on the digit boxes for prediction using a trained model.
          8. Loads a pre-trained model for digit recognition.
          9. Makes predictions using the trained model.
          10. Processes the predicted digits and displays them visually.
          11. Prints and stores the predicted digits and their corresponding prediction accuracies.

![image](https://github.com/akasharjun3123/Sudoku-Grid-Digit-Extractor/assets/139098586/45c05d23-311c-4403-a78c-710347b3f152)
![image](https://github.com/akasharjun3123/Sudoku-Grid-Digit-Extractor/assets/139098586/674d7e4f-57e5-46db-8ae8-efd2565b7941)











