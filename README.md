My project during the learning program named "Automotive Software Engineering", hosted by Continental Iasi, I have developed a lane detection algorithm using computer vision in order to delimit traffic lanes.

I used cv2 library to work with video's frames and numpy library to work with pixels flows.

There were many steps, such as in the photo below: 
1) Resizing the frame
2) Converting the frame to grayscale
3) Creating a trapezoidal mask
4) Applying the trapezoidal mask
5) Getting a top-down view
6) Applying the Gaussian blur to reduce noise
7) Using sobel filters(vertical and horizontal)
8) Binarizing the frame
9) Removing the noise from the edges
10) Drawing the detected lane lines on the binarized frame
11) Transforming lane lines back to the original perspective
![hatzul](https://github.com/user-attachments/assets/7ead5d03-7629-4381-b0d4-f23fc4a67a83)
