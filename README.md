# Identity-Verification-via-Machine-Learning
User Identity Verification via Machine Learning, written in Python as a final year dissertation project. I recieved a first for this work.


## Software Instructions
### Recording new data
To record mouse and keyboard data, open the rawKeyMouseLogger.py script and enter a name, once then enter button has been pressed, every action taken by the user is recorded into a csv file. To conclude recording of the input the escape button is pressed. If the new addition is to become training data, the new addition to the userbase should then be added to the users.csv file located in the /Data/ folder and the csv file created should be moved into the /Data/Raw/ folder. If the recorded data is done for testing purposes, then the csv file should be moved into the /Data/Test/ folder.

### Creating machine learning models
Once the users.csv file reflects all the users that have taken part, the rawToReadable.py script is ran, this will convert each of the raw csv file for each user within the /Data/Raw/ folder into two separate files in the /Data/Training/ folder. One for the mouse values and the other for the keyboard values. Once the script has converted all of the raw files, the next step is to open up the buildTraining.py script. This will process all of the data found within the /Data/Training/ folder, and build the machine learning models that will be saved within the /Data/Models/ folder. Once completed (more than a 15-minute task on my workstation), the machine learning models are ready to be tested.

### Evaluating performance of models
Two scripts exist for testing the performance of the models, the first being predictRawUser.py. This script will analyse a raw data file located in the /Data/Test/ folder and output a sorted percentage match to the models created earlier. The last script for testing the performance of the models is the predictLoggerUser.py script. This script is ran and records the information of the user, the user then goes on and completes the task, once concluded, the escape button is to be pressed and the script will output a sorted percentage match to the models created earlier.
