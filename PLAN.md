- Model: Code in Python (All of us)
  - Chetan will start
- Frontend
  - Looks (Shreehan)
  - Logic (Hamid)
- Backend Logic (Hamid)
  
- After a random number of failed attempts it sends the notification.
- Push notifications to phone or watch giving an alert when unauthorized user is attempting to login to your profile/when   it's unsble to realize it's you.
- The data input collection part is done, sending the data is what's not done yet

Model notes:
We can use a CNN-GRU hybrid (idea from Gemini).
The data we take in:
- Relative (x,y) (Distance to target)
- Velocity (Including Direction - out of 360 angle)
- Acceleration
- Jerk
- Click / Not Click
- Timestamp (Time from Start)
- 60 Times Per Second

If we have trouble with training speed or model size or quality, we can try a few things:
If we have trouble with GRU we can try a transformer

Filter out sections that are uneven sampling rate
Try higher sampling rates
We can take data before authentication failures and when correct password is entered add it to training data to catch variations like tired, coffee, etc.
Requirements (from Gemini, following AAL2 and Android biometric standards. we are trying to be on par with things like phone fingerprint sensors which follow these.)
False Acceptance Rate (FAR): 1 in 50k
False Rejection Rate (FRR): 1 in 10
Can't be fooled by data replay: Random buttons mitigates this
Must be tied to "something you have" aka. a specific auth module in a device: This is a prototype but can be easily implemented with IOMMU (to ensure only physical mice can unlock, not software) and isolated auth modules on most devices.
Hyperparams to tune in a loop:
Epochs we can just use early stopping with val loss (Gemini's idea)
Amount of training data
Acceptance threshold
Order to tune (nested loops):
1. Pick a number of epochs
2. Pick amount of training data
Early stop with val loss for epochs
Tune hyperparams for val loss
3. Tune acceptance threshold
Find model that maximizes:
False Acceptance Rate
False Rejection Rate
We filter to models under 10% FRR first, then find the one with the lowest FAR. If no models under 10% FRR, then the loop tries to minimize FRR.
Make sure to store stats on every single test case of each trained model (epochs, training data, threshold, far, frr) so after we get our theoretical "best" model we can analyze this data to see trends and find better ways to tune further. like if it decides that max trainign data is best, then we can see what the best is with lower training data.
We will need to ask many people to use our website, and we will need user IDs. so we can collect a baseline for the model to train on.
