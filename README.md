<img src="https://github.com/elijahmclaughlin/fitml_published/blob/main/fitml.png" alt="Alt Text" style="width:30%; height:auto;">

# FitML - Release 1.0

## How To Use  
**Step 5:** Record your exercise (ensure that there are no obstructions in your video)  
**Step 2:** Visit [FitML on Streamlit](https://fitmlpublished-sstapfypi8uyjkcmqurcim.streamlit.app/)  
**Step 3:** Select the exercise for analysis  
**Step 4:** Upload the exercise video (limited to 200MB or about 2 minutes)  
**Step 5:** Download analyzed video!  

## Initial Release
The initial release of FitML includes basic range of motion (ROM) analysis of three exercises: squat, bench press, and deadlift. The current output will overlay a pose estimation of the user. It will also overlay the calculated angle of the exercise in the top left hand corner of the uploaded video, along with the classification of the depth.

## What is FitML
FitML is a free to use web app for exercise analysis. As an avid weightlifter and gym-goer, I like to analyze my form to ensure my safety during exercising. As I don't always have a buddy with me to help, I decided to use TensorFlow's MoveNet model for Pose Estimation and OpenCV for video manipulation. Then, using the keypoints from MoveNet, FitML calculates the angle of movement during a given exercise (for example, the angle of a user's squat). After pose estimation and angle calculation, FitML uses OpenCV to overlay the estimated pose as well as the angle and range of motion classification. 

## Roadmap
**Phase 1: Deeper Analysis**
- Stability Analysis
- Velocity Analysis
- Alignment Analysis
- Form Analysis
- FitML Exercise Grading System
- Improvement suggestions
- Graph interface and analytics on Streamlit page

**Phase 2: Variants**
- Barbell variant
- Dumbbell variant
- Kettlebell variant
- Bodyweight variant
- Single/isolated exercise variant (think Bulgarian Split Squats, single arm bench press, etc.)

**Phase 3: More Exercises**
- More weightlifting exercises
- Olympic lifting exercises
- Bodybuilding exercises
- Plyometric exercises

**Phase 4: Suggestions/TBD***
- I will take user suggestions for improvement
- Anything I think of during development
