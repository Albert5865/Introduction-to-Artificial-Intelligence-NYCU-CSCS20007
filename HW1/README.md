# Face Detection Project
This repository contains an implementation of a face detection system based on the Viola-Jones algorithm with experiments on the data_small and FDDB datasets. The project demonstrates key concepts in classical face detection, boosting classifiers, and evaluating detection performance.


#### Data folder structue:
	├── data/                 # Place datasets here
	│   ├── data_small/       
	│   └── data_FDDB/
	├── src/                  # Source code (implementation of Viola-Jones & classifiers)
	├── results/              # Output images and logs from experiments
	├── report.pdf            # Detailed report with analysis and answers
	└── README.md             # Project documentation


## Result & Analysis

 ### Data_small:
•	Model accuracy improves with more training iterations.
•	High true detection rate but also high false positive rate.
•	Good at recognizing actual faces, showing robustness to facial features.

 ### Data_FDDB:
•	Accuracy improvements were limited compared to data_small.
•	Higher false positive rate and difficulty generalizing to complex images.

 ### Bonus Implementation (Part 6):
•	Explored error calculations and classifier selection using list comprehensions.
•	Performance did not reach the same level as the original classifier, especially for FDDB.
 
 ### Limitations of Viola-Jones:
•	Sensitive to lighting variations
•	Struggles with occlusion
•	Limited scale adaptability
•	Susceptible to false positives

![result_Oppenheimer](https://github.com/user-attachments/assets/6a0d3b10-cda8-4c37-9f3e-808adb12c051)

## Reference
•	Viola, P. and Jones, M. (2001). Rapid Object Detection using a Boosted Cascade of Simple Features.
•	FDDB: Face Detection Data Set and Benchmark.
