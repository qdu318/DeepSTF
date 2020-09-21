# Deep Statial-Temporal Forecast Model of Traffic Flow (DeepSTF)

 ## Data Set
 https://outreach.didichuxing.com/app-vue/HaiKou?id=999
 ## Requirements
  
  * Torch == 1.2.0
  * TorchVision==0.4.0
  * NumPy == 1.18.1
  * Python == 3.7
  
 ## install
 `pip install -r requirements.txt`
 
 ## Run
 `python3 main.py`
 ## Directory Structure
 ~~~~
├── data                     // Data set
├── model            
     ├── DeepSTF.py           // The model of DeepSTF
     ├── TCN.py               // The model of TCN
├── process                  
     ├── train.py             // Train model
     ├── evaluate.py          // Evalute model
├── utils
     ├── data_load.py         // Load data set 
     ├── utils.py             // Custom method
├── main.py                  // Main method
├── config.json              // Config information
├── requirements.txt         // Project interpreter
├── README.md                // Introduction
