Click here to read the PRIUS document

[![PDF](https://github.com/kalyan0510/RFI/blob/master/readmemedia/page1.png)](https://drive.google.com/file/d/1AxSFng_JMC7KRKC9sU7hEIzw9RUSbNne/view?usp=sharing)

#### Original signal spectogram and mitigated signal spectogram
[![](https://github.com/kalyan0510/RFI/blob/master/readmemedia/mit_comp.png)](https://drive.google.com/file/d/1AxSFng_JMC7KRKC9sU7hEIzw9RUSbNne/view?usp=sharing)


## Run

#### 1. Synthesize train data
   ``` python synthesize.py img 48 48 1 1000 ```

#### 2. MAKE THE NN LEARN THE SIMULATED IMAGES
   ```    python mitigate.py --save-model 1 --weights output/lenet_weights_kal.hdf5 -p 48 -n 20    ``` 

#### 3. CREATE/SIMULATE LARGE RF INTERFERED SPECTROGRAMS
   ```    python synthesize.py img 480 480 1 10    ``` 

#### 4. MAKE THE NN TO ESTIMATE THE FILTERS USED FOR MITIGATION
   ```    python mitigate.py --load-model 1 --weights output/lenet_weights_kal.hdf5 -p 480    ``` 
   
   RESULTS WILL BE STORED IN  ./img/R


