# MultiViewClip
A reimplementation of Multiview-Clip for multimodal sarcasm detection

The dataset can be accessed from here: https://github.com/JoeYing1019/MMSD2.0/tree/main/data

Create a file named 'Data' in the project directory. Place the downloaded 'text_json_clean' and 'text_json_final' folders inside this newly created 'Data' folder. 
Download the image files and place them in 'Data\Image\dataset_image' folder

The file structure should look like this: 
- **Data**
  - **Image**
    - **dataset_image**
      - `682716753374351360.jpg`
      - `682721949072625664.jpg`
      - ...
  - **text_json_clean**
    - `train.json`
    - `valid.json`
    - `test.json`
  - **text_json_final**
    - `train.json`
    - `valid.json`
    - `test.json`

- `datasets.py`

- `main.py`

- `model.py`

- `predict.py`

- `train.py`


To train the model on MMSD dataset and then to test it, run the following command:  
```bash
python main.py --data_dir 'Data' --dataset 'MMSD' --dropout 0.3  --lr 3e-4 --lr_clip 3e-7 --epoch 10
```
To train the model on MMSD2.0 dataset and then to test, run the following command:  
```bash
python main.py --data_dir 'Data' --dataset 'MMSD2.0' --dropout 0.1  --lr 5e-4 --lr_clip 1e-6 --epoch 10
```



