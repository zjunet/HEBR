# HEBR
This project implements the method of hierarchical electricity-theft behavior recognition proposed in [1], which is an algorithm for detecting electricity theft.

## Testing

This project is implemented in Python 3.6

### Dependency: 

- Python 3.6. Version 3.6.2 has been tested.
- Tensorflow. Version 1.2.0 has been tested. Note that the GPU support is encouraged as it greatly boosts training efficiency.
- Other Python modules. Some other Python module dependencies are listed in ```requirements.txt```, which can be easily installed with pip ```pip install -r requirements.txt```

### Testing the Project:

``` 
python run.py
```

## Usage

Given an array of multi-source time series data, including users' electricity records, non-technical losses in transformer area, temperature records, this program can be used to estimate the electricity-theft probabilities for users.

### Input Format

The input files are expected to be four parts: 

(1) Electricity Usage Records: a CSV file, which contains daily records for every user: `#userID`, `#areaID`, `#date`,  `#total electricity`, `#top electricity`, `#on-peak electricity`, `#flat electricity`, `#off-peak electricity`. 

(2) Non-Technical Loss Records: a CSV file, which contains daily records for every area: `#areaID`, `#date`, `#cost electricity`, `#billed electricity`, `#lost electricity`. Here, the `areaID` corresponds to the `areaID` in (1)

(3) Temperature Records: a CSV file, which records daily weather, containing `#areaID`, `#data`, `#high temperature`, `#low temperature`. We can spider the data by the code in `./data_factory/temperature_spider.py`

(4) Labels of Electricity Thieves: a CSV file, which records the time that the electricity thieves are caught, containing `#userID`, `#areaID`, `#date`.

### Output Format
The program outputs to a file named ```result.csv``` which contains the results of electricity-theft probabilities   estimated by HEBR.
### Main Script
The help of main script can be obtained by excuting command:
```
python run.py -h
```
## Reference
[1] Wenjie, H; Yang, Y; Jianbo, W; Xuanwen, H and Ziqiang, C, 2020, [Understanding Electricity-Theft Behavior via Multi-Source Data](), In WWW, 2020 

```
 @inproceedings{hu2020theft, 
    title={Understanding Electricity-Theft Behavior via Multi-Source Data},
    author={Wenjie Hu and Yang Yang and Jianbo Wang and Xuanwen Huang and Ziqiang Cheng},
    booktitle={Proceedings of WWW},
    year={2020}
    }
```

