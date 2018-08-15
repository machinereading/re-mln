# MLN-based Relation Extraction

## About

Markov Logic Newtwork(MLN)-based Relation Extraction Model. This is the re-implementation of ["Han and Sun. Global distant supervision for relation extraction. AAAI. 2016"](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12006) for Korean Dataset.

## Prerequisite
- Python 3.5+
	- numpy 1.13.0+
	- scikit-learn 0.18.2+
- [Alchemy 1.0](https://alchemy.cs.washington.edu/)

## How to use

### Configuration
1. Copy `config_sample.py` to `config.py`
2. Edit variables in `config.py` fit to your environment.
* data_path : location of data directory
* alchemy_path : location of binary file directory of Alchemy 1.0

`An example of config.py`

```python
# data directory
data_path = './data/'

# alchemy path
alchemy_path = '/home/user0/alchemy/bin/'
```

### Location of data files
- Training file : `./data/train_data`
- Test file : `./data/test_data`

### Training
```
python3 train.py
```

### Test
```
python3 test.py
```
After test script is run, you can check the result on `./data/prec_recall_per_prop.txt`

## Licenses
* `CC BY-NC-SA` [Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/2.0/)
* If you want to commercialize this resource, [please contact to us](http://mrlab.kaist.ac.kr/contact)

## Maintainer
Kijong Han `han0ah@kaist.ac.kr`

## Publisher
[Machine Reading Lab](http://mrlab.kaist.ac.kr/) @ KAIST

## Citation
- Kijong Han, Sangha Nam, Younggyun Hahm, Jiseong Kim, Jin-Dong Kim, Key-Sun Choi, "Analysis of Distant Supervision for Relation Extraction Dataset", The 17th International Semantic Web Conference (ISWC 2018), Posters and Demonstrations, 2018. 

## Acknowledgement
This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2013-0-00109, WiseKB: Big data based self-evolving knowledge base and reasoning platform)
