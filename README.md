# DSTGCN
This is the repository of our paper "DSTGCN: Dynamic Spatial-Temporal Graph Convolution Network for ElectricBicycle Flow Forecasting" and it is implemented in Pytorch. Some baselines are also implemented in this repository.

## Requirements

* matplotlib			3.3.3
* numpy					1.18.5
* pandas				1.2.2
* scikit-learn			1.0.1
* scipy					1.6.0
* statsmodels			0.13.0
* torch					1.2.0
* tqdm					4.62.3

## Baselines & How to run
|  baseline   |   |
|  ----  | ----  |
| HA  | python ARIMA/Arima_Main.py |
| ARIMA  | python HA/HA_main.py |
| SVR  | python SVR/SVR_main.py |
| LightGBM  | python Lightgbm/Lightgbm_main.py |
| STGCN  | python train_zhengzhou.py |
| ASTGCN  | python train_zhengzhou.py |

all results will be printed to console












