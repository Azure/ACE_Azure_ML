# Overview

Post on [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:6512538611181846528) (includes video demonstration)

# Pre-requisites

## Skills

## Software Dependencies

vs code (optional)
conda
conda environement

## Hardware Dependencies

Standard NC6 sufficient, faster learning with NC6_v2/3 or ND6

## Dataset

[UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)

## Agenda

### Getting Started

1. [Data Preparation](./docs/data_prep_w_pillow.md)
2. [Model Development](./docs/model_development.md)
3. [Hyperparameter Tuning](./docs/hyperparameter_tuning.md)
4. [Anomaly Detection](./docs/anomaly_detection.md)
5. [Deployment](./docs/deployment.md)

### Advanced Topics (coming soon)

1. Transfer learning
1. Supervised Anomaly Detection

## References / Resources

- Research Article: [Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104) by Lotter, W., Kreiman, G. and Cox, D., 2016.

	```
	@article{lotter2016deep,
	title={Deep predictive coding networks for video prediction and unsupervised learning},
	author={Lotter, William and Kreiman, Gabriel and Cox, David},
	journal={arXiv preprint arXiv:1605.08104},
	year={2016}
	}
	```
- Original Prednet implentation is on [github.com](https://coxlab.github.io/prednet/). Note, that the original implementation will only work in Python 2, but not in Python 3.

- Interesting blog post on [Self-Supervised Video Anomaly Detection](https://launchpad.ai/blog/video-anomaly-detection) by [Steve Shimozaki](https://launchpad.ai/blog?author=590f381c3e00bed4273e304b) 
