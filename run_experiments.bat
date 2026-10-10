conda activate lfrl
python train.py --signal self_certainty --domain math --max_samples 500
python train.py --signal self_certainty --domain logic --max_samples 500
python train.py --signal self_certainty --domain code --max_samples 500
python train.py --signal entropy --domain math --max_samples 500
python train.py --signal entropy --domain logic --max_samples 500
python train.py --signal entropy --domain code --max_samples 500
python train.py --signal majority_voting --domain math --max_samples 500
python train.py --signal majority_voting --domain logic --max_samples 500
python train.py --signal majority_voting --domain code --max_samples 500
