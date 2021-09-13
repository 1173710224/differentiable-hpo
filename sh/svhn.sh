# nohup python main.py --dataset svhn --model bayes >> log/svhn_bayes.log 2>&1 &
# nohup python main.py --dataset svhn --model zoopt >> log/svhn_zoopt.log 2>&1 &
# nohup python main.py --dataset svhn --model rand >> log/svhn_rand.log 2>&1 &
# nohup python main.py --dataset svhn --model dhpo_oneround >> log/svhn_dhpo_round1.log 2>&1 &
# nohup python main.py --dataset svhn --model hyper_band >> log/svhn_hyperband.log 2>&1 &
# nohup python main.py --dataset svhn --model dhpo >> log/svhn_dhpo.log 2>&1 &