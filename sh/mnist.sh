nohup python main.py --dataset mnist --model bayes >> log/mnist_bayes.log 2>&1 &
nohup python main.py --dataset mnist --model zoopt >> log/mnist_zoopt.log 2>&1 &
nohup python main.py --dataset mnist --model rand >> log/mnist_rand.log 2>&1 &
nohup python main.py --dataset mnist --model dhpo_oneround >> log/mnist_dhpo_round1.log 2>&1 &
nohup python main.py --dataset mnist --model hyper_band >> log/mnist_hyperband.log 2>&1 &
nohup python main.py --dataset mnist --model dhpo >> log/mnist_dhpo.log 2>&1 &


nohup python optimizers.py >> log/total.log 2>&1 &