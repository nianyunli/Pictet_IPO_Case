
for seed in {1..30}
do 
    python classfication.py --outcome high_return --model xgboost --seed $seed
done


for seed in {1..30}
do 
    python regression.py --outcome '1st_Day_Return' --model xgboost --seed $seed
done